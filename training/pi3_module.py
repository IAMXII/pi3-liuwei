import os
import torch
from torch.optim.lr_scheduler import OneCycleLR
from lightning import LightningModule
from loss import compute_losses, compute_loss_sky
from safetensors.torch import load_file as load_safetensors

from training.utils.logger import RankedLogger
from training.utils.scheduler import MultiLinearWarmupCosineAnnealingLR
from training.utils.misc import (
    compose_batches_from_list,
    convert_defaultdict_to_dict,
    deep_merge_dict,
)


class Pi3LightningModule(LightningModule):

    def __init__(
        self,
        stage1_epochs=80,
        stage2_epochs=80,
        iters_per_epoch=800,
        lr_encoder=5e-6,
        lr_rest=5e-5,
        lr_conf=5e-5,
        grad_clip=1.0,
        train_sky=False,
        pretrained_vggt_checkpoint=None,
        freeze_encoder=True,
    ):
        """
        完整覆盖 prepare_model_for_training 的功能：
        - 加载预训练
        - 冻结 encoder
        - 自动分组参数
        """

        super().__init__()
        from pi3.models.pi3 import Pi3
        self.model = Pi3()

        self.train_sky = train_sky
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.iters_per_epoch = iters_per_epoch
        self.lr_encoder = lr_encoder
        self.lr_rest = lr_rest
        self.lr_conf = lr_conf
        self.grad_clip = grad_clip

        # 保存
        self.save_hyperparameters()

        # ★★★★★ 关键：加载预训练 + 冻结 + 分组
        (
            self.encoder_params,
            self.rest_params,
            self.conf_params
        ) = self.prepare_model_for_training(
            self.model,
            pretrained_vggt_checkpoint,
            freeze_encoder
        )

    # ------------------------------------------------------------------
    # prepare_model_for_training —— 加载预训练模型
    # ------------------------------------------------------------------
    def prepare_model_for_training(self, model, pretrained_vggt_checkpoint, freeze_encoder):

        # -------- 1. 加载预训练权重 --------
        if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
            ckpt = torch.load(pretrained_vggt_checkpoint, map_location="cpu")
            try:
                model_state = model.state_dict()
                load_state = {k: v for k, v in ckpt.items() if k in model_state}
                model_state.update(load_state)
                model.load_state_dict(model_state)
                print(f"Loaded partial weights from {pretrained_vggt_checkpoint}")
            except Exception as e:
                print("Warning: loading checkpoint failed:", e)

        # -------- 2. 分组 encoder / rest / conf，同时可冻结 --------
        encoder_params = []
        rest_params = []
        conf_params = []

        for name, p in model.named_parameters():

            # encoder 参数
            if "encoder" in name:
                encoder_params.append(p)
                if freeze_encoder:
                    p.requires_grad = False

            # conf head
            elif "conf" in name or "confidence" in name:
                conf_params.append(p)

            else:
                rest_params.append(p)

        return encoder_params, rest_params, conf_params

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):

        param_groups = []

        # encoder（可能被冻结）
        if len(self.encoder_params) > 0:
            param_groups.append({"params": self.encoder_params, "lr": self.lr_encoder})

        # 其它
        if len(self.rest_params) > 0:
            param_groups.append({"params": self.rest_params, "lr": self.lr_rest})

        # conf head
        if len(self.conf_params) > 0:
            param_groups.append({"params": self.conf_params, "lr": self.lr_conf})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

        # scheduler steps
        total_steps = (self.stage1_epochs + self.stage2_epochs) * self.iters_per_epoch

        scheduler = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        batched_inputs = compose_batches_from_list(batch, device=self.device)
        # imgs, gt = batch
        res = self.model(batched_inputs)

        if self.train_sky:
            loss = compute_loss_sky(res, batched_inputs)
            self.log("loss_sky", loss, prog_bar=True)
            return loss
        else:
            losses = compute_losses(res, batched_inputs, self.device)
            loss = losses["total_loss"]
            self.log("loss", loss, prog_bar=True)
            return loss

    # ------------------------------------------------------------------
    # Lightning 自动梯度裁剪
    # ------------------------------------------------------------------
    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)