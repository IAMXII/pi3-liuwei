# import os
# import torch
# from torch.optim.lr_scheduler import OneCycleLR
# from lightning import LightningModule
# from loss import compute_losses, compute_loss_sky
# from safetensors.torch import load_file as load_safetensors
#
# from training.utils.logger import RankedLogger
# from training.utils.scheduler import MultiLinearWarmupCosineAnnealingLR
# from training.utils.misc import (
#     compose_batches_from_list,
#     convert_defaultdict_to_dict,
#     deep_merge_dict,
# )
#
#
# class Pi3LightningModule(LightningModule):
#
#     def __init__(
#         self,
#         stage1_epochs=80,
#         stage2_epochs=80,
#         iters_per_epoch=800,
#         lr_encoder=5e-6,
#         lr_rest=5e-5,
#         lr_conf=5e-5,
#         grad_clip=1.0,
#         train_sky=False,
#         pretrained_vggt_checkpoint=None,
#         freeze_encoder=True,
#     ):
#         """
#         完整覆盖 prepare_model_for_training 的功能：
#         - 加载预训练
#         - 冻结 encoder
#         - 自动分组参数
#         """
#
#         super().__init__()
#         from pi3.models.pi3 import Pi3
#         self.model = Pi3()
#
#         self.train_sky = train_sky
#         self.stage1_epochs = stage1_epochs
#         self.stage2_epochs = stage2_epochs
#         self.iters_per_epoch = iters_per_epoch
#         self.lr_encoder = lr_encoder
#         self.lr_rest = lr_rest
#         self.lr_conf = lr_conf
#         self.grad_clip = grad_clip
#
#         # 保存
#         self.save_hyperparameters()
#
#         # ★★★★★ 关键：加载预训练 + 冻结 + 分组
#         (
#             self.encoder_params,
#             self.rest_params,
#             self.conf_params
#         ) = self.prepare_model_for_training(
#             self.model,
#             pretrained_vggt_checkpoint,
#             freeze_encoder
#         )
#
#     # ------------------------------------------------------------------
#     # prepare_model_for_training —— 加载预训练模型
#     # ------------------------------------------------------------------
#     def prepare_model_for_training(self, model, pretrained_vggt_checkpoint, freeze_encoder):
#
#         # -------- 1. 加载预训练权重 --------
#         if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
#             ckpt = torch.load(pretrained_vggt_checkpoint, map_location="cpu")
#             try:
#                 model_state = model.state_dict()
#                 load_state = {k: v for k, v in ckpt.items() if k in model_state}
#                 model_state.update(load_state)
#                 model.load_state_dict(model_state)
#                 print(f"Loaded partial weights from {pretrained_vggt_checkpoint}")
#             except Exception as e:
#                 print("Warning: loading checkpoint failed:", e)
#
#         # -------- 2. 分组 encoder / rest / conf，同时可冻结 --------
#         encoder_params = []
#         rest_params = []
#         conf_params = []
#
#         for name, p in model.named_parameters():
#
#             # encoder 参数
#             if "encoder" in name:
#                 encoder_params.append(p)
#                 if freeze_encoder:
#                     p.requires_grad = False
#
#             # conf head
#             elif "conf" in name or "confidence" in name:
#                 conf_params.append(p)
#
#             else:
#                 rest_params.append(p)
#
#         return encoder_params, rest_params, conf_params
#
#     # ------------------------------------------------------------------
#     # Optimizer
#     # ------------------------------------------------------------------
#     def configure_optimizers(self):
#
#         param_groups = []
#
#         # encoder（可能被冻结）
#         if len(self.encoder_params) > 0:
#             param_groups.append({"params": self.encoder_params, "lr": self.lr_encoder})
#
#         # 其它
#         if len(self.rest_params) > 0:
#             param_groups.append({"params": self.rest_params, "lr": self.lr_rest})
#
#         # conf head
#         if len(self.conf_params) > 0:
#             param_groups.append({"params": self.conf_params, "lr": self.lr_conf})
#
#         optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
#
#         # scheduler steps
#         total_steps = (self.stage1_epochs + self.stage2_epochs) * self.iters_per_epoch
#
#         scheduler = OneCycleLR(
#             optimizer,
#             max_lr=[g["lr"] for g in param_groups],
#             total_steps=total_steps,
#             pct_start=0.1,
#             anneal_strategy="cos",
#             div_factor=10.0,
#             final_div_factor=1e4,
#         )
#
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
#         }
#
#     # ------------------------------------------------------------------
#     # Training Step
#     # ------------------------------------------------------------------
#     def training_step(self, batch, batch_idx):
#         batched_inputs = compose_batches_from_list(batch, device=self.device)
#         # imgs, gt = batch
#         res = self.model(batched_inputs)
#
#         if self.train_sky:
#             loss = compute_loss_sky(res, batched_inputs)
#             self.log("loss_sky", loss, prog_bar=True)
#             return loss
#         else:
#             losses = compute_losses(res, batched_inputs, self.device)
#             loss = losses["total_loss"]
#             self.log("loss", loss, prog_bar=True)
#             return loss
#
#     # ------------------------------------------------------------------
#     # Lightning 自动梯度裁剪
#     # ------------------------------------------------------------------
#     def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None):
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
import os
import torch
from torch.optim.lr_scheduler import OneCycleLR
from lightning import LightningModule
from loss import compute_losses, compute_loss_sky
from torch.optim import AdamW
from training.utils.misc import compose_batches_from_list


class Pi3LightningModule(LightningModule):

    def __init__(
        self,
        current_stage: int = 0,     # ★ 新增：显式指定训练阶段 (0 / 1 / 2)
        stage0_epochs=10,
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
        super().__init__()

        assert current_stage in [0, 1, 2], \
            f"current_stage must be 0, 1 or 2, got {current_stage}"

        from pi3.models.pi3 import Pi3
        self.model = Pi3()

        # -------------------------
        # Stage config
        # -------------------------
        self.current_stage = current_stage
        self.stage0_epochs = stage0_epochs
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

        self.iters_per_epoch = iters_per_epoch
        self.grad_clip = grad_clip
        self.train_sky = train_sky

        self.lr_encoder = lr_encoder
        self.lr_rest = lr_rest
        self.lr_conf = lr_conf

        self.save_hyperparameters()

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
    # Load pretrained + parameter grouping
    # ------------------------------------------------------------------
    # def prepare_model_for_training(
    #     self,
    #     model,
    #     pretrained_vggt_checkpoint,
    #     freeze_encoder,
    # ):
    #     if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
    #         ckpt = torch.load(pretrained_vggt_checkpoint, map_location="cpu")
    #         model_state = model.state_dict()
    #         keys_to_remove = []
    #         for k in model_state.keys():
    #             if "fc_rot" in k:  # 旧 9D rotation head
    #                 keys_to_remove.append(k)

    #         for k in keys_to_remove:
    #             print(f"Removing old rotation weight: {k}")
    #             del model_state[k]
    #         load_state = {k: v for k, v in ckpt.items() if k in model_state}
    #         model_state.update(load_state)
    #         model.load_state_dict(model_state)
    #         print(f"[Pi3] Loaded pretrained weights from {pretrained_vggt_checkpoint}")

    #     encoder_params, rest_params, conf_params = [], [], []

    #     for name, p in model.named_parameters():
    #         if "encoder" in name:
    #             encoder_params.append(p)
    #             p.requires_grad = not freeze_encoder
    #         elif "conf" in name or "confidence" in name:
    #             conf_params.append(p)
    #         else:
    #             rest_params.append(p)

    #     return encoder_params, rest_params, conf_params
    def prepare_model_for_training(
        self,
        model,
        pretrained_vggt_checkpoint,
        freeze_encoder,
    ):
        """
        Load pretrained weights safely and robustly.
        - Supports both .pt/.pth and .safetensors
        - Drops old 9D rotation head (fc_rot)
        - Keeps backbone / MLP / translation head
        """

        if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
            print(f"[Pi3] Loading pretrained weights from {pretrained_vggt_checkpoint}")

            # ------------------------------------------------------------
            # 1. Load checkpoint (safe)
            # ------------------------------------------------------------
            if pretrained_vggt_checkpoint.endswith(".safetensors"):
                from safetensors.torch import load_file
                ckpt_state = load_file(pretrained_vggt_checkpoint, device="cpu")
            else:
                # Future-proof & safe
                ckpt = torch.load(
                    pretrained_vggt_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                # 兼容 {"model": state_dict} / 直接 state_dict
                ckpt_state = ckpt.get("model", ckpt)

            model_state = model.state_dict()

            # ------------------------------------------------------------
            # 2. Drop old rotation head weights (9D -> 6D)
            # ------------------------------------------------------------
            filtered_ckpt = {}
            for k, v in ckpt_state.items():
                # 明确丢弃旧的 9D rotation head
                if "fc_rot" in k and "fc_rot_6d" not in k:
                    print(f"[Pi3] Skip old rotation weight: {k}")
                    continue
                # 只加载当前模型中存在的 key
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_ckpt[k] = v

            # ------------------------------------------------------------
            # 3. Load with strict=False (expected missing fc_rot_6d)
            # ------------------------------------------------------------
            missing, unexpected = model.load_state_dict(
                filtered_ckpt,
                strict=False
            )

            print("[Pi3] Pretrained loading summary:")
            print("  Missing keys:")
            for k in missing:
                print("   -", k)
            print("  Unexpected keys:")
            for k in unexpected:
                print("   -", k)

        # ------------------------------------------------------------
        # 4. Parameter groups (unchanged logic)
        # ------------------------------------------------------------
        encoder_params, rest_params, conf_params = [], [], []

        for name, p in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(p)
                p.requires_grad = not freeze_encoder
            elif "conf" in name or "confidence" in name:
                conf_params.append(p)
            else:
                rest_params.append(p)

        return encoder_params, rest_params, conf_params

    # ------------------------------------------------------------------
    # Training start hook (freeze / unfreeze encoder)
    # ------------------------------------------------------------------
    def on_train_start(self):
        if self.current_stage < 2:
            for p in self.encoder_params:
                p.requires_grad = False
        else:
            for p in self.encoder_params:
                p.requires_grad = True

        self.log("train_stage", float(self.current_stage), prog_bar=True)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):

        param_groups = []

        if len(self.encoder_params) > 0:
            param_groups.append({
                "params": self.encoder_params,
                "lr": self.lr_encoder
            })

        if len(self.rest_params) > 0:
            param_groups.append({
                "params": self.rest_params,
                "lr": self.lr_rest
            })

        if len(self.conf_params) > 0:
            param_groups.append({
                "params": self.conf_params,
                "lr": self.lr_conf
            })

        optimizer = AdamW(
            param_groups,
            weight_decay=1e-2
        )

        # 根据 stage 决定训练总步数
        if self.current_stage == 0:
            total_epochs = self.stage0_epochs
        elif self.current_stage == 1:
            total_epochs = self.stage1_epochs
        else:
            total_epochs = self.stage2_epochs

        total_steps = total_epochs * self.iters_per_epoch

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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):

        batched_inputs = compose_batches_from_list(
            batch,
            device=self.device
        )

        res = self.model(batched_inputs)

        # -------- Stage 0: pose warmup --------
        if self.current_stage == 0:
            losses = compute_losses(
                res,
                batched_inputs,
                self.device,
                pose_only=True,
                pose_weight=1.0,
            )

        # -------- Stage 1: pose + gaussian --------
        elif self.current_stage == 1:
            losses = compute_losses(
                res,
                batched_inputs,
                self.device,
                pose_only=False,
                pose_weight=1.0,
            )

        # -------- Stage 2: full finetune --------
        else:
            losses = compute_losses(
                res,
                batched_inputs,
                self.device,
                pose_only=False,
                pose_weight=0.3,
            )

        loss = losses["total_loss"]

        self.log_dict(
            {f"loss/{k}": v for k, v in losses.items()},
            prog_bar=True,
            on_step=True,
        )

        return loss

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: float,
        gradient_clip_algorithm: str = "norm",
    ):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )