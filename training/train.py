import hydra
from omegaconf import DictConfig
from lightning import Trainer
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
import torch
import os


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    # 1. 实例化模型和数据
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # 2. 实例化训练器
    # 确保 cfg.trainer 中包含了 checkpoint 回调
    trainer: Trainer = instantiate(cfg.trainer)

    # 3. 开始训练
    trainer.fit(model, datamodule)

    # --- 训练后处理 ---

    # 4. 手动保存权重 (除了 Checkpoint，有时我们需要保存干净的 state_dict)
    # 因为你用了 PyTorchModelHubMixin，可以保存为 HuggingFace 格式
    if trainer.is_global_zero:  # 确保只在主进程保存
        save_path = os.path.join(trainer.default_root_dir, "final_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model state_dict saved to {save_path}")

    # 5. 开启评估模式并测试
    if cfg.get("test_after_train", False):
        print("Starting testing...")

        # 自动加载训练过程中表现最好的 Checkpoint
        # Lightning 会自动将模型设为 eval 模式 (model.eval())

        # 特别注意：如果你在 stage 0 训练，但想在 stage 1 测试
        # model.stage = 1

        trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()