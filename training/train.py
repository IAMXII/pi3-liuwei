import hydra
from omegaconf import DictConfig
from lightning import Trainer
from hydra.utils import instantiate
from lightning.pytorch import seed_everything

@hydra.main(config_path="configs", config_name="train", version_base=1.3)
def main(cfg: DictConfig):

    seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer: Trainer = instantiate(cfg.trainer)

    trainer.fit(model, datamodule)

    if cfg.get("test_after_train", False):
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()