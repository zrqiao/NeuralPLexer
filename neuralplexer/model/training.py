"""Training warppers and task heads"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from neuralplexer.data.dataloading import BindingDataModule, MolPropDataModule
from neuralplexer.model.mht_encoder import (MolPretrainingWrapper,
                                            MolPropRegressor)
from neuralplexer.model.wrappers import NeuralPlexer


def setup_pl_model(config):
    if config.task.task_type == "graph_regression":
        model = MolPropRegressor(config=config)
    elif config.task.task_type == "geometry_pretraining":
        if config.pretrained_path is not None:
            model = MolPretrainingWrapper.load_from_checkpoint(
                config=config, checkpoint_path=config.pretrained_path, strict=False
            )
        else:
            model = MolPretrainingWrapper(config=config)
    elif config.task.task_type == "protein_pretraining":
        if config.pretrained_path is not None:
            model = NeuralPlexer.load_from_checkpoint(
                config=config, checkpoint_path=config.pretrained_path, strict=False
            )
        else:
            model = NeuralPlexer(config=config)
    elif config.task.task_type in [
        "LBA",
        "all_atom_prediction",
    ]:
        if config.pretrained_path is not None:
            model = NeuralPlexer.load_from_checkpoint(
                config=config, checkpoint_path=config.pretrained_path, strict=False
            )
        else:
            model = NeuralPlexer(config=config)
    else:
        raise NotImplementedError
    return model


def setup_pl_trainer(config, gpus=1):
    if config.task.task_type == "graph_regression":
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor="val_loss", mode="min"
        )
    elif config.task.task_type == "geometry_pretraining":
        checkpoint_callback = ModelCheckpoint(
            save_last=True, save_top_k=1, monitor="train_loss", mode="min"
        )
    elif config.task.task_type in [
        "all_atom_prediction",
    ]:
        if config.task.use_ema:
            from neuralplexer.util.ema import EMAModelCheckpoint

            checkpoint_callback = EMAModelCheckpoint(
                save_last=True, save_top_k=1, monitor="val/loss", mode="min"
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                save_last=True, save_top_k=1, monitor="val/loss", mode="min"
            )
    elif config.task.task_type in [
        "LBA",
        "protein_pretraining",
    ]:
        checkpoint_callback = ModelCheckpoint(
            save_last=True, save_top_k=1, monitor="val_loss", mode="min"
        )
    else:
        raise NotImplementedError
    lr_monitor = LearningRateMonitor(logging_interval=None)
    wandb_logger = WandbLogger(
        save_dir="wandb_runs",
        project=config.project_name,
        name=config.run_name,
        log_model=True,
    )
    callbacks = [lr_monitor, checkpoint_callback]
    if config.task.use_ema:
        from neuralplexer.util.ema import EMA

        ema_callback = EMA(decay=0.999, every_n_steps=4)
        callbacks.append(ema_callback)

    if config.task.task_type in [
        "geometry_pretraining",
        "protein_pretraining",
        "all_atom_prediction",
    ]:
        n_reload = 1
    else:
        n_reload = 0
    return pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=gpus,
        auto_select_gpus=True,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=config.task.max_epoch,
        reload_dataloaders_every_n_epochs=n_reload,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # accumulate_grad_batches=4,
        # detect_anomaly=True,
        amp_backend="native",
        # amp_level="O1",
    )


def setup_pl_datamodule(config, model=None):
    if config.task.task_type in ["graph_regression", "geometry_pretraining"]:
        return MolPropDataModule(config)
    elif config.task.task_type in [
        "LBA",
        "protein_pretraining",
        "all_atom_prediction",
    ]:
        return BindingDataModule(config, model=model)
    else:
        raise NotImplementedError
