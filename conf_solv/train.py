from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.profiler import PyTorchProfiler
from conf_solv.dataloaders.loader import SolventData3DModule
from conf_solv.trainer import LitConfSolvModule
import sys

import os
from argparse import ArgumentParser

if 'linux' in sys.platform:
    import resource  # for ancdata error
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def train_conf_solv(config):
    seed_everything(config["seed"], workers=True)
    solvation_data = SolventData3DModule(config)
    config["node_dim"] = solvation_data.node_dim
    config["edge_dim"] = solvation_data.edge_dim
    model = LitConfSolvModule(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["log_dir"],
        filename='best_model',
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=config["log_dir"],
        filename='latest_model_{epoch}',
        monitor="epoch",
        mode="max",
        save_top_k=-1,
        save_weights_only=False
    )
    nan_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        check_finite=True,
    )
    neptune_logger = NeptuneLogger(
        project="lagnajit/conf-solv",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        tags=[],
        mode="offline",
        source_files=["conf_solv/*/*.py"],
    )
    try:
        neptune_logger.run
    except Exception:
        pass
    trainer = pl.Trainer(
        logger=neptune_logger,
        default_root_dir=config["log_dir"],
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=[LearningRateMonitor(),
                   checkpoint_callback,
                   latest_checkpoint_callback,
                   nan_callback,
                   ],
        gradient_clip_val=10.0,
        profiler=PyTorchProfiler(dirpath=config["log_dir"]) if config["profile"] else None,
        auto_lr_find="lr",
        auto_scale_batch_size="binsearch",
        strategy="ddp" if config["gpus"] > 1 else None,
        track_grad_norm=2 if config["debug"] else -1,
    )

    if config["tune"]:
        trainer.tune(model=model, datamodule=solvation_data)

    trainer.fit(model=model, datamodule=solvation_data, ckpt_path=args.restart_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitConfSolvModule.add_args(parser)
    args = parser.parse_args()
    train_conf_solv(vars(args))
