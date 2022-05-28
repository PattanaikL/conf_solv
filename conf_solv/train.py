from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import NeptuneLogger

from conf_solv.dataloaders.loader import SolventData3DModule
from conf_solv.trainer import LitConfSolvModule

import os
from argparse import ArgumentParser

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
    neptune_logger = NeptuneLogger(
        project="lagnajit/conf_solv",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        tags=[],
        mode="offline",
    )
    try:
        neptune_logger.run
    except Exception:
        pass
    trainer = pl.Trainer(
        logger=neptune_logger,
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=[LearningRateMonitor(),
                   checkpoint_callback,
                   ],
        gradient_clip_val=10.0,
    )
    trainer.fit(model, solvation_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitConfSolvModule.add_args(parser)
    args = parser.parse_args()
    train_conf_solv(vars(args))
