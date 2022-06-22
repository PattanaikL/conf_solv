from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.profiler import PyTorchProfiler
from conf_solv.dataloaders.loader import SolventData3DModule
from conf_solv.trainer import LitConfSolvModule
import sys

import os
from argparse import ArgumentParser

from conf_solv.utils.inference_utils import save_predictions
from conf_solv.dataloaders.features import SOLVENTS

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

    trainer.fit(model=model, datamodule=solvation_data)

    # get predictions on the test set
    scaler = trainer.datamodule.scaler
    for solvent in list(SOLVENTS.keys())[1:]:
        predict_dataloader = solvation_data.predict_dataloader()
        predict_dataloader.dataset.solvent = SOLVENTS[solvent]
        # predictions_all_batches = trainer.predict(model=model, dataloaders=predict_dataloader)
        save_predictions(
            model=model,
            dataloader=predict_dataloader,
            scaler=scaler,
            config=config,
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitConfSolvModule.add_args(parser)
    args = parser.parse_args()
    train_conf_solv(vars(args))
