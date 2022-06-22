import os
from argparse import ArgumentParser

from pytorch_lightning import seed_everything
from conf_solv.dataloaders.loader import SolventData3DModule
from conf_solv.trainer import LitConfSolvModule
from conf_solv.utils.inference_utils import save_predictions
from conf_solv.dataloaders.features import SOLVENTS


def predict_conf_solv(config):

    # load model and data
    seed_everything(config["seed"], workers=True)
    solvation_data = SolventData3DModule(config)
    model = LitConfSolvModule.load_from_checkpoint(
        checkpoint_path=os.path.join(config["trained_model_dir"], "best_model.ckpt"),
    )

    # get predictions on the test set
    scaler = model.scaler
    for solvent in list(SOLVENTS.keys())[1:]:
        predict_dataloader = solvation_data.predict_dataloader()
        predict_dataloader.dataset.solvent = SOLVENTS[solvent]
        save_predictions(
            model=model,
            dataloader=predict_dataloader,
            scaler=scaler,
            save_dir=config["trained_model_dir"],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitConfSolvModule.add_args(parser)
    parser.add_argument('--trained_model_dir', type=str)
    args = parser.parse_args()
    predict_conf_solv(vars(args))
