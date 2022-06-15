from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.profiler import PyTorchProfiler
from conf_solv.dataloaders.loader import SolventData3DModule
from conf_solv.trainer import LitConfSolvModule
import sys
import torch
import pandas as pd

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
    predictions_all_batches = trainer.predict(ckpt_path='best')
    scaler = trainer.datamodule.scaler

    unscaled_preds_all = torch.tensor([])
    y_true_all = torch.tensor([])
    test_dict_info = {'mol_id': [], 
                      'conf_id': [],
                      'solvent': [], 
                      'dG_pred (kJ/mol)': [],
                      'dG_true (kJ/mol)': [],
                     }
    # iterate over the batches
    for data_batch, preds_batch in zip(solvation_data.predict_dataloader(), predictions_all_batches):
        preds = preds_batch['preds'].squeeze()
        y_true = data_batch.y
        unscaled_preds = scaler.inverse_transform(preds)
        normalized_coeff = data_batch.solute_mask.sum()
        assert normalized_coeff == len(unscaled_preds)
        
        # make sure that the lowest energy conf is labeled as energy 0
        lowest_idx = y_true.argmin()
        y_true_0_offset = y_true - y_true[lowest_idx]
        
        # make sure that the lowest energy conf is labeled as energy 0
        lowest_idx = unscaled_preds.argmin()
        unscaled_preds_0_offset = unscaled_preds - unscaled_preds[lowest_idx]

        unscaled_preds_all = torch.cat((unscaled_preds_all, unscaled_preds_0_offset))
        y_true_all = torch.cat((y_true_all, y_true_0_offset))
                
        for i, conf_id in enumerate(data_batch.conf_ids[0]):
            test_dict_info['mol_id'].append(data_batch.mol_id[0])
            test_dict_info['solvent'].append(data_batch.solvent_name[0])    
            test_dict_info['conf_id'].append(conf_id)
    
    test_dict_info['dG_pred (kJ/mol)'] = unscaled_preds_all.detach().numpy()
    test_dict_info['dG_true (kJ/mol)'] = y_true_all.detach().numpy()
    df_test_info = pd.DataFrame(test_dict_info)
    df_test_info.to_csv(os.path.join(f"{config['log_dir']}", 'test_predictions.csv'), index=False)

    # group the test errors by solvent
    stats_dict = {'solvent': df_test_info.solvent.unique().tolist(),
                  'MAE (kJ/mol)': [],
                  'RMSE (kJ/mol)': [],
                 }
    for solvent in stats_dict['solvent']:
        df_tmp = df_test_info[df_test_info.solvent == solvent]
        preds = df_tmp['dG_pred (kJ/mol)'].values
        true = df_tmp['dG_true (kJ/mol)'].values
        
        rmse = np.sqrt(sum((preds - true)**2) / len(true))
        mae = np.abs(preds - true).sum() / len(true)
        stats_dict['MAE (kJ/mol)'].append(mae)
        stats_dict['RMSE (kJ/mol)'].append(rmse)

    preds = df_test_info['dG_pred (kJ/mol)'].values
    true = df_test_info['dG_true (kJ/mol)'].values
    rmse = np.sqrt(sum((preds - true)**2) / len(true))
    mae = np.abs(preds - true).sum() / len(true)

    stats_dict['solvent'].append('Overall')
    stats_dict['MAE (kJ/mol)'].append(mae)
    stats_dict['RMSE (kJ/mol)'].append(rmse)
    df_stats = pd.DataFrame(stats_dict)
    df_stats.to_csv(os.path.join(f"{config['log_dir']}", 'test_stats.csv'), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitConfSolvModule.add_args(parser)
    args = parser.parse_args()
    train_conf_solv(vars(args))
