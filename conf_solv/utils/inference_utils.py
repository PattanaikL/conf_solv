import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
from conf_solv.dataloaders.features import SOLVENTS_REVERSE


def make_predictions(model, data_batch):
    """
    Make predictions on a single batch (used for parallelization)

    :param model: ConfSolv model used to make predictions
    :param data_batch: Instance of batch data on which to make predictions
    :return: DataFrame of relative dG predictions per conformer
    """

    with torch.no_grad():
        torch.set_num_threads(1)
        test_dict_info = {}
        n_confs = len(data_batch.y)

        preds_batch = model._step(data_batch, n_confs, 0, mode="predict")
        preds = preds_batch['preds'].squeeze(dim=0)[:, 0]
        y_true = data_batch.y
        unscaled_preds = model.scaler.inverse_transform(preds)

        # make sure that the lowest energy conf is labeled as energy 0
        lowest_idx = y_true.argmin()
        y_true_0_offset = y_true - y_true[lowest_idx]

        # make sure that the lowest energy conf is labeled as energy 0
        lowest_idx = unscaled_preds.argmin()
        unscaled_preds_0_offset = unscaled_preds - unscaled_preds[lowest_idx]

        # save to dict
        test_dict_info['mol_id'] = [data_batch.mol_id[0]] * n_confs
        test_dict_info['solvent'] = [data_batch.solvent_name[0]] * n_confs
        test_dict_info['conf_id'] = data_batch.conf_ids[0]
        test_dict_info['dG_pred (kJ/mol)'] = unscaled_preds_0_offset.numpy()
        test_dict_info['dG_true (kJ/mol)'] = y_true_0_offset.numpy()

        return pd.DataFrame(test_dict_info)


@torch.no_grad()
def save_predictions(model, dataloader, save_dir, n_jobs=1):
    """
    Saves the test set predictions to a csv.

    :param model: Model
    :param dataloader: Dataloader that stores true values for the test set
    :param n_jobs: How many jobs to run in parallel
    :return: None
    """
    solvent = dataloader.dataset.solvent
    solvent_name = SOLVENTS_REVERSE[solvent]

    # set to eval mode for inference
    model.eval()

    # iterate over the batches
    test_dfs = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(make_predictions)(model, data_batch) for data_batch in tqdm(
            dataloader, desc=f'{solvent_name}', total=len(dataloader)
        )
    )

    df_test_info = pd.concat(test_dfs)
    df_test_info.to_csv(os.path.join(save_dir, f'{solvent_name}_test_predictions.csv'), index=False)

    # group the test errors by solvent
    stats_dict = {
        'solvent': [solvent],
        'MAE (kJ/mol)': [],
        'RMSE (kJ/mol)': [],
    }

    preds = df_test_info['dG_pred (kJ/mol)'].values
    true = df_test_info['dG_true (kJ/mol)'].values

    rmse = np.sqrt(sum((preds - true)**2) / len(true))
    mae = np.abs(preds - true).sum() / len(true)
    stats_dict['MAE (kJ/mol)'].append(mae)
    stats_dict['RMSE (kJ/mol)'].append(rmse)

    df_stats = pd.DataFrame(stats_dict)
    df_stats.to_csv(os.path.join(save_dir, f'{solvent_name}_test_stats.csv'), index=False)
