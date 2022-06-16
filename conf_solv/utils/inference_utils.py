import os

import numpy as np
import pandas as pd
import torch


def save_predictions(predictions, dataloader, scaler, config):
    """
    Saves the test set predictions to a csv.

    :param predictions: Test set predictions
    :param dataloader: Dataloader that stores true values for the test set
    :param scaler: Scaler that was fit to the training data. Used to unscale the predictions
    :param config: configuration dictionary
    :return: None
    """
    unscaled_preds_all = torch.tensor([])
    y_true_all = torch.tensor([])
    test_dict_info = {'mol_id': [], 
                      'conf_id': [],
                      'solvent': [], 
                      'dG_pred (kJ/mol)': [],
                      'dG_true (kJ/mol)': [],
                      }
    # iterate over the batches
    for data_batch, preds_batch in zip(dataloader, predictions):
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
