import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print('root_path:', root_path)
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gru_ode_bayes
import gru_ode_bayes.data_utils_nyc as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
from gru_ode_bayes import Logger

def test_evaluation(model, params_dict, class_criterion, device, dl_test, simulation_name):
    with torch.no_grad():
        model.eval()
        loss_test = 0
        num_obs = 0
        
        # 分别统计两个特征的指标 (维度 0 和 1)
        mse_per_feature = np.zeros(2)
        mae_per_feature = np.zeros(2)
        num_obs_per_feature = np.zeros(2)
        
        # 用于保存所有 batch 的 p_vec
        all_p_vecs = []

        for i, b in enumerate(dl_test):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels   = b["y"].to(device)
            group_id = b["group_id"].to(device)
            edge_index = b["edge_index"].to(device)
            batch_size = labels.size(0)

            X_val     = b["X_val"].to(device) if b["X_val"] is not None else None
            M_val     = b["M_val"].to(device) if b["M_val"] is not None else None
            times_val = b["times_val"]
            times_idx = b["index_val"]

            # 模型预测
            hT, loss, class_pred, t_vec, p_vec, h_vec, _ , _ = model(
                times, time_ptr, X, M, obs_idx, 
                delta_t=params_dict["delta_t"], T=params_dict["T"], 
                cov=cov, return_path=True, group_idx=group_id, 
                edge_index=edge_index, test=True
            )
            
            # 暂存 p_vec (转换成 cpu numpy 以节省显存)
            all_p_vecs.append(p_vec.cpu().numpy()[:, :, :2])

            if params_dict["lambda"] == 0 and X_val is not None:
                # 提取预测值
                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1) 

                # 累加总 Log-likelihood
                last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                loss_test += last_loss.item()
                num_obs += M_val.sum().item()

                # 分特征累加误差
                for d in range(2): 
                    mask = M_val[:, d]
                    diff = (X_val[:, d] - m[:, d]) * mask
                    
                    mae_per_feature[d] += torch.sum(torch.abs(diff)).item()
                    mse_per_feature[d] += torch.sum(diff**2).item()
                    num_obs_per_feature[d] += torch.sum(mask).item()

        # 计算平均值
        for d in range(2):
            if num_obs_per_feature[d] > 0:
                mae_per_feature[d] /= num_obs_per_feature[d]
                mse_per_feature[d] /= num_obs_per_feature[d]

        avg_loglik = loss_test / num_obs if num_obs > 0 else 0
        rmse_per_feature = np.sqrt(mse_per_feature)

        # 保存 p_vec
        np.save(f"test_p_vec_{simulation_name}.npy", np.concatenate(all_p_vecs, axis=1))
        print(f"--- Predictions (p_vec) saved to test_p_vec_paths.npy ---")

        return (avg_loglik, 
                mae_per_feature[0], mae_per_feature[1], 
                mse_per_feature[0], mse_per_feature[1], 
                rmse_per_feature[0], rmse_per_feature[1])

if __name__ =="__main__":

    # simulation_name="taxi_zeroshot"
    # simulation_name="bike_zeroshot"
    # simulation_name="bike_supervised"
    simulation_name="crime_zeroshot"
    device = torch.device("cuda")
    test_idx = np.load(f"../../gru_ode_bayes/datasets/NYC/folds_gen_h_{simulation_name}/small_chunk_fold_idx_0/test_idx.npy",allow_pickle=True)
    
    #Model parameters.
    params_dict=dict()

    # params_dict["csv_file_path"] = "../../preproc/NYC/NYCtaxi/data_accu/test/NYCtaxi_2020-01_irregular_region_81_160_norm.csv"
    # params_dict["A_file"] = "../../preproc/NYC/NYCtaxi/data_accu/test/adjacency_matrix_80x80_zero-shot.npy"
    # params_dict["csv_file_path"] = "../../preproc/NYC/NYCbike/data_accu/test/NYCbike_2020-01_irregular_region_81_160_norm.csv"
    # params_dict["A_file"] = "../../preproc/NYC/NYCbike/data_accu/test/adjacency_matrix_80x80_zero-shot.npy"
    # params_dict["csv_file_path"] = "../../preproc/NYC/NYCbike/data_accu/test/NYCbike_2021-12_irregular_region_1_80_norm.csv"
    # params_dict["A_file"] = "../../preproc/NYC/NYCbike/data_accu/test/adjacency_matrix_80x80.npy"
    # params_dict["csv_file_path"] = "../../preproc/NYC/NYCcrime/data/test/NYCcrime_2021_irregular_region_1_80_norm.csv"
    # params_dict["A_file"] = "../../preproc/NYC/NYCcrime/data/test/adjacency_matrix_80x80.npy"
    params_dict["csv_file_path"] = "../../preproc/NYC/NYCcrime/data/test/NYCcrime_2020_irregular_region_81_160_norm.csv"
    params_dict["A_file"] = "../../preproc/NYC/NYCcrime/data/test/adjacency_matrix_80x80_zero-shot.npy"
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"]  = None
    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
    params_dict["delta_t"]=1
    params_dict["T"]=216
    params_dict["lambda"] = 0 #Weighting between classification and MSE loss.

    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.0001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"]=0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0 #from 0 to 3 (highest)

    params_dict["T_val"] = 144
    params_dict["max_val_samples"] = None
    
    csv_file_path = params_dict["csv_file_path"]
    csv_file_cov = params_dict["csv_file_cov"]
    csv_file_tags = params_dict["csv_file_tags"]
    A_file = params_dict["A_file"]
    if params_dict["lambda"]==0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None
    data_test = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options, A_file=A_file)
    id2group = data_test.id2group
    test_sampler = data_utils.GroupBatchSampler(id2group=id2group, groups_per_batch=5, shuffle_groups=False, drop_last=False)
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, batch_sampler=test_sampler)
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    
    params_dict["input_size"] = data_test.variable_num
    params_dict["cov_size"] = data_test.cov_dim

    model = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            classification_hidden=params_dict["classification_hidden"],
                                            cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                            dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"], 
                                            impute = params_dict["impute"], dataset='nyc').to(device)
    state_dict = torch.load('../trained_models/NYC_v17_merged_DDP/NYC_v17_merged_DDP_MAX.pt', map_location=device)
    model.load_state_dict(state_dict)
    
    loglik, mae0, mae1, mse0, mse1, rmse0, rmse1 = test_evaluation(model, params_dict, class_criterion, device, dl_test, simulation_name)
    
    # 按照要求的顺序打印
    print("\n" + "="*30)
    print("TEST EVALUATION RESULTS")
    print("="*30)
    # 先打印 MAE
    print(f"[Variable: Dim 0]")
    print(f"  MAE:  {mae0:.4f}")
    print(f"  MSE:  {mse0:.4f}")
    print("-" * 40)

    # 维度 1 的指标
    print(f"[Variable: Dim 1]")
    print(f"  MAE:  {mae1:.4f}")
    print(f"  MSE:  {mse1:.4f}")
    print("="*30)