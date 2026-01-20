import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print('root_path:', root_path)
# import matplotlib.pyplot as plt
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

def test_evaluation(model, params_dict, class_criterion, device, dl_test):
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        raw_mse_test = 0
        corr_test = 0
        num_obs = 0
        epoch_results = []
        epoch_results_raw_scale = []
        epoch_sample_idx = 0
        raw_mae0 = 0
        raw_mae1 = 0
        raw_mse0 = 0
        raw_mse1 = 0
        h_cache = []
        # print('len(dl_test):', len(dl_test)) # 1
        for i, b in enumerate(dl_test):
            print(f"Processing test batch {i+1}/{len(dl_test)}")
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

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(device)
                M_val     = b["M_val"].to(device)
                times_val = b["times_val"]
                # print("First 10 times_val11:", times_val[:10].round(4))
                times_idx = b["index_val"]
                # interval_val = b['interval_val'].to(device)
            # print('len(times_val):', len(times_val))
            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _ , _ = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True, group_idx=group_id, edge_index=edge_index, test=True)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
            h_vec = h_vec[:-1, :, :]
            h_vec = h_vec.permute(1, 0, 2) #[100, 120, 50]
            h_cache.append(h_vec.cpu().numpy())
            # try:
            #     auc_test=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            # except ValueError:
            #     if params_dict["verbose"]>=3:
            #         print("Only one class. AUC is wrong")
            #     auc_test = 0
            #     pass

            if params_dict["lambda"]==0:
                # t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                # print("First 10 t_vec:", t_vec[120:140])
                # print("First 10 times_val:", times_val[:10])
                # print("First 10 times_idx:", times_idx[:10])
                m, v = torch.chunk(p_val,2,dim=1)
                # m_aligned = m.clone()

                # for sid in np.unique(times_idx):
                #     idx = (times_idx == sid)          # 当前 sample 的所有时间点
                #     if idx.sum() == 0:
                #         continue

                #     m_aligned[idx] = m[idx] - m[idx][0] + X_val[idx][0]

                # m = m_aligned
                if m.min().item() < -1 or m.max().item() > 1:
                    print("m min/max:", m.min().item(), m.max().item())
                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                # plot figure #
                sample_id = 1
                indices = (times_idx == sample_id)
                # print("First 10 indices:", indices[:10])
                # print('indices.true:', indices.sum().item())
                m_sample = m[indices]
                x_sample = X_val[indices]
                # print("m.shape:", m.shape)
                # print("X_val.shape:", X_val.shape)
                # print("indices.sum():", indices.sum().item())
                # print('m:', m)
                # print("m_sample.shape:", m_sample.shape)
                # print("x_sample.shape:", x_sample.shape)
                # min_vals = torch.tensor([-4.999994958844029, -4.999998255850045], device=m_sample.device)
                # max_vals = torch.tensor([4.999998542893824,4.999998728242916], device=m_sample.device)
                # m_denorm = (m_sample + 1) * (max_vals - min_vals) / 2 + min_vals
                # x_denorm = (x_sample + 1) * (max_vals - min_vals) / 2 + min_vals
                m_denorm = m_sample
                x_denorm = x_sample
                # print('m_denorm:', m_denorm[:10, :2])
                # print('x_denorm:', x_denorm[:10, :2])
                # plt.figure(figsize=(8, 4))
                # plt.plot(x_denorm[:, 0].cpu(), x_denorm[:, 1].cpu(), label='Ground Truth', marker='o')
                # plt.scatter(x_denorm[0, 0].cpu(), x_denorm[0, 1].cpu(), color='blue', s=100, label='GT Start', zorder=5)
                # plt.plot(m_denorm[:, 0].cpu(), m_denorm[:, 1].cpu(), label='Prediction', marker='x')
                # plt.scatter(m_denorm[0, 0].cpu(), m_denorm[0, 1].cpu(), color='orange', s=100, label='Pred Start', zorder=5)
                # plt.title(f'Sample {sample_id}')
                # plt.xlabel('Time')
                # plt.ylabel('Value')
                # plt.legend()
                # plt.grid(True)
                # plt.tight_layout()
                # plt.savefig(f'sample_{sample_id}.png', dpi=300)
                # plt.close()  # ✅ 关闭图像窗口，防止显示
                
                # print('times_idx.shape:', times_idx.shape) # (480986, )
                # print('m.shape:', m.shape) # [480986, 4]
                # value0_mean = 0.010112594038270284
                # value0_std = 1.4162526902604045
                # value1_mean = -0.016848696946102172
                # value1_std = 1.4035620100735466
                # X_raw = torch.zeros_like(X_val)
                # m_raw = torch.zeros_like(m)
                # X_raw[:, 0] = (X_val[:, 0] * value0_std + value0_mean)
                # m_raw[:, 0] = (m[:, 0] * value0_std + value0_mean)
                # X_raw[:, 1] = (X_val[:, 1] * value1_std + value1_mean)
                # m_raw[:, 1] = (m[:, 1] * value1_std + value1_mean)
                # raw_mse_loss = (torch.pow(X_raw-m_raw,2)*M_val).sum()
                # bs = X_raw.shape[0]
                # for j in range(bs):
                #     gt_0 = float(X_raw[j, 0].detach().cpu().numpy())
                #     gt_1 = float(X_raw[j, 1].detach().cpu().numpy())
                #     pred_0 = float(m_raw[j, 0].detach().cpu().numpy())
                #     pred_1 = float(m_raw[j, 1].detach().cpu().numpy())
                #     epoch_results_raw_scale.append({
                #         "global_idx": epoch_sample_idx + j,  # 全局样本编号
                #         "dim0_gt": gt_0,
                #         "dim0_pred": pred_0,
                #         "dim1_gt": gt_1,
                #         "dim1_pred": pred_1,
                #     })
                # epoch_sample_idx += bs
                # if len(epoch_results_raw_scale) > 0:
                #     df_epoch = pd.DataFrame(epoch_results_raw_scale)
                #     df_epoch.to_csv("test_predictions_raw_scale.csv", index=False)
                bs = X_val.shape[0]
                for j in range(bs):
                    gt_0 = float(X_val[j, 0].detach().cpu().numpy())
                    gt_1 = float(X_val[j, 1].detach().cpu().numpy())
                    pred_0 = float(m[j, 0].detach().cpu().numpy())
                    pred_1 = float(m[j, 1].detach().cpu().numpy())
                    epoch_results.append({
                        "global_idx": epoch_sample_idx + j,  # 全局样本编号
                        "dim0_gt": gt_0,
                        "dim0_pred": pred_0,
                        "dim1_gt": gt_1,
                        "dim1_pred": pred_1,
                    })
                epoch_sample_idx += bs
                if len(epoch_results) > 0:
                    df_epoch = pd.DataFrame(epoch_results)
                    df_epoch.to_csv("test_predictions_scale.csv", index=False)

                # err = X_raw - m_raw
                # err2 = (err ** 2) * M_val
                # mae = torch.abs(err) * M_val
                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                # raw_mse_test += raw_mse_loss.cpu().numpy()
                # raw_mse0 += err2[:, 0].sum().cpu().numpy()
                # raw_mse1 += err2[:, 1].sum().cpu().numpy()
                # raw_mae0 += mae[:, 0].sum().cpu().numpy()
                # raw_mae1 += mae[:, 1].sum().cpu().numpy()
                # corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_test += total_loss.cpu().detach().numpy()
            # auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /=  num_obs
        # raw_mse_test /= num_obs
        # auc_total_test /= (i+1)
        # raw_mae0 /= (num_obs/2)
        # raw_mae1 /= (num_obs/2)
        # raw_mse0 /= (num_obs/2)
        # raw_rmse0 = np.sqrt(raw_mse0)
        # raw_mse1 /= (num_obs/2)
        # raw_rmse1 = np.sqrt(raw_mse1)
        h_cache = np.concatenate(h_cache, axis=0)
        return loss_test, mse_test, num_obs, h_cache

if __name__ =="__main__":

    simulation_name="Simsystem_v1"
    device = torch.device("cuda")


    # train_idx = np.load("../../gru_ode_bayes/datasets/NYCtaxi/folds/small_chunk_fold_idx_0/train_idx.npy",allow_pickle=True)
    # val_idx = np.load("../../gru_ode_bayes/datasets/NYCtaxi/folds/small_chunk_fold_idx_0/val_idx.npy",allow_pickle=True)
    test_idx = np.load("../../gru_ode_bayes/datasets/NYC/folds_gen_h_taxi_supervised/small_chunk_fold_idx_0/test_idx.npy",allow_pickle=True)
    #Model parameters.
    params_dict=dict()

    params_dict["csv_file_path"] = "../../preproc/NYC/NYCtaxi/data_accu/test/NYCtaxi_2021-12_irregular_region_1_80_norm.csv"
    params_dict["A_file"] = "../../preproc/NYC/NYCtaxi/data_accu/test/adjacency_matrix_80x80.npy"
    # params_dict["A_file"] = None
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
                                        cov_file= csv_file_cov, idx=test_idx, A_file=A_file)
    mask, gt = data_test.build_mask_and_gt(T=params_dict["T"], delta_t=params_dict["delta_t"])
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
    
    test_loglik, test_mse, num_obs, h_cache = test_evaluation(model, params_dict, class_criterion, device, dl_test)
    np.savez_compressed("h_mask_gt.npz",
                        h = h_cache,
                        mask=mask,
                        gt=gt,
                        )
    print('test_loglik:', test_loglik, "test_mse:", test_mse, 'num_obs:', num_obs)