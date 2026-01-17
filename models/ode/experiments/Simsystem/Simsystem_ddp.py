import sys, os
import torch
import numpy as np
import pandas as pd
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)
print('root_path:', root_path)
from torch.utils.data import Dataset, DataLoader
import gru_ode_bayes
import gru_ode_bayes.data_utils_sim as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
from gru_ode_bayes import Logger

# --- DDP 依赖引入 ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print('root_path:', root_path)

# --- DDP 初始化与清理函数 ---
def setup(rank, world_size, process_seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    import random
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed) # 多 GPU
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_gruode(rank, world_size, simulation_name, params_dict, full_train_idx, full_val_idx, epoch_max=40, seed=42):
    # 1. 启动 DDP 环境
    setup(rank, world_size, seed)
    device = torch.device(f"cuda:{rank}")

    # 2. 只有 Rank 0 初始化 Logger
    logger = None
    if rank == 0:
        log_dir = f'./Logs/{simulation_name}'
        logger = Logger(log_dir)
        print(f"Rank {rank}: Logger initialized at {log_dir}")

    csv_file_path = params_dict["csv_file_path"]
    csv_file_cov = params_dict["csv_file_cov"]
    csv_file_tags = params_dict["csv_file_tags"]
    A_file = params_dict["A_file"]

    # 3. 关键修改：手动切分数据集 ID
    # 快速读取 CSV 中的 ID 和 group_ID 映射关系 (只读这两列，速度很快)
    # 注意：这里我们假设 CSV 有表头，且列名为 'ID' 和 'group_ID'
    df_meta = pd.read_csv(csv_file_path, usecols=['ID', 'group_ID'])
    
    # 定义一个辅助函数来执行 "Group 级" 切分
    def get_local_indices(full_indices, df_meta, rank, world_size):
        # 1. 筛选出当前模式（Train/Val）下的所有数据
        # 这一步是为了确保我们只处理属于 train_idx 或 val_idx 的行
        df_subset = df_meta[df_meta['ID'].isin(full_indices)]
        
        # 2. 获取所有唯一的 Group ID
        unique_groups = df_subset['group_ID'].unique()
        
        # 3. 对 Group ID 进行切分 (而不是对 Sample ID 切分)
        # 这样保证同一个 Group 的所有样本永远在一起
        local_groups = np.array_split(unique_groups, world_size)[rank]
        
        # 4. 找回这些 Local Group 对应的所有 Sample ID
        local_indices = df_subset[df_subset['group_ID'].isin(local_groups)]['ID'].unique()
        
        return local_indices
    
    # 执行切分
    local_train_idx = get_local_indices(full_train_idx, df_meta, rank, world_size)
    local_val_idx = get_local_indices(full_val_idx, df_meta, rank, world_size)

    print(f"Rank {rank}: Assigned {len(local_train_idx)} samples from {len(np.unique(df_meta[df_meta['ID'].isin(local_train_idx)]['group_ID']))} groups.")

    if params_dict["lambda"]==0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None

    # 4. 初始化 Dataset (使用切分后的 local idx)
    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov, 
                                        idx=local_train_idx, A_file=A_file) # 使用 local_train_idx
    
    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov, 
                                        idx=local_val_idx, validation=validation, val_options=val_options, A_file=A_file) # 使用 local_val_idx

    # 5. Sampler 设置
    # 因为我们已经手动切分了 ID，这里不需要 DistributedSampler，
    # 直接在本地子集上使用 GroupBatchSampler 即可。
    id2group_train = data_train.id2group
    train_sampler = data_utils.GroupBatchSampler(id2group=id2group_train, groups_per_batch=100, shuffle_groups=True, drop_last=False)
    
    id2group_val = data_val.id2group
    val_sampler = data_utils.GroupBatchSampler(id2group=id2group_val, groups_per_batch=100, shuffle_groups=False, drop_last=False)
    
    dl   = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, batch_sampler=train_sampler)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, batch_sampler=val_sampler)

    params_dict["input_size"] = data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim
    
    # 保存参数 (仅 Rank 0)
    if rank == 0:
        os.makedirs("./../trained_models/", exist_ok=True)
        np.save(f"./../trained_models/{simulation_name}_params.npy", params_dict)

    # 6. 模型初始化与 DDP 包装
    nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(
        input_size=params_dict["input_size"], hidden_size=params_dict["hidden_size"],
        p_hidden=params_dict["p_hidden"], prep_hidden=params_dict["prep_hidden"],
        logvar=params_dict["logvar"], mixing=params_dict["mixing"],
        classification_hidden=params_dict["classification_hidden"],
        cov_size=params_dict["cov_size"], cov_hidden=params_dict["cov_hidden"],
        dropout_rate=params_dict["dropout_rate"], full_gru_ode=params_dict["full_gru_ode"], 
        impute=params_dict["impute"], dataset='Sim'
    )
    
    nnfwobj.to(device)
    # DDP 包装
    nnfwobj = DDP(nnfwobj, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay=params_dict["weight_decay"])
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    
    if rank == 0:
        print("Start Training")
        
    val_metric_prev = -1000
    mse_val_prev = 1000
    
    for epoch in range(epoch_max):
        nnfwobj.train()
        total_train_loss = 0.0
        
        # 训练循环
        # 注意：tqdm 只在 Rank 0 显示，避免控制台刷屏
        iterator = tqdm.tqdm(dl) if rank == 0 else dl
        
        for i, b in enumerate(iterator):
            optimizer.zero_grad()
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

            # DDP 模型前向传播
            hT, loss, class_pred, path_t, path_p, path_h, _, _  = nnfwobj(
                times, time_ptr, X, M, obs_idx, 
                delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, 
                return_path=True, group_idx=group_id, edge_index=edge_index
            )
            
            # Loss 计算 (DDP 会自动处理梯度同步)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels)) / batch_size
            total_train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()

        # 调整学习率
        data_utils.adjust_learning_rate(optimizer, epoch, params_dict["lr"])

        # 记录训练 Loss (仅 Rank 0)
        if rank == 0:
            print(f"Rank {rank} Loss train : {total_train_loss/(i+1)}")
            info = { 'training_loss' : total_train_loss/(i+1) }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

        # --- 验证阶段 ---
        with torch.no_grad():
            nnfwobj.eval()
            
            # 本地累加器
            local_loss_val = 0.0
            local_mse_val = 0.0
            local_corr_val = 0.0
            local_num_obs = 0.0
            local_auc_sum = 0.0
            
            for i_val, b in enumerate(dl_val):
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
                    times_idx = b["index_val"]

                hT, loss, class_pred, t_vec, p_vec, h_vec, _, _  = nnfwobj(
                    times, time_ptr, X, M, obs_idx, 
                    delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, 
                    return_path=True, group_idx=group_id, edge_index=edge_index, test=True
                )
                
                total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
                local_loss_val += total_loss.item() # 使用 item() 转为 python float

                if params_dict["lambda"]==0:
                    p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                    m, v = torch.chunk(p_val, 2, dim=1)
                    mse_loss = (torch.pow(X_val-m, 2) * M_val).sum()
                    corr_val_loss = data_utils.compute_corr(X_val, m, M_val)
                    last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                    
                    local_mse_val += mse_loss.item()
                    local_corr_val += corr_val_loss.mean().item()
                    local_num_obs += M_val.sum().item()
                    # 这里累加的是 log_lik loss 的和，后面再除以 num_obs
                    # 原代码 loss_val += last_loss
                else:
                    local_num_obs += 1

            # --- DDP 全局指标同步 ---
            # 为了确保验证结果准确，我们需要把所有卡的 MSE 和 num_obs 汇总
            
            # 创建 Tensor 用于 Reduce
            metrics_tensor = torch.tensor([local_mse_val, local_num_obs, local_loss_val, local_corr_val], device=device)
            # 汇总所有卡的值 (Sum)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            global_mse_sum = metrics_tensor[0].item()
            global_num_obs = metrics_tensor[1].item()
            global_loss_sum = metrics_tensor[2].item()
            global_corr_sum = metrics_tensor[3].item()

            # 计算最终平均指标
            final_mse = global_mse_sum / global_num_obs
            final_loss_loglik = global_loss_sum / (len(dl_val) * world_size) # 近似处理
            final_corr = global_corr_sum / (len(dl_val) * world_size) # 简单平均，严格来说应加权

            # 只有 Rank 0 进行判断和保存
            if rank == 0:
                print(f"Epoch {epoch} | Val MSE: {final_mse:.5f} | Val Num Obs: {global_num_obs}")
                
                info = { 
                    'validation_loss' : final_loss_loglik, # 简单记录
                    'validation_mse' : final_mse, 
                    'correlation_mean' : final_corr
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch)
                
                save_dir = f"./../trained_models/{simulation_name}"
                os.makedirs(save_dir, exist_ok=True)

                if final_mse < mse_val_prev:
                    print(f"New highest validation metric reached! : {final_mse}")
                    print("Saving Model")
                    best_path = os.path.join(save_dir, f"{simulation_name}_MAX.pt")
                    # 注意：保存 DDP 模型要用 .module
                    torch.save(nnfwobj.module.state_dict(), best_path)
                    mse_val_prev = final_mse
                elif epoch % 10 == 0:
                    normal_path = os.path.join(save_dir, f"{simulation_name}.pt")
                    torch.save(nnfwobj.module.state_dict(), normal_path)

    cleanup()

# --- 主启动逻辑 ---
if __name__ == "__main__":
    seed_value = 42
    simulation_name = "Simsystem_v17_merged_DDP"
    
    # 获取显卡数量
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP training...")

    # 加载全部索引
    train_idx = np.load("../../gru_ode_bayes/datasets/Simsystem/folds_merged/fold_0/train_idx.npy", allow_pickle=True)
    val_idx = np.load("../../gru_ode_bayes/datasets/Simsystem/folds_merged/fold_0/val_idx.npy", allow_pickle=True)

    # 参数设置
    params_dict = dict()
    params_dict["csv_file_path"] = "../../preproc/Simsystem/data_merged/train/Simsystem_binned_norm.csv" 
    params_dict["A_file"] = "../../preproc/Simsystem/data_merged/train/edges_train_merged.npy" 
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"]  = None

    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 
    params_dict["delta_t"] = 1
    params_dict["T"] = 120
    params_dict["lambda"] = 0 
    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.0001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"] = 0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0
    params_dict["T_val"] = 60
    params_dict["max_val_samples"] = None

    # 使用 mp.spawn 启动多进程
    # nprocs = world_size (启动与 GPU 数量相同的进程)
    mp.spawn(
        train_gruode,
        args=(world_size, simulation_name, params_dict, train_idx, val_idx, 300, seed_value),
        nprocs=world_size,
        join=True
    )