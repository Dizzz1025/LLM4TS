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
import gru_ode_bayes.data_utils_nyc as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
from gru_ode_bayes import Logger

# --- DDP Dependency Imports ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Path Configuration
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_path)
print('root_path:', root_path)

# --- DDP Initialization and Cleanup Functions ---
def setup(rank, world_size, process_seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    import random
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed) # Multi-GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_gruode(rank, world_size, simulation_name, params_dict, full_train_idx, full_val_idx, epoch_max=40, seed=42):
    # 1. Initialize DDP environment
    setup(rank, world_size, seed)
    device = torch.device(f"cuda:{rank}")

    # 2. Only Rank 0 initializes the Logger
    logger = None
    if rank == 0:
        log_dir = f'./Logs/{simulation_name}'
        logger = Logger(log_dir)
        print(f"Rank {rank}: Logger initialized at {log_dir}")

    csv_file_path = params_dict["csv_file_path"]
    csv_file_cov = params_dict["csv_file_cov"]
    csv_file_tags = params_dict["csv_file_tags"]
    A_file = params_dict["A_file"]

    # 3. Manual dataset ID splitting
    df_meta = pd.read_csv(csv_file_path, usecols=['ID', 'group_ID'])
    
    def get_local_indices(full_indices, df_meta, rank, world_size):
        # 1. Filter all data for the current mode (Train/Val)
        df_subset = df_meta[df_meta['ID'].isin(full_indices)]
        
        # 2. Get all unique Group IDs
        unique_groups = df_subset['group_ID'].unique()
        
        # 3. Split Group IDs
        local_groups = np.array_split(unique_groups, world_size)[rank]
        
        # 4. Retrieve all Sample IDs corresponding to these Local Groups
        local_indices = df_subset[df_subset['group_ID'].isin(local_groups)]['ID'].unique()
        
        return local_indices
    
    # Execute splitting
    local_train_idx = get_local_indices(full_train_idx, df_meta, rank, world_size)
    local_val_idx = get_local_indices(full_val_idx, df_meta, rank, world_size)

    print(f"Rank {rank}: Assigned {len(local_train_idx)} samples from {len(np.unique(df_meta[df_meta['ID'].isin(local_train_idx)]['group_ID']))} groups.")

    if params_dict["lambda"]==0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None

    # 4. Initialize Dataset (using the split local idx)
    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov, 
                                        idx=local_train_idx, A_file=A_file) # Using local_train_idx
    
    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov, 
                                        idx=local_val_idx, validation=validation, val_options=val_options, A_file=A_file) # Using local_val_idx

    # 5. Sampler Configuration
    # Since IDs are manually split, DistributedSampler is not required.
    # GroupBatchSampler is used directly on the local subset.
    id2group_train = data_train.id2group
    train_sampler = data_utils.GroupBatchSampler(id2group=id2group_train, groups_per_batch=5, shuffle_groups=True, drop_last=False)
    
    id2group_val = data_val.id2group
    val_sampler = data_utils.GroupBatchSampler(id2group=id2group_val, groups_per_batch=5, shuffle_groups=False, drop_last=False)
    
    dl   = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, batch_sampler=train_sampler)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, batch_sampler=val_sampler)

    params_dict["input_size"] = data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim
    
    # Save parameters (Rank 0 only)
    if rank == 0:
        os.makedirs("./../trained_models/", exist_ok=True)
        np.save(f"./../trained_models/{simulation_name}_params.npy", params_dict)

    # 6. Model initialization and DDP wrapping
    nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(
        input_size=params_dict["input_size"], hidden_size=params_dict["hidden_size"],
        p_hidden=params_dict["p_hidden"], prep_hidden=params_dict["prep_hidden"],
        logvar=params_dict["logvar"], mixing=params_dict["mixing"],
        classification_hidden=params_dict["classification_hidden"],
        cov_size=params_dict["cov_size"], cov_hidden=params_dict["cov_hidden"],
        dropout_rate=params_dict["dropout_rate"], full_gru_ode=params_dict["full_gru_ode"], 
        impute=params_dict["impute"], dataset='nyc'
    )
    
    nnfwobj.to(device)
    # DDP wrapping
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
        
        # Training loop
        # Note: tqdm only shown on Rank 0 to avoid console spamming
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

            # DDP Forward Pass
            hT, loss, class_pred, path_t, path_p, path_h, _, _  = nnfwobj(
                times, time_ptr, X, M, obs_idx, 
                delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, 
                return_path=True, group_idx=group_id, edge_index=edge_index
            )
            
            # Loss Calculation
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels)) / batch_size
            total_train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()

        # Adjust learning rate
        data_utils.adjust_learning_rate(optimizer, epoch, params_dict["lr"])

        # Log training loss
        if rank == 0:
            print(f"Rank {rank} Loss train : {total_train_loss/(i+1)}")
            info = { 'training_loss' : total_train_loss/(i+1) }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

        # --- Validation Phase ---
        with torch.no_grad():
            nnfwobj.eval()
            
            # Local accumulators
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
                local_loss_val += total_loss.item() # Using item() to convert to python float

                if params_dict["lambda"]==0:
                    p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                    m, v = torch.chunk(p_val, 2, dim=1)
                    mse_loss = (torch.pow(X_val-m, 2) * M_val).sum()
                    corr_val_loss = data_utils.compute_corr(X_val, m, M_val)
                    last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                    
                    local_mse_val += mse_loss.item()
                    local_corr_val += corr_val_loss.mean().item()
                    local_num_obs += M_val.sum().item()
                    # Accumulating the sum of log_lik loss; will be divided by num_obs later
                else:
                    local_num_obs += 1

            # --- DDP Global Metric Synchronization ---
            # Create Tensor for Reduction
            metrics_tensor = torch.tensor([local_mse_val, local_num_obs, local_loss_val, local_corr_val], device=device)
            # Sum values across all ranks
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            global_mse_sum = metrics_tensor[0].item()
            global_num_obs = metrics_tensor[1].item()
            global_loss_sum = metrics_tensor[2].item()
            global_corr_sum = metrics_tensor[3].item()

            # Calculate final averaged metrics
            final_mse = global_mse_sum / global_num_obs
            final_loss_loglik = global_loss_sum / (len(dl_val) * world_size) # Approximate handling
            final_corr = global_corr_sum / (len(dl_val) * world_size) # Simple average (strictly should be weighted)

            # Decision making and saving on Rank 0 only
            if rank == 0:
                print(f"Epoch {epoch} | Val MSE: {final_mse:.5f} | Val Num Obs: {global_num_obs}")
                
                info = { 
                    'validation_loss' : final_loss_loglik, # Logging simple record
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
                    torch.save(nnfwobj.module.state_dict(), best_path)
                    mse_val_prev = final_mse
                elif epoch % 10 == 0:
                    normal_path = os.path.join(save_dir, f"{simulation_name}.pt")
                    torch.save(nnfwobj.module.state_dict(), normal_path)

    cleanup()

# --- Main Launch Logic ---
if __name__ == "__main__":
    seed_value = 2026
    simulation_name = "NYC_v17_merged_DDP_v7"
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP training...")

    # Load all indices
    train_idx = np.load("../../gru_ode_bayes/datasets/NYC/folds_merged/fold_0/train_idx.npy", allow_pickle=True)
    val_idx = np.load("../../gru_ode_bayes/datasets/NYC/folds_merged/fold_0/val_idx.npy", allow_pickle=True)

    # Parameter setup
    params_dict = dict()
    params_dict["csv_file_path"] = "../../preproc/NYC/NYCmerged/train/NYCmerged_irregular_region_1_80_norm.csv" 
    params_dict["A_file"] = None
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"]  = None

    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 0.5 
    params_dict["delta_t"] = 1
    params_dict["T"] = 216
    params_dict["lambda"] = 0 
    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"] = 0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0
    params_dict["T_val"] = 144
    params_dict["max_val_samples"] = None

    # Use mp.spawn to start multi-process training
    # nprocs = world_size (starts one process per GPU)
    mp.spawn(
        train_gruode,
        args=(world_size, simulation_name, params_dict, train_idx, val_idx, 300, seed_value),
        nprocs=world_size,
        join=True
    )