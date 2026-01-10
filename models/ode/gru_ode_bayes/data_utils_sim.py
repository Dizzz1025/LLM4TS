import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy import special

class ODE_DatasetNumpy(Dataset):
    """Dataset class for ODE type of data. Fed from numpy arrays.
    Args:
        times    array of times
        ids      ids (ints) of patients (samples)
        values   value matrix, each line is one observation
        masks    observation mask (1.0 means observed, 0.0 missing)
    """
    def __init__(self, times, ids, values, masks):
        assert times.shape[0] == ids.shape[0]
        assert times.shape[0] == values.shape[0]
        assert values.shape   == masks.shape

        times  = times.astype(np.float32)
        values = values.astype(np.float32)
        masks  = masks.astype(np.float32)

        df_values = pd.DataFrame(values, columns=[f"Value_{i}" for i in range(values.shape[1])])
        df_masks  = pd.DataFrame(masks, columns=[f"Mask_{i}" for i in range(masks.shape[1])])

        self.df = pd.concat([
            pd.Series(times, name="Time"),
            pd.Series(ids, name="ID"),
            df_values,
            df_masks,
        ], axis=1)
        self.df.sort_values("Time", inplace=True)
        self.length = self.df["ID"].nunique()
        self.df.set_index("ID", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        covs = self.df.loc[idx,"Time"] # To change !!
        ## returning also idx to allow empty samples
        return {"idx": idx, "y": 0, "path": subset, "cov": covs }


class ODE_Dataset(Dataset):
    """
    Dataset class for ODE type of data. With 2 values.
    Can be fed with either a csv file containg the dataframe or directly with a panda dataframe.
    One can further provide samples idx that will be used (for training / validation split purposes.)
    """
    def __init__(self, csv_file=None, cov_file=None, label_file=None, panda_df=None, cov_df=None, label_df=None, root_dir="./", t_mult=1.0, idx=None, jitter_time=0, validation = False, val_options = None, A_file=None):
        """
        Args:
            csv_file   CSV file to load the dataset from
            panda_df   alternatively use pandas df instead of CSV file
            root_dir   directory of the CSV file
            t_mult     multiplier for time values (1.0 default)
            jitter_time  jitter size (0 means no jitter), to add randomly to Time.
                         Jitter is added before multiplying with t_mult
            validation boolean. True if this dataset is for validation purposes
            val_options  dictionnary with validation dataset options.
                                    T_val : Time after which observations are considered as test samples
                                    max_val_samples : maximum number of test observations per trajectory.

        """
        self.validation = validation

        if panda_df is not None:
            assert (csv_file is None), "Only one feeding option should be provided, not both"
            self.df = panda_df
            self.cov_df = cov_df
            self.label_df  = label_df
        else:
            assert (csv_file is not None) , "At least one feeding option required !"
            self.df = pd.read_csv(root_dir + "/" + csv_file)
            assert self.df.columns[0]=="ID"
            if label_file is None:
                self.label_df = None
            else:
                self.label_df = pd.read_csv(root_dir + "/" + label_file)
                assert self.label_df.columns[0]=="ID"
                assert self.label_df.columns[1]=="label"
            if cov_file is None :
                self.cov_df = None
            else:
                self.cov_df = pd.read_csv(root_dir + "/" + cov_file)
                assert self.cov_df.columns[0]=="ID"
        if A_file is not None:
            self.A_all = np.load(A_file)
        #Create Dummy covariates and labels if they are not fed.
        if self.cov_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.cov_df = pd.DataFrame({"ID":self.df["ID"].unique(),"Cov": num_unique})
        if self.label_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.label_df = pd.DataFrame({"ID":self.df["ID"].unique(),"label": num_unique})

        #If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False
        if self.validation:
            df_beforeIdx = self.df.loc[self.df["Time"]<=val_options["T_val"],"ID"].unique()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                df_afterIdx  = self.df.loc[self.df["Time"]>=val_options["T_val_from"],"ID"].unique()
                self.store_last = True #Dataset get will return a flag for the collate to compute the last sample before T_val
            else:
                df_afterIdx  = self.df.loc[self.df["Time"]>val_options["T_val"],"ID"].unique()
            
            valid_idx = np.intersect1d(df_beforeIdx,df_afterIdx)
            self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
            self.label_df = self.label_df.loc[self.label_df["ID"].isin(valid_idx)]
            self.cov_df   = self.cov_df.loc[self.cov_df["ID"].isin(valid_idx)]

        if idx is not None:
            self.df = self.df.loc[self.df["ID"].isin(idx)].copy()
            map_dict= dict(zip(self.df["ID"].unique(),np.arange(self.df["ID"].nunique())))
            self.df["ID"] = self.df["ID"].map(map_dict) # Reset the ID index.

            self.cov_df = self.cov_df.loc[self.cov_df["ID"].isin(idx)].copy()
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict) # Reset the ID index.

            self.label_df = self.label_df.loc[self.label_df["ID"].isin(idx)].copy()
            self.label_df["ID"] = self.label_df["ID"].map(map_dict) # Reset the ID index.


        assert self.cov_df.shape[0]==self.df["ID"].nunique()

        self.variable_num = sum([c.startswith("Value") for c in self.df.columns]) #number of variables in the dataset
        self.cov_dim = self.cov_df.shape[1]-1

        self.cov_df = self.cov_df.astype(np.float32)
        self.cov_df.set_index("ID", inplace=True)

        self.label_df.set_index("ID",inplace=True)

        self.df.Time    = self.df.Time * t_mult

        #TO DO : make jitter compatible with several variables
        if jitter_time != 0:
            self.df = add_jitter(self.df, jitter_time=jitter_time)
            self.df.Value_1 = self.df.Value_1.astype(np.float32)
            self.df.Value_2 = self.df.Value_2.astype(np.float32)
            self.df.Mask_1  = self.df.Mask_1.astype(np.float32)
            self.df.Mask_2  = self.df.Mask_2.astype(np.float32)

        else:
            self.df = self.df.astype(np.float32)

        if self.validation:
            assert val_options is not None, "Validation set options should be fed"
            self.df_before = self.df.loc[self.df["Time"]<=val_options["T_val"]].copy()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                self.df_after  = self.df.loc[self.df["Time"]>=val_options["T_val_from"]].sort_values("Time").copy()
            else:
                self.df_after  = self.df.loc[self.df["Time"]>val_options["T_val"]].sort_values("Time").copy()

            if val_options.get("T_closest") is not None:
                df_after_temp = self.df_after.copy()
                df_after_temp["Time_from_target"] = (df_after_temp["Time"]-val_options["T_closest"]).abs()
                df_after_temp.sort_values(by=["Time_from_target","Value_0"], inplace = True,ascending=True)
                df_after_temp.drop_duplicates(subset=["ID"],keep="first",inplace = True)
                self.df_after = df_after_temp.drop(columns = ["Time_from_target"])
            elif val_options.get("max_val_samples") is not None:
                self.df_after  = self.df_after.groupby("ID").head(val_options["max_val_samples"]).copy()
            else:
                pass

            self.df = self.df_before #We remove observations after T_val


            self.df_after.ID = self.df_after.ID.astype(int)
            self.df_after.sort_values("Time", inplace=True)
        else:
            self.df_after = None


        self.length     = self.df["ID"].nunique()
        self.df.ID      = self.df.ID.astype(int)
        
        id_group_series = self.df.groupby("ID")["group_ID"].first().sort_index() # 索引是每个ID，值是这个ID对应的group_ID
        # sanity check：长度要等于 unique ID 数
        assert len(id_group_series) == self.df["ID"].nunique()
        self.id2group = id_group_series.values.astype(int)
        
        self.df.set_index("ID", inplace=True)

        self.df.sort_values("Time", inplace=True)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        A = self.A_all[self.id2group[idx]]
        subset = self.df.loc[idx]
        assert subset["group_ID"].iloc[0] == self.id2group[idx]
        if len(subset.shape)==1: #Don't ask me anything about this.
            subset = self.df.loc[[idx]]
        group_id = int(subset["group_ID"].iloc[0])  # 同一ID下应恒定
        covs = self.cov_df.loc[idx].values
        tag  = self.label_df.loc[idx].astype(np.float32).values
        if self.validation :
            val_samples = self.df_after.loc[self.df_after["ID"]==idx]
        else:
            val_samples = None
        ## returning also idx to allow empty samples
        return {"idx":idx, "group_ID": group_id, "y": tag, "path": subset, "cov": covs , "A": A, "val_samples":val_samples, "store_last":self.store_last}
    def build_mask_and_gt(self, T=120, delta_t=1.0):
        """
        为每条序列(ID)生成:
        - mask: [num_seq, L_resample]，某时刻有观测则为1
        - gt:   [num_seq, L_resample, variable_num]，在观测时刻填Value
        """

        num_seq = self.length                  # unique ID 数
        variable_num = self.variable_num       # 有多少 Value_k
        mask = np.zeros((num_seq, T), dtype=np.float32)
        gt   = np.zeros((num_seq, T, variable_num), dtype=np.float32)

        # 按 ID 分组（df 已经 set_index("ID")）
        grouped = self.df.groupby(level=0)

        for seq_id, df_seq in grouped:
            times = df_seq["Time"].values                      # [num_obs]
            values = df_seq.filter(regex="^Value").values      # [num_obs, variable_num]

            for i, t in enumerate(times):
                t_idx = int(round(float(t) / float(delta_t)))

                if 0 <= t_idx < T:
                    mask[seq_id, t_idx] = 1.0
                    gt[seq_id, t_idx, :] = values[i]

        return mask, gt

def add_jitter(df, jitter_time=1e-3):
    """Modifies Double OU dataset, so that observations with both dimensions
       are split. One is randomly shifted earlier by amount 'jitter_time'.
    """
    if df.columns.shape[0] != 6:
        raise ValueError("Only df with 6 columns: supports 2 value and 2 mask columns.")

    both = (df["Mask_1"] == 1.0) & (df["Mask_2"] == 1.0)
    df_single = df[both == False]
    df_both   = df[both]
    df_both1  = df_both.copy()
    df_both2  = df_both.copy()

    df_both1["Mask_2"] = 0.0
    df_both2["Mask_1"] = 0.0
    jitter = np.random.randint(2, size=df_both1.shape[0])
    df_both1["Time"] -= jitter_time * jitter
    df_both2["Time"] -= jitter_time * (1 - jitter)

    df_jit = pd.concat([df_single, df_both1, df_both2])
    ## make sure time is not negative:
    df_jit.Time.clip_lower(0.0, inplace=True)
    return df_jit

class GroupBatchSampler(Sampler):
    """
    以 group 为单位采样；每个 batch 打包多个 group（groups_per_batch）。
    保证：同组样本不拆分。
    """
    def __init__(
        self,
        id2group,                 # array-like, len = len(dataset), dataset_idx -> group_id
        groups_per_batch=100,       # 一个 batch 含多少个 group
        shuffle_groups=True,      # 是否打乱 group 顺序
        shuffle_within_batch=False,  # batch 内是否打乱样本索引顺序
        drop_last=False,
        seed=0,
    ):
        self.id2group = np.asarray(id2group)
        self.groups_per_batch = int(groups_per_batch)
        self.shuffle_groups = shuffle_groups
        self.shuffle_within_batch = shuffle_within_batch
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        # group -> [dataset_indices]
        self.group2indices = {}
        for ds_idx, g in enumerate(self.id2group):
            g = int(g)
            self.group2indices.setdefault(g, []).append(ds_idx)

        self.groups = list(self.group2indices.keys())

    def __iter__(self):
        groups = self.groups.copy()
        if self.shuffle_groups:
            self.rng.shuffle(groups)

        batch = []
        g_in_batch = 0

        for g in groups:
            batch.extend(self.group2indices[g])
            g_in_batch += 1

            if g_in_batch == self.groups_per_batch:
                if self.shuffle_within_batch:
                    self.rng.shuffle(batch)
                yield batch
                batch = []
                g_in_batch = 0

        # leftover
        if len(batch) > 0 and (not self.drop_last):
            if self.shuffle_within_batch:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        # batch 数按 group 计
        n_groups = len(self.groups)
        if self.drop_last:
            return n_groups // self.groups_per_batch
        return (n_groups + self.groups_per_batch - 1) // self.groups_per_batch

def custom_collate_fn(batch):
    A_list = [b["A"] for b in batch]
    num_nodes = A_list[0].shape[0]
    idx2batch = pd.Series(np.arange(len(batch)), index = [b["idx"] for b in batch])

    pat_idx   = [b["idx"] for b in batch]
    group_idx   = torch.tensor([b["group_ID"] for b in batch], dtype=torch.long)
    df        = pd.concat([b["path"] for b in batch],axis=0)
    df.sort_values(by=["Time"], inplace=True)

    df_cov    = torch.Tensor([b["cov"] for b in batch])

    labels  = torch.tensor([b["y"] for b in batch])

    batch_ids     = idx2batch[df.index.values].values

    ## calculating number of events at every time
    times, counts = np.unique(df.Time.values, return_counts=True)
    time_ptr      = np.concatenate([[0], np.cumsum(counts)])

    ## tensors for the data in the batch

    value_cols = [c.startswith("Value") for c in df.columns]
    mask_cols  = [c.startswith("Mask") for c in df.columns]

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b["val_samples"] for b in batch)
        df_after.sort_values(by=["ID","Time"], inplace=True)
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val  = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:,value_cols_val].values)
        M_val = torch.tensor(df_after.iloc[:,mask_cols_val].values)
        times_val = df_after["Time"].values
        index_val = idx2batch[df_after["ID"].values].values
        # Last observation before the T_val cut_off. THIS IS LIKELY TO GIVE ERRORS IF THE NUMBER OF VALIDATION SAMPLES IS HIGHER THAN 2. CHECK THIS.
        if batch[0]["store_last"]:
            df_last = df[~df.index.duplicated(keep="last")].copy()
            index_last = idx2batch[df_last.index.values].values
        
            perm_last = sort_array_on_other(index_val,index_last)
            tens_last = torch.tensor(df_last.iloc[:,value_cols].values[perm_last,:])
            index_last = index_last[perm_last]
        else:
            index_last = 0
            tens_last = 0
    else:
        X_val = None
        M_val = None
        times_val = None
        index_val = None
        tens_last = None
        index_last = None

    edge_index = batch_edge_index(A_list, num_nodes)
    res = {}
    res["pat_idx"]  = pat_idx
    res["group_id"] = group_idx
    res["times"]    = times
    res["time_ptr"] = time_ptr
    res["X"]        = torch.tensor(df.iloc[:, value_cols].values)
    res["M"]        = torch.tensor(df.iloc[:, mask_cols].values)
    res["obs_idx"]  = torch.tensor(batch_ids)
    res["y"]        = labels
    res["cov"]      = df_cov
    res["edge_index"] = edge_index
    
    res["X_val"]    = X_val
    res["M_val"]    = M_val
    res["times_val"]= times_val
    res["index_val"]= index_val
    res["X_last"]   = tens_last
    res["obs_idx_last"]= index_last
    return res

def adj_to_edge_index(A):
    """
    A: [N, N] adjacency matrix (numpy or torch)
    return: edge_index [2, E]
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A)
    src, dst = torch.nonzero(A, as_tuple=True)
    return torch.stack([src, dst], dim=0)

def batch_edge_index(A_list, num_nodes):
    """
    A_list: list of adjacency matrices, length = batch_size
    """
    edge_indices = []
    # print('len(A_list):', len(A_list))
    # print('num_nodes:', num_nodes)
    for b in range(0, len(A_list), num_nodes):
        A = A_list[b]
        ei = adj_to_edge_index(A)
        ei = ei + b   # ⭐ offset
        edge_indices.append(ei)
    return torch.cat(edge_indices, dim=1)


def seq_collate_fn(batch):
    """
    Returns several tensors. Tensor of lengths should not be sent to CUDA.
    """
    idx2batch  = pd.Series(np.arange(len(batch)), index = [b["idx"] for b in batch])
    df         = pd.concat(b["path"] for b in batch)
    value_cols = [c.startswith("Value") for c in df.columns]
    mask_cols  = [c.startswith("Mask") for c in df.columns]

    ## converting mask to int
    df.iloc[:, mask_cols] = df.iloc[:, mask_cols].astype(np.bool)
    df["num_obs"]         = -df.iloc[:,mask_cols].sum(1)
    df.sort_values(by=["Time", "num_obs"], inplace=True)

    ## num_obs is not a value/mask column
    value_cols.append(False)
    mask_cols.append(False)

    cov = torch.Tensor([b["cov"] for b in batch])

    batch_ids = idx2batch[df.index.values].values

    ## calculating number of events at every time
    times, counts = np.unique(df.Time.values, return_counts=True)
    time_ptr      = np.concatenate([[0], np.cumsum(counts)])
    assert df.shape[0] == time_ptr[-1]

    ## tensors for the data in the batch
    X = df.iloc[:, value_cols].values
    M = df.iloc[:, mask_cols].values

    ## selecting only observed X and splitting
    lengths = (-df.num_obs.values).tolist()
    Xsplit  = torch.split(torch.from_numpy(X[M]), lengths)
    Fsplit  = torch.split(torch.from_numpy(np.where(M)[1]), lengths)

    Xpadded = torch.nn.utils.rnn.pad_sequence(Xsplit, batch_first=True)
    Fpadded = torch.nn.utils.rnn.pad_sequence(Fsplit, batch_first=True)

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b["val_samples"] for b in batch)
        df_after.sort_values(by=["ID","Time"], inplace=True)
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val  = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:,value_cols_val].values)
        M_val = torch.tensor(df_after.iloc[:,mask_cols_val].values)
        times_val = df_after["Time"].values
        index_val = idx2batch[df_after["ID"].values].values
    else:
        X_val = None
        M_val = None
        times_val = None
        index_val = None

    res = {}
    res["times"]    = times
    res["time_ptr"] = time_ptr
    res["Xpadded"]  = Xpadded
    res["Fpadded"]  = Fpadded
    res["X"]        = torch.from_numpy(X)
    res["M"]        = torch.from_numpy(M.astype(np.float32))
    res["lengths"]  = torch.LongTensor(lengths)
    res["obs_idx"]  = torch.tensor(batch_ids)
    res["y"]        = torch.tensor([b["y"] for b in batch])
    res["cov"]      = cov

    res["X_val"]    = X_val
    res["M_val"]    = M_val
    res["times_val"]=times_val
    res["index_val"]=index_val

    return res


def extract_from_path(t_vec, p_vec, eval_times, path_idx_eval):
    '''
    Takes :
    t_vec : numpy vector of absolute times length [T]. Should be ordered.
    p_vec : numpy array of means and logvars of a trajectory at times t_vec. [T x batch_size x (2xfeatures)]
    eval_times : numpy vector of absolute times at which we want to retrieve p_vec. [L]
    path_idx_eval : index of trajectory that we want to retrieve. Should be same length of eval_times. [L]
    Returns :
    Array of dimensions [L,(2xfeatures)] of means and logvar of the required eval times and trajectories
    '''
    #Remove the evaluation after the updates. Only takes the prediction before the Bayesian update. 
    t_vec, unique_index = np.unique(t_vec,return_index=True)
    p_vec = p_vec[unique_index,:,:]

    present_mask = np.isin(eval_times, t_vec)
    eval_times[~present_mask] = map_to_closest(eval_times[~present_mask],t_vec)

    mapping = dict(zip(t_vec,np.arange(t_vec.shape[0])))

    time_idx = np.vectorize(mapping.get)(eval_times)

    return(p_vec[time_idx,path_idx_eval,:])

def map_to_closest(input,reference):
    output = np.zeros_like(input)
    for idx, element in enumerate(input):
        closest_idx=(np.abs(reference-element)).argmin()
        output[idx]=reference[closest_idx]
    return(output)

def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch > 20:
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr/3
            
def compute_corr(X_true, X_hat, Mask):
    means_true = X_true.sum(0)/Mask.sum(0)
    means_hat  = X_hat.sum(0)/Mask.sum(0)
    corr_num = ((X_true-means_true)*(X_hat-means_hat)*Mask).sum(0)
    corr_denum1 = ((X_true-means_true).pow(2)*Mask).sum(0).sqrt()
    corr_denum2 = ((X_hat-means_hat).pow(2)*Mask).sum(0).sqrt()
    return corr_num/(corr_denum1*corr_denum2)


def sort_array_on_other(x1,x2):
    """
    This function returns the permutation y needed to transform x2 in x1 s.t. x2[y]=x1
    """

    temp_dict = dict(zip(x1,np.arange(len(x1))))
    index = np.vectorize(temp_dict.get)(x2)
    perm = np.argsort(index)

    assert((x2[perm]==x1).all())

    return(perm)

def log_lik_gaussian(x,mu,logvar):
    return np.log(np.sqrt(2*np.pi)) + (logvar/2) + ((x-mu).pow(2)/(2*logvar.exp()))

def tail_fun_gaussian(x,mu,logvar):
    """
    Returns the probability that the given distribution is HIGHER than x.
    """
    return 0.5-0.5*special.erf((x-mu)/((0.5*logvar).exp()*np.sqrt(2)))