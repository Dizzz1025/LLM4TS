import torch
import math
import numpy as np
from torchdiffeq import odeint

from torch.nn.utils.rnn import pack_padded_sequence

# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time
            h        hidden state (current)

        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class FullGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, x, h):
        """
        Executes one step with GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class FullGRUODECell_Autonomous(torch.nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        # self.lin_mz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_mn = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        
    # def graph_message(self, h, edge_index):
    #     # h: [N, H]
    #     # edge_index: [2, E]
    #     num_nodes = 5
    #     B = h.shape[0] // num_nodes
    #     H = h.shape[1]
    #     N = num_nodes
    #     h = h.view(B, N, H)
    #     agg = torch.zeros_like(h)
    #     src, dst = edge_index
    #     for b in range(B):
    #         msg = h[b, src] - h[b, dst]
    #         agg[b].index_add_(0, dst, msg)
    #     return agg.view(B * N, H)    
    def graph_message(self, h, edge_index):
        # h: [N, H]
        # edge_index: [2, E]
        # print("edge_index max:", edge_index.max().item())
        src, dst = edge_index
        msg = h[src] - h[dst]           # 相对量（关键）
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return agg    


    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)
        return dh

class GRUObservationCellLogvar(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs        = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c    = np.log(np.sqrt(2*np.pi))
        losses       = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)
        if losses.sum()!=losses.sum():
            import ipdb; ipdb.set_trace()

        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses

class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs     = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        var       = torch.abs(var) + self.var_eps
        error     = (X_obs - mean) / torch.sqrt(var)

        ## log normal loss, over all observations
        loss         = 0.5 * ((torch.pow(error, 2) + torch.log(var)) * M_obs).sum()


        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, var, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

# class TimeMixing(torch.nn.Module):
#     """
#     Time-only Mixing
#     Input : [B, L, H]
#     Output: [B, L, H]
#     """
#     def __init__(self, L, dropout=0.0):
#         super().__init__()
#         self.time_linear = torch.nn.Linear(L, L, bias=True)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.norm = torch.nn.LayerNorm(L)

#     def forward(self, x):
#         # x: [B, L, H]
#         # B, L, H = x.shape
#         # 把时间维度拿出来混
#         x_t = x.permute(0, 2, 1)      # [B, H, L]
#         y_t = self.time_linear(x_t)   # [B, H, L]
#         y_t = self.dropout(y_t)
#         # residual + norm（在 time 维）
#         y_t = self.norm(y_t + x_t)

#         return y_t.permute(0, 2, 1)   # [B, L, H]
# class TimeMixingStack(torch.nn.Module):
#     def __init__(self, L, num_layers=3, dropout=0.0):
#         super().__init__()
#         self.layers = torch.nn.ModuleList([
#             TimeMixing(L, dropout) for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         # x: [B, L, H]
#         for layer in self.layers:
#             x = layer(x)
#         return x
class TimeProjection(torch.nn.Module):
    """
    Project time dimension: L -> T_pred
    """
    def __init__(self, L, T_pred):
        super().__init__()
        self.proj = torch.nn.Linear(L, T_pred, bias=True)

    def forward(self, x):
        # x: [B, L, H]
        x = x.permute(0, 2, 1)    # [B, H, L]
        x = self.proj(x)          # [B, H, T]
        return x.permute(0, 2, 1) # [B, T, H]

# class TimeMixerForecast(torch.nn.Module):
#     def __init__(self, L, T_pred, num_layers=3, dropout=0.0):
#         super().__init__()
#         self.time_mixer = TimeMixingStack(L, num_layers, dropout)
#         self.time_proj  = TimeProjection(L, T_pred)

#     def forward(self, latent_seq):
#         """
#         latent_seq: [B, L, H]
#         return:     [B, T_pred, H]
#         """
#         x = self.time_mixer(latent_seq)
#         pred = self.time_proj(x)
#         return pred

class TimeMixingDilatedConv(torch.nn.Module):
    def __init__(self, H, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv1d(
            H, H,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(H)

    def forward(self, x):
        # x: [B, L, H]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(y)
        return self.norm(x + y)

class TimeMixingDilatedStack(torch.nn.Module):
    def __init__(self, H, num_layers=4, kernel_size=3, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TimeMixingDilatedConv(
                    H,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class TimeMixerForecastDilatedConv(torch.nn.Module):
    def __init__(self, L, T_pred, H,
                 num_layers=4, kernel_size=3, dropout=0.0):
        super().__init__()
        self.time_mixer = TimeMixingDilatedStack(
            H=H,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.time_proj = TimeProjection(L, T_pred)

    def forward(self, latent_seq):
        # latent_seq: [B, L, H]
        x = self.time_mixer(latent_seq)   # [B, L, H]
        pred = self.time_proj(x)          # [B, T_pred, H]
        return pred


class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, full_gru_ode=False, solver="euler", impute = True, stage=None, dataset=None, **options):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super().__init__()

        self.impute = impute
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )
        # self.p_model1 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_size, p_hidden, bias=bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout_rate),
        #     torch.nn.Linear(p_hidden, input_size, bias=bias),
        # )
        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden,1,bias=bias)
        )
        if full_gru_ode:
            if impute is False:
                print('choose!')
                self.gru_c   = FullGRUODECell_Autonomous(hidden_size, bias = bias)
            else:
                self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)

        else:
            if impute is False:
                self.gru_c = GRUODECell_Autonomous(hidden_size, bias= bias)
            else:
                self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)

        if logvar:
            self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        else:
            self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )

        assert solver in ["euler", "midpoint", "dopri5"], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        self.solver     = solver
        self.store_hist = options.pop("store_hist",False)
        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)
        
        self.group_attn = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=5, batch_first=True)
        self.group_attn_ln = torch.nn.LayerNorm(hidden_size)
        self.graph_corr = GraphCorrection(hidden_size)
        self.graph_lambda = 0.005   # 可以从 0.01 ~ 0.1 试
        self.graph_gate = torch.nn.Sequential(torch.nn.Linear(2 * hidden_size, hidden_size), torch.nn.Sigmoid())
        
        if dataset == 'Sim':
            self.T_pred = 60
            self.L_resample = 120  # 你规则采样得到的 latent 长度
        if dataset == 'nyc':
            self.T_pred = 72
            self.L_resample = 216  # 你规则采样得到的 latent 长度
        
        self.time_proj_weight = torch.nn.Parameter(torch.randn(self.T_pred, self.L_resample))
        self.time_proj_bias = torch.nn.Parameter(torch.zeros(self.T_pred))
        self.forecast_head = TimeMixerForecastDilatedConv(L=self.L_resample, T_pred=self.T_pred, H=hidden_size, num_layers=5, dropout=0.1)
        self.stage = stage
        self.latent_cache = {}

    def _apply_group_attention(self, h, group_idx, K=5):
        """
        h: [B, H]
        group_idx: [B]  (int)
        """
        if group_idx is None:
            return h
        B, H = h.shape
        perm = torch.argsort(group_idx)
        inv_perm = torch.argsort(perm)
        
        h_sorted = h[perm]
        g_sorted = group_idx[perm]
        G = B // K
        assert (g_sorted.view(G, K) == g_sorted.view(G, K)[:, :1]).all().item(), 'Group中的样本不属于同一个Group_ID!'
        x = h_sorted.view(G, K, H)
        out, _ = self.group_attn(x, x, x)
        x = self.group_attn_ln(x + out)
        
        h_sorted = x.view(B, H)
        h = h_sorted[inv_perm]
        return h
    def graph_message(self, h, edge_index):
        # h: [N, H]
        # edge_index: [2, E]
        # print("edge_index max:", edge_index.max().item())
        src, dst = edge_index
        msg = h[src] - h[dst]           # 相对量（关键）
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return agg   
    def ode_step(self, h, p, delta_t, current_time):
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        if self.impute is False:
            p = torch.zeros_like(p)
            
        if self.solver == "euler":
            # h = h + delta_t * self.gru_c(p, h)
            # agg = self.graph_corr(h, self.edge_index)
            # g = self.graph_gate(h)
            # h = h + self.graph_lambda * g * agg
            # p = self.p_model(h)
            h0 = h
            dh = self.gru_c(p, h0)
            h = h0 + delta_t * dh
            
            agg = self.graph_corr(h0, self.edge_index)
            g = self.graph_gate(torch.cat([h0, agg], dim=-1))
            h = h + self.graph_lambda * g * agg
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
            solution, eval_times, eval_vals = odeint(self.gru_c,h,torch.tensor([0,delta_t]),method=self.solver,options={"store_hist":self.store_hist})
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1,:,:]
            p = self.p_model(h)
        
        current_time += delta_t
        return h,p,current_time, eval_times, eval_ps

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, T_val=None,
                group_idx=None, return_path=False, smoother = False, class_criterion = None, labels=None, edge_index=None, test=False, return_cache=False):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """
        T_val = self.L_resample - self.T_pred
        self.edge_index = edge_index
        h = self.covariates_map(cov) # [batch_size, cov_dim]
        # print('h.shape:', h.shape)
        assert group_idx is None or group_idx.shape[0] == h.shape[0]
        
        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1 = 0 #Pre-jump loss
        loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)
        # if return_cache:
        #     h_cache = []
        #     # mask = np.zeros((self.T_pred+self.L_resample, h.shape[0]))
        #     mask_cache = []
        #     gt = []
        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0],device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0],device = h.device)
            class_criterion = class_criterion
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total = torch.tensor([],dtype = torch.float64, device = h.device)
        eval_vals_total  = torch.tensor([],dtype = torch.float32, device = h.device)

        forecasting = False
        if not test: 
            for i, obs_time in enumerate(times):
                if obs_time == 0:
                    continue
                if (not forecasting) and (obs_time > T_val):
                    forecasting = True
                    forecasting_idx = i
                    # break
                ## Propagation of the ODE until next observation
                # dt = obs_time - current_time
                while current_time < obs_time:     
                    if self.solver == "dopri5":
                        # h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, dt, current_time)
                        raise NotImplementedError("dopri5 solver is not implemented for one-step integration.")
                    else:
                        h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time) 
                    eval_times_total = torch.cat((eval_times_total, eval_times))
                    eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                    #Storing the predictions.
                    if return_path and current_time < obs_time:
                        path_t.append(current_time)
                        path_p.append(p)
                        path_h.append(h)
                    if return_cache:
                        h_cache.append(h)
                        mask.append(torch.ones(h.shape[0],device=h.device))
                        gt.append(torch.zeros(h.shape[0],self.input_size,device=h.device))
                # assert abs(current_time - obs_time) < 1e-5
                ## Reached an observation
                start = time_ptr[i]
                end   = time_ptr[i+1]

                X_obs = X[start:end]
                M_obs = M[start:end]
                i_obs = obs_idx[start:end]
                assert i_obs.unique().numel() == i_obs.numel(), "i_obs有重复值!"
                if not forecasting:
                    ## Using GRUObservationCell to update h. Also updating p and loss
                    h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
                    if smoother:
                        class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                        num_evals_vec[i_obs] +=1
                    if losses.sum()!=losses.sum():
                        import ipdb;ipdb.set_trace()
                    loss_1    = loss_1+ losses.sum()
                    p         = self.p_model(h)

                    loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)
                else:
                    p = self.p_model(h)
                    # loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)
                    # break
                
                if return_path:
                    path_t.append(obs_time)
                    path_p.append(p)
                    path_h.append(h) # len = 61, path_h[0].shape=[500, 50], T,B,H
            current_time = obs_time
            # 预防times里的时间戳没到T
            while current_time < T - 1:
                h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)      
            latent_seq = torch.stack(path_h, dim=1)  # [B, L, H]
            pred = self.forecast_head(latent_seq)  # [B, T_pred, H]
            pred_seq = pred.permute(1, 0, 2) # [T_pred, B, H]
            path_h = []
            path_t = []
            path_p = []
            t0 = latent_seq.shape[1] * delta_t
            for t in range(pred_seq.shape[0]):
                path_h.append(pred_seq[t])    # 每个是 [B, H]
                path_t.append(t0 + delta_t * t)
                p_pred = self.p_model(pred_seq[t])
                path_p.append(p_pred)
            t_pred = 0
            # forecasting_idx = 61
            for j in range(forecasting_idx, len(times)):
                start = time_ptr[j]
                end   = time_ptr[j + 1]

                X_obs = X[start:end]
                M_obs = M[start:end]
                i_obs = obs_idx[start:end]
                h_pred = pred_seq[t_pred]
                p_pred = self.p_model(h_pred)
                # mean, var = torch.chunk(p_pred[i_obs], 2, dim=1)
                # loss_2 += ((X_obs - mean) ** 2 * M_obs).sum() / (M_obs.sum() + 1e-8) / self.mixing
                loss_2 = loss_2 + compute_KL_loss(p_obs = p_pred[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)
                t_pred += 1                
            loss = loss_1 + self.mixing * loss_2
            
        if test: 
            # mask = np.zeros((self.T_pred+self.L_resample, h.shape[0]))
            for i, obs_time in enumerate(times):
                if obs_time == 0:
                    continue
                if (not forecasting) and (obs_time > T_val):
                    forecasting = True
                    forecasting_idx = i
                    # break
                ## Propagation of the ODE until next observation
                # dt = obs_time - current_time
                while current_time < obs_time:     
                    if self.solver == "dopri5":
                        # h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, dt, current_time)
                        raise NotImplementedError("dopri5 solver is not implemented for one-step integration.")
                    else:
                        h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time) 
                    eval_times_total = torch.cat((eval_times_total, eval_times))
                    eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                    #Storing the predictions.
                    if return_path and current_time < obs_time:
                        path_t.append(current_time)
                        path_p.append(p)
                        path_h.append(h)
                    # if return_cache and current_time < obs_time:
                    #     h_cache.append(h)
                    #     mask[int(current_time/delta_t),:] = 1
                    #     gt.append(torch.zeros(h.shape[0],self.input_size,device=h.device))
                # assert abs(current_time - obs_time) < 1e-5
                ## Reached an observation
                start = time_ptr[i]
                end   = time_ptr[i+1]

                X_obs = X[start:end]
                M_obs = M[start:end]
                i_obs = obs_idx[start:end]
                t_idx = int(float(obs_time) / float(delta_t))
                # mask[t_idx, i_obs] = 1
                # gt[t_idx, i_obs, :] = X_obs
                assert i_obs.unique().numel() == i_obs.numel(), "i_obs有重复值!"
                if not forecasting:
                    ## Using GRUObservationCell to update h. Also updating p and loss
                    h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
                    if smoother:
                        class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                        num_evals_vec[i_obs] +=1
                    if losses.sum()!=losses.sum():
                        import ipdb;ipdb.set_trace()
                    loss_1    = loss_1+ losses.sum()
                    p         = self.p_model(h)

                    loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)
                else:
                    p = self.p_model(h)
            
                if return_path:
                    path_t.append(obs_time)
                    path_p.append(p)
                    path_h.append(h) # len = 61, path_h[0].shape=[500, 50], T,B,H
            assert current_time == obs_time, 'time error'
            while current_time < T - 1:     
                if self.solver == "dopri5":
                    # h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, dt, current_time)
                    raise NotImplementedError("dopri5 solver is not implemented for one-step integration.")
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time) 
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)        
            latent_seq = torch.stack(path_h, dim=1)  # [B, L, H]
            pred = self.forecast_head(latent_seq)  # [B, T_pred, H]
            pred_seq = pred.permute(1, 0, 2) # [T_pred, B, H]
            # path_h = []
            path_t = []
            path_p = []
            forecasting_idx = len(times)
            t0 = forecasting_idx * delta_t
            path_h = path_h[:t0]
            for t in range(pred_seq.shape[0]):
                path_h.append(pred_seq[t])    # 每个是 [B, H]
                path_t.append(t0 + delta_t * t)
                p_pred = self.p_model(pred_seq[t])
                path_p.append(p_pred)

            loss = loss_1 + self.mixing * loss_2
     
        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        
        class_pred = self.classification_model(h)
       
        if return_path:
            if smoother:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
            else:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), eval_times_total, eval_vals_total
        else:
            if smoother:
                return h, loss, class_pred, class_loss_vec
            else:
                return h, loss, class_pred, loss_1

    def set_train_stage(self, stage: int):
        """
        stage = 1 : 只训练 GRU-ODE
        stage = 2 : 只训练 Time Conv
        """
        self.stage = stage
        for name, p in self.named_parameters():
            if stage == 1:
                # ===== Stage 1: GRU-ODE =====
                if "forecast_head" in name:
                    # p.requires_grad = False
                    p.requires_grad = True
                else:
                    p.requires_grad = True

            elif stage == 2:
                # ===== Stage 2: Time Conv =====
                if "forecast_head" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
                # for p in self.p_model1.parameters():
                #     p.requires_grad = True


def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

    return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)

class GraphCorrection(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h, edge_index):
        """
        h: [B*N, H]  每个节点一个隐状态
        edge_index: [2, E]（已经 batch offset）
        """
        src, dst = edge_index   # src -> dst

        rel = h[src] - h[dst]   # 相对量（关键）
        msg = self.phi(rel) - self.phi(-rel)
        # msg = self.phi(rel)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        return agg


class GRUODEBayesSeq(torch.nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, obs_noise_std=1e-2, full_gru_ode=False):
        super().__init__()
        self.obs_noise_std = obs_noise_std
        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, classification_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden, 1, bias=bias),
        )
        if full_gru_ode:
            self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)
        else:
            self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)

        self.gru_bayes = SeqGRUBayes(input_size=input_size, hidden_size=hidden_size, prep_hidden=prep_hidden, p_hidden=p_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
        )
        self.input_size = input_size
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.
        self.apply(init_weights)

    def forward(self, times, time_ptr, Xpadded, Fpadded, X, M, lengths,
                obs_idx, delta_t, T, cov, return_path=False):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            Xpadded    data tensor (padded)
            Fpadded    feature id of each data point (padded)
            X          observation tensor
            M          mask tensor
            obs_idx    observed patients of each datapoint (current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        h       = self.covariates_map(cov) # [batch_size, cov_dim]
        p       = self.gru_bayes.p_model(h)
        time    = 0.0
        counter = 0

        loss_1 = 0 # Pre-jump loss
        loss_2 = 0 # Post-jump loss

        if return_path:
            path_t = [0]
            path_p = [p]

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until obs_time
            while time < obs_time:
                h = h + delta_t * self.gru_c(p, h)
                p = self.gru_bayes.p_model(h)

                ## using counter to avoid numerical errors
                counter += 1
                time = counter * delta_t
                ## Storing the predictions.
                if return_path:
                    path_t.append(time)
                    path_p.append(p)

            ## Reached obs_time
            start = time_ptr[i]
            end   = time_ptr[i+1]

            L_obs = lengths[start:end]
            X_obs = pack_padded_sequence(Xpadded[start:end], L_obs, batch_first=True)
            F_obs = pack_padded_sequence(Fpadded[start:end], L_obs, batch_first=True)
            i_obs = obs_idx[start:end]

            Xf_batch = X[start:end]
            Mf_batch = M[start:end]

            ## Using GRU-Bayes to update h. Also updating p and loss
            h, loss_i, loss_pre = self.gru_bayes(h, X_obs, F_obs, i_obs, X=Xf_batch, M=Mf_batch)
            loss_1    = loss_1 + loss_i + loss_pre.sum()
            p         = self.gru_bayes.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = Xf_batch, M_obs = Mf_batch, obs_noise_std=self.obs_noise_std)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)

        while time < T:
            h = h + delta_t * self.gru_c(p, h)
            p = self.gru_bayes.p_model(h)

            counter += 1
            time = counter * delta_t
            if return_path:
                path_t.append(time)
                path_p.append(p)

        loss = loss_1 + self.mixing * loss_2
        class_pred = self.classification_model(h)
        if return_path:
            return h, loss, class_pred, np.array(path_t), torch.stack(path_p)
        return h, loss, class_pred


class SeqGRUBayes(torch.nn.Module):
    """

    Inputs to forward:
        h      tensor of hiddens
        X_obs  PackedSequence of observation values
        F_obs  PackedSequence of feature ids
        i_obs  indices of h that have been observed

    Returns updated h.
    """
    def __init__(self, input_size, hidden_size, prep_hidden, p_hidden, bias=True):
        super().__init__()
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )
        self.gru = torch.nn.GRUCell(prep_hidden, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6

    def p_mean_logvar(self, h):
        p      = self.p_model(h)
        mean, logvar = torch.chunk(p, 2, dim=1)
        return mean, logvar

    def step_1feature(self, hidden, X_step, F_step):
        ## 1) Compute error on the observed features
        mean, logvar = self.p_mean_logvar(hidden)
        ## mean, logvar both are [ Batch  x  input_size ]
        hrange = torch.arange(hidden.shape[0])
        mean   = mean[   hrange, F_step ]
        logvar = logvar[ hrange, F_step ]

        sigma  = torch.exp(0.5 * logvar)
        error  = (X_step - mean) / sigma

        ## log normal loss, over all observations
        loss   = 0.5 * (torch.pow(error, 2) + logvar).sum()

        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_step, mean, logvar, error], dim=1).unsqueeze(1)
        ## 2) Select the matrices from w_prep and bias_prep; multiply
        W         = self.w_prep[F_step, :, :]
        bias      = self.bias_prep[F_step]
        gru_input = torch.matmul(gru_input, W).squeeze(1) + bias
        gru_input.relu_()

        return self.gru(gru_input, hidden), loss

    def ode_step(self, h, p, delta_t):
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)
            return h, p

        if self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h2 = h + delta_t * self.gru_c(pk, k)
            p2 = self.p_model(h2)
            return h2, p2

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, h, X_obs, F_obs, i_obs, X, M):
        """
        See https://github.com/pytorch/pytorch/blob/a462edd0f6696a4cac4dd04c60d1ad3c9bc0b99c/torch/nn/_functions/rnn.py#L118-L154
        """
        ## selecting h to be updated
        hidden = h[i_obs]

        output          = []
        input_offset    = 0
        last_batch_size = X_obs.batch_sizes[0]
        hiddens         = []
        #flat_hidden     = not isinstance(hidden, tuple)


        ## computing loss before any updates
        mean, logvar = self.p_mean_logvar(hidden)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X - mean) / sigma
        losses_pre   = 0.5 * ((torch.pow(error, 2) + logvar) * M)

        ## updating hidden
        loss = 0
        input_offset = 0
        for batch_size in X_obs.batch_sizes:
            X_step = X_obs.data[input_offset:input_offset + batch_size]
            F_step = F_obs.data[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(hidden[-dec:])
                hidden = hidden[:-dec]
            last_batch_size = batch_size

            hidden, loss_b = self.step_1feature(hidden, X_step, F_step)
            loss = loss + loss_b

        hiddens.append(hidden)
        hiddens.reverse()

        hidden = torch.cat(hiddens, dim=0)

        ## updating observed trajectories
        h2        = h.clone()
        h2[i_obs] = hidden

        return h2, loss, losses_pre

class Discretized_GRU(torch.nn.Module):
    ## Discretized GRU model (GRU-ODE-Bayes without ODE and without Bayes)
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, impute=True):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        """
        
        super().__init__()
        self.impute = impute
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden,1,bias=bias)
        )

        self.gru = torch.nn.GRUCell(2*input_size, hidden_size, bias = bias)

        if logvar:
            self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        else:
            self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )


        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def ode_step(self, h, p, delta_t):
 
        h = self.gru(p,h)

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov,
                return_path=False, smoother = False, class_criterion = None, labels=None):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """


        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1 = 0 #Pre-jump loss
        loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0],device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0],device = h.device)
            class_criterion = class_criterion
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                
                if self.impute is False:
                    p = torch.zeros_like(p)
                h = self.gru(p, h)
                p = self.p_model(h)

                ## using counter to avoid numerical errors
                counter += 1
                current_time = counter * delta_t
                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
           
            if smoother:
                class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                num_evals_vec[i_obs] +=1
            loss_1    = loss_1 + losses.sum()
            p         = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)
                path_h.append(h)


        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.impute is False:
                p = torch.zeros_like(p)
            h = self.gru(p,h)
            p = self.p_model(h)

            counter += 1
            current_time = counter * delta_t
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = loss_1 + self.mixing * loss_2

        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        class_pred = self.classification_model(h)
       
        if return_path:
            if smoother:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
            else:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h)
        else:
            if smoother:
                return h, loss, class_pred, class_loss_vec
            else:
                return h, loss, class_pred
