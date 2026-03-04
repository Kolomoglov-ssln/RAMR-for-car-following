import os
import warnings
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================================
# --- 0. Global Configurations ---
# =====================================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================================
# --- 1. Optimal Hyperparameters & Constants ---
# =====================================================================================
# Hyperparameters derived from the HPO experiments detailed in the manuscript (Table 3)
OPTIMAL_HPARAMS_PER_PREDICTOR = {
    "MLP": {
        'resid_code_dim': 64, 'main_model_hidden_dim': 96, 'beta_lr': 0.005138,
        'gamma_lr': 0.000244, 'main_model_updates_per_meta_iter': 9, 'weight_reg_coeff': 0.004699
    },
    "GRU": {
        'resid_code_dim': 96, 'main_model_hidden_dim': 96, 'beta_lr': 0.004110,
        'gamma_lr': 0.008254, 'main_model_updates_per_meta_iter': 9, 'weight_reg_coeff': 0.056367
    },
    "CNN1D": {
        'resid_code_dim': 96, 'main_model_hidden_dim': 32, 'beta_lr': 0.003607,
        'gamma_lr': 0.000734, 'main_model_updates_per_meta_iter': 8, 'weight_reg_coeff': 0.002485
    }
}

STATE_FEATURES_DIM = 3
STATE_FEATURE_KEYS_FROM_FILE = ['relative_speed', 'spacing', 'ego_speed']
IDM_ACCEL_FEATURE_DIM = 1
RESIDUAL_ENCODER_INPUT_DIM = 1
MAIN_MODEL_OUTPUT_DIM = 1
RESIDUAL_SEQ_WINDOW_T = 10
BATCH_SIZE = 64
META_ITERATIONS = 150
ALPHA_INNER_LR_MAIN_MODEL = 1e-3

# Data Paths (Ensure these directories and files exist in your repository)
BASE_DATA_DIR = '.'
EGO_ACCEL_FILE = os.path.join(BASE_DATA_DIR,
                              'results_ego_acceleration_waymo_structured/waymo_ego_acceleration_ego_accelerations.npz')
FEATURES_FILE = os.path.join(BASE_DATA_DIR,
                             'results_analysis_extended_Waymo_structured_GIPPS/Waymo_Gipps_analysis_final_extracted_features.npz')
RESIDUALS_FILE = os.path.join(BASE_DATA_DIR,
                              'results_analysis_extended_Waymo_structured_GIPPS/Waymo_Gipps_analysis_final_residuals.npz')

# =====================================================================================
# --- 2. Infinite Iterator for Meta-Learning Data Flow ---
# =====================================================================================
def get_infinite_iterator(dataloader):
    """
    Creates an infinite generator from a PyTorch DataLoader to prevent StopIteration
    and avoid validation set meta-overfitting during the extensive bi-level updates.
    """
    while True:
        for batch in dataloader:
            yield batch


# =====================================================================================
# --- 3. Dataset Parsing and Loading ---
# =====================================================================================
def create_dataset_from_keys(traj_keys, npz_files, residual_seq_window, state_feature_keys):
    """Parses raw npz dictionaries into PyTorch TensorDatasets matching the RAMR input structure."""
    ego_accels_npz, features_npz, residuals_npz = npz_files
    all_x, all_idm, all_human, all_r = [], [], [], []

    for key in tqdm(traj_keys, desc="    Extracting Trajectories", leave=False):
        try:
            a_h = np.asarray(ego_accels_npz[key]).astype(np.float32).flatten()
            r_t = np.asarray(residuals_npz[key]).astype(np.float32).flatten()
            min_len = min(len(a_h), len(r_t))
            a_h, r_t = a_h[:min_len], r_t[:min_len]

            # Reconstruct baseline physical prediction: a_base = a_human - residual
            a_i = a_h - r_t

            f_dict = features_npz[key].item()
            x_comps = [np.asarray(f_dict[f_key]).astype(np.float32).flatten() for f_key in state_feature_keys]
            min_len = min(min_len, min(len(c) for c in x_comps))
            a_h, r_t, a_i = a_h[:min_len], r_t[:min_len], a_i[:min_len]
            x_aligned = [c[:min_len] for c in x_comps]
            x_traj = np.stack(x_aligned, axis=1)

            if x_traj.shape[0] < residual_seq_window + 1: continue

            for t in range(residual_seq_window, x_traj.shape[0]):
                all_x.append(x_traj[t])
                all_idm.append(a_i[t])
                all_human.append(a_h[t])
                all_r.append(r_t[t - residual_seq_window:t].reshape(-1, 1))
        except Exception as e:
            warnings.warn(f"Warning: Error processing trajectory {key}: {e}. Skipped.")

    if not all_x:
        return None

    dataset = TensorDataset(
        torch.tensor(np.array(all_x), dtype=torch.float32),
        torch.tensor(np.array(all_idm), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(all_human), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(all_r), dtype=torch.float32)
    )
    return dataset


# =====================================================================================
# --- 4. Model Architectures (Supports Functional Forward Pass) ---
# =====================================================================================
def _get_combined_input(x_state, a_idm_pred, z_resid):
    features = [x_state]
    if a_idm_pred is not None: features.append(a_idm_pred if a_idm_pred.ndim > 1 else a_idm_pred.unsqueeze(1))
    if z_resid is not None: features.append(z_resid if z_resid.ndim > 1 else z_resid.unsqueeze(1))
    return torch.cat(features, dim=1)


class MainPredictorMLP(nn.Module):
    def __init__(self, state_feat_dim, idm_feat_dim, resid_code_dim, hidden_dim, output_dim):
        super().__init__()
        input_dim = state_feat_dim + idm_feat_dim + resid_code_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_state, a_idm_pred, z_resid, params_list=None):
        combined_input = _get_combined_input(x_state, a_idm_pred, z_resid)
        if params_list is None:
            x = F.relu(self.fc1(combined_input));
            x = F.relu(self.fc2(x));
            return self.fc3(x)
        else:
            x = F.relu(F.linear(combined_input, params_list[0], params_list[1]))
            x = F.relu(F.linear(x, params_list[2], params_list[3]))
            return F.linear(x, params_list[4], params_list[5])


class CNN1D_Predictor(nn.Module):
    def __init__(self, state_feat_dim, idm_feat_dim, resid_code_dim, hidden_dim, output_dim):
        super().__init__()
        input_dim = state_feat_dim + idm_feat_dim + resid_code_dim
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_state, a_idm_pred, z_resid, params_list=None):
        combined_input = _get_combined_input(x_state, a_idm_pred, z_resid).unsqueeze(1)
        if params_list is None:
            x = F.relu(self.conv1(combined_input));
            x = F.relu(self.conv2(x));
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x));
            return self.fc2(x)
        else:
            x = F.relu(F.conv1d(combined_input, params_list[0], params_list[1], padding=1))
            x = F.relu(F.conv1d(x, params_list[2], params_list[3], padding=1));
            x = x.view(x.size(0), -1)
            x = F.relu(F.linear(x, params_list[4], params_list[5]));
            return F.linear(x, params_list[6], params_list[7])


class GRU_Predictor(nn.Module):
    def __init__(self, state_feat_dim, idm_feat_dim, resid_code_dim, hidden_dim, output_dim):
        super().__init__()
        input_dim = state_feat_dim + idm_feat_dim + resid_code_dim
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_state, a_idm_pred, z_resid, params_list=None):
        combined_input = _get_combined_input(x_state, a_idm_pred, z_resid)
        h_0 = torch.zeros(combined_input.size(0), self.hidden_dim, device=combined_input.device)
        if params_list is None:
            return self.fc_out(self.gru_cell(combined_input, h_0))
        else:
            gi = F.linear(combined_input, params_list[0], params_list[2])
            gh = F.linear(h_0, params_list[1], params_list[3])
            i_r, i_z, i_n = gi.chunk(3, 1);
            h_r, h_z, h_n = gh.chunk(3, 1)
            resetgate = torch.sigmoid(i_r + h_r);
            updategate = torch.sigmoid(i_z + h_z)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_1 = newgate + updategate * (h_0 - newgate)
            return F.linear(h_1, params_list[4], params_list[5])


class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, r_sequence):
        if r_sequence.ndim == 2: r_sequence = r_sequence.unsqueeze(-1)
        rnn_out, _ = self.rnn(r_sequence)
        return self.fc(rnn_out[:, -1, :])


class WeightingNetwork(nn.Module):
    def __init__(self, z_resid_dim, loss_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(z_resid_dim + loss_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z_resid, current_loss_per_sample):
        if current_loss_per_sample.ndim == 1: current_loss_per_sample = current_loss_per_sample.unsqueeze(1)
        out = F.relu(self.fc1(torch.cat([z_resid, current_loss_per_sample], dim=1)))
        return torch.sigmoid(self.fc2(out))


# =====================================================================================
# --- 5. RAMR Bi-level Optimization Core ---
# =====================================================================================
def train_step_ramr(main_model, res_encoder, weight_net, opt_main, opt_meta, train_iter, val_iter, config, loss_fn):
    main_model.train();
    res_encoder.train();
    weight_net.train()

    # Phase A: Meta-Update
    x_mt, a_idm_mt, y_mt, r_seq_mt = next(train_iter)
    x_mv, a_idm_mv, y_mv, r_seq_mv = next(val_iter)

    x_mt, a_idm_mt, y_mt, r_seq_mt = [d.to(DEVICE) for d in [x_mt, a_idm_mt, y_mt, r_seq_mt]]
    x_mv, a_idm_mv, y_mv, r_seq_mv = [d.to(DEVICE) for d in [x_mv, a_idm_mv, y_mv, r_seq_mv]]

    with torch.no_grad():
        z_mt = res_encoder(r_seq_mt)
        l_i = loss_fn(main_model(x_mt, a_idm_mt, z_mt), y_mt)

    w_mt = weight_net(z_mt.detach(), l_i.detach())

    z_mt_grad = res_encoder(r_seq_mt)
    l_i_prime = loss_fn(main_model(x_mt, a_idm_mt, z_mt_grad), y_mt)
    L_prime = (w_mt * l_i_prime).mean()

    grads = torch.autograd.grad(L_prime, main_model.parameters(), create_graph=True)
    fast_params = [p - config['alpha_inner_lr'] * g for p, g in zip(main_model.parameters(), grads)]

    z_mv = res_encoder(r_seq_mv)
    L_meta = F.mse_loss(main_model(x_mv, a_idm_mv, z_mv, params_list=fast_params), y_mv)

    if config['weight_reg_coeff'] > 0:
        L_meta += config['weight_reg_coeff'] * -torch.mean(w_mt * torch.log(w_mt.clamp(1e-8)))

    opt_meta.zero_grad();
    L_meta.backward();
    opt_meta.step()

    # Phase B: Main Model Update
    for _ in range(config['main_model_updates_per_meta_iter']):
        x_tr, a_idm_tr, y_tr, r_seq_tr = next(train_iter)
        x_tr, a_idm_tr, y_tr, r_seq_tr = [d.to(DEVICE) for d in [x_tr, a_idm_tr, y_tr, r_seq_tr]]

        with torch.no_grad():
            z_tr = res_encoder(r_seq_tr)
            l_i_tr = loss_fn(main_model(x_tr, a_idm_tr, z_tr), y_tr)
            w_tr = weight_net(z_tr, l_i_tr)

        l_main = loss_fn(main_model(x_tr, a_idm_tr, z_tr.detach()), y_tr)
        L_main = (w_tr.detach() * l_main).mean()

        opt_main.zero_grad();
        L_main.backward();
        opt_main.step()


def evaluate_final_model(main_model, residual_encoder, data_loader):
    main_model.eval();
    residual_encoder.eval()
    total_mse, total_mae, total_samples = 0, 0, 0
    with torch.no_grad():
        for x_state, a_idm, a_human, r_seq in data_loader:
            x_state, a_idm, a_human, r_seq = [d.to(DEVICE) for d in [x_state, a_idm, a_human, r_seq]]
            predictions = main_model(x_state, a_idm, residual_encoder(r_seq))
            total_mse += F.mse_loss(predictions, a_human, reduction='sum').item()
            total_mae += F.l1_loss(predictions, a_human, reduction='sum').item()
            total_samples += x_state.size(0)
    return total_mae / total_samples, total_mse / total_samples


# =====================================================================================
# --- 6. Main Execution Pipeline ---
# =====================================================================================
if __name__ == '__main__':
    print(f"--- Initialization RAMR Core Framework | Device: {DEVICE} ---")

    # 1. Load Real Processed Dataset
    print("\n[Step 1] Loading processed .npz datasets extracted from FollowNet...")
    try:
        npz_files = (
            np.load(EGO_ACCEL_FILE, allow_pickle=True),
            np.load(FEATURES_FILE, allow_pickle=True),
            np.load(RESIDUALS_FILE, allow_pickle=True)
        )
    except FileNotFoundError as e:
        print(f"Error: Required dataset files not found. Ensure .npz files are in {BASE_DATA_DIR}\nDetails: {e}")
        exit()

    common_traj_keys = sorted(list(set(npz_files[0].files) & set(npz_files[1].files) & set(npz_files[2].files)))
    np.random.shuffle(common_traj_keys)

    # 70% Train / 10% Validation / 20% Test Split logic
    n_total = len(common_traj_keys)
    n_test = int(0.20 * n_total)
    n_val = int(0.10 * n_total)
    test_keys = common_traj_keys[:n_test]
    val_keys = common_traj_keys[n_test: n_test + n_val]
    train_keys = common_traj_keys[n_test + n_val:]

    print(f"Trajectory Split -> Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)}")

    train_ds = create_dataset_from_keys(train_keys, npz_files, RESIDUAL_SEQ_WINDOW_T, STATE_FEATURE_KEYS_FROM_FILE)
    val_ds = create_dataset_from_keys(val_keys, npz_files, RESIDUAL_SEQ_WINDOW_T, STATE_FEATURE_KEYS_FROM_FILE)
    test_ds = create_dataset_from_keys(test_keys, npz_files, RESIDUAL_SEQ_WINDOW_T, STATE_FEATURE_KEYS_FROM_FILE)

    if not all([train_ds, val_ds, test_ds]):
        print("Error: Failed to construct datasets from provided keys.");
        exit()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    train_iter = get_infinite_iterator(train_loader)
    val_iter = get_infinite_iterator(val_loader)
    print(f"Sample instances -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    # 2. Setup Predictors Map
    predictors_to_test = {"MLP": MainPredictorMLP, "CNN1D": CNN1D_Predictor, "GRU": GRU_Predictor}
    all_results = []

    # 3. Execution Loop
    main_progress_bar = tqdm(predictors_to_test.items(), desc="Training Predictors")
    for predictor_name, predictor_class in main_progress_bar:
        main_progress_bar.set_postfix_str(f"Optimizing: {predictor_name}")

        config = {**OPTIMAL_HPARAMS_PER_PREDICTOR[predictor_name],
                  'meta_iterations': META_ITERATIONS, 'alpha_inner_lr': ALPHA_INNER_LR_MAIN_MODEL}

        main_predictor = predictor_class(STATE_FEATURES_DIM, IDM_ACCEL_FEATURE_DIM,
                                         config['resid_code_dim'], config['main_model_hidden_dim'],
                                         MAIN_MODEL_OUTPUT_DIM).to(DEVICE)
        residual_encoder = ResidualEncoder(RESIDUAL_ENCODER_INPUT_DIM, 32, config['resid_code_dim']).to(DEVICE)
        weighting_network = WeightingNetwork(config['resid_code_dim'], loss_dim=1, hidden_dim=16).to(DEVICE)

        optimizer_main = optim.Adam(main_predictor.parameters(), lr=config['gamma_lr'])
        optimizer_meta = optim.Adam(list(weighting_network.parameters()) + list(residual_encoder.parameters()),
                                    lr=config['beta_lr'])
        mse_loss_unreduced = nn.MSELoss(reduction='none')

        for epoch in range(config['meta_iterations']):
            train_step_ramr(main_predictor, residual_encoder, weighting_network,
                            optimizer_main, optimizer_meta, train_iter, val_iter, config, mse_loss_unreduced)

        final_mae, final_mse = evaluate_final_model(main_predictor, residual_encoder, test_loader)
        all_results.append({"Predictor Architecture": predictor_name, "Final MAE": final_mae, "Final MSE": final_mse})

    # 4. Results Summary
    print("\n" + "=" * 60)
    print("### Framework Execution Complete - Results Summary ###")
    print("=" * 60)
    results_df = pd.DataFrame(all_results)
    pd.set_option('display.float_format', '{:.5f}'.format)
    print(results_df.to_string(index=False))