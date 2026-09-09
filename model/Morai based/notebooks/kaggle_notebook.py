# ===========================================================================
# KAGGLE NOTEBOOK - Volatility Forecasting: Moirai, Moirai2, MoiraiMoE
# ===========================================================================
# Copy tung cell vao Kaggle. File nay chi dung ASCII space, khong co U+00A0.
# ===========================================================================

# %% CELL 1 (bash cell) - chay trong bash cell tren Kaggle
# !git clone https://github.com/SalesforceAIResearch/uni2ts.git uni2ts
# !pip install hydra-core jaxtyping -q


# %% CELL 2 (python) ---
BASE = 'uni2ts/src/uni2ts/model'
open(BASE + '/moirai/__init__.py', 'w').write('from .module import MoiraiModule\n\n__all__ = ["MoiraiModule"]\n')
open(BASE + '/moirai2/__init__.py', 'w').write('from .module import Moirai2Module\n\n__all__ = ["Moirai2Module"]\n')
open(BASE + '/moirai_moe/__init__.py', 'w').write('from .module import MoiraiMoEModule\n\n__all__ = ["MoiraiMoEModule"]\n')
print('Patched: moirai, moirai2, moirai_moe __init__.py')


# %% CELL 3 (python) ---
import sys
import os

uni2ts_src = os.path.abspath('uni2ts/src')
if uni2ts_src not in sys.path:
    sys.path.insert(0, uni2ts_src)

from uni2ts.model.moirai import MoiraiModule
from uni2ts.model.moirai2 import Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEModule

print('-> Import thanh cong MoiraiModule, Moirai2Module, MoiraiMoEModule!')
print('-> Numpy version:', __import__('numpy').__version__)
print('-> Torch version:', __import__('torch').__version__)


# %% CELL 4 (python) --- VolatilityDataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizons=None):
        if horizons is None:
            horizons = [1, 3, 5, 10, 21]
        self.lookback = lookback
        self.horizons = horizons

        df = df.sort_values('time').reset_index(drop=True)
        close = df['close'].values

        returns = np.diff(np.log(close)) * 100
        self.returns = np.concatenate([[0.0], returns])

        df['returns'] = self.returns
        self.times = df['time'].values

        # Chi tao mau tu 2010-01-01 tro di (truoc 2010 la burn-in).
        # `time` is the forecast origin t; every target is strictly future
        # realized RMS volatility, matching experiments/moirai_var_aware/data.py.
        self.valid_indices = df[df['time'] >= '2010-01-01'].index.tolist()
        self.samples = []
        n = len(self.returns)

        for origin_position, t in enumerate(self.valid_indices):
            if t < lookback - 1 or t + max(horizons) >= n:
                continue

            # Observations through and including t are available at the origin.
            x = self.returns[t - lookback + 1: t + 1]
            y = []
            for h in horizons:
                # h-day realized RMS volatility from returns not observable at t.
                future_returns = self.returns[t + 1: t + h + 1]
                y.append(np.sqrt(np.mean(future_returns ** 2)))
            self.samples.append({
                'x': torch.tensor(x, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                'time': self.times[t],
                'log_return': self.returns[t + 1],
                'origin_position': origin_position,
            })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['x'], self.samples[idx]['y']


# %% CELL 5 (python) --- Split helper + SPLIT_INFO
def load_and_split_dataset(csv_path, index_name, split_info):
    df = pd.read_csv(csv_path)
    full_ds = VolatilityDataset(df)
    train_size = split_info[index_name]['train']
    val_size = split_info[index_name]['validation']
    test_size = split_info[index_name]['test']
    # Purge the final max-horizon origins of each split so that no target
    # reaches into the subsequent validation or test period.
    max_horizon = max(full_ds.horizons)
    def subset_for(start, end):
        indices = [
            i for i, sample in enumerate(full_ds.samples)
            if start <= sample['origin_position'] < end - max_horizon
        ]
        return torch.utils.data.Subset(full_ds, indices)

    train_ds = subset_for(0, train_size)
    val_ds = subset_for(train_size, train_size + val_size)
    test_ds = subset_for(train_size + val_size, train_size + val_size + test_size)
    return train_ds, val_ds, test_ds


SPLIT_INFO = {
    'DAX_40':       {'train': 1983, 'validation': 1104, 'test': 972},
    'EuroNext_100': {'train': 2005, 'validation': 1115, 'test': 979},
    'IBEX_35':      {'train': 1815, 'validation': 1362, 'test': 923},
    'KOSPI_index':  {'train': 1944, 'validation': 1109, 'test': 881},
    'Nikkei_225':   {'train': 1677, 'validation': 1174, 'test': 1062},
    'SMI':          {'train': 1987, 'validation': 1135, 'test': 901},
    'snp500':       {'train': 1988, 'validation': 1109, 'test': 927},
    'VN30_INDEX':   {'train': 1971, 'validation': 1096, 'test': 925},
    'VN_INDEX':     {'train': 1964, 'validation': 1103, 'test': 925},
}


# %% CELL 6 (python) --- VolatilityFeatureExtractor
import torch.nn as nn
import torch.nn.functional as F



class VolatilityFeatureExtractor(nn.Module):
    def __init__(self, model_type='moirai', size='small', device='cuda', weights_dir=None):
        super().__init__()
        self.model_type = model_type
        self.device = device

        # Xac dinh duong dan tai weights (Online tu HF Hub hoac Offline tu folder cuc bo)
        if weights_dir is not None:
            if model_type == 'moirai':
                pretrained_path = os.path.join(weights_dir, f"moirai-1.1-R-{size}")
            elif model_type == 'moirai2':
                pretrained_path = os.path.join(weights_dir, f"moirai-2.0-R-{size}")
            elif model_type == 'moirai_moe':
                pretrained_path = os.path.join(weights_dir, f"moirai-moe-1.0-R-{size}")
        else:
            if model_type == 'moirai':
                pretrained_path = f"Salesforce/moirai-1.1-R-{size}"
            elif model_type == 'moirai2':
                pretrained_path = f"Salesforce/moirai-2.0-R-{size}"
            elif model_type == 'moirai_moe':
                pretrained_path = f"Salesforce/moirai-moe-1.0-R-{size}"

        if model_type == 'moirai':
            from uni2ts.model.moirai import MoiraiModule
            from uni2ts.common.torch_util import packed_attention_mask
            self.backbone = MoiraiModule.from_pretrained(pretrained_path)
            self.d_model = self.backbone.d_model
            self.packed_attention_mask = packed_attention_mask
            self.max_patch = max(self.backbone.patch_sizes)
        elif model_type == 'moirai2':
            from uni2ts.model.moirai2 import Moirai2Module
            from uni2ts.common.torch_util import packed_causal_attention_mask
            self.backbone = Moirai2Module.from_pretrained(pretrained_path)
            self.d_model = self.backbone.d_model
            self.packed_causal_attention_mask = packed_causal_attention_mask
            self.max_patch = self.backbone.patch_size
        elif model_type == 'moirai_moe':
            from uni2ts.model.moirai_moe import MoiraiMoEModule
            from uni2ts.common.torch_util import packed_causal_attention_mask
            self.backbone = MoiraiMoEModule.from_pretrained(pretrained_path)
            self.d_model = self.backbone.d_model
            self.packed_causal_attention_mask = packed_causal_attention_mask
            self.max_patch = max(self.backbone.patch_sizes)
        else:
            raise ValueError("model_type must be 'moirai', 'moirai2', or 'moirai_moe'")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.backbone.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]  # 60

        # Moirai 2.0 Small yêu cầu patch_size cố định là 16.
        # Moirai 1.0 và MoiraiMoE hỗ trợ đa patch_size [8, 16, 32, 64, 128]
        patch_size_val = 16
        seq_len_patched = 4  # 64 / 16 = 4 tokens
        pad_len = 64 - seq_len  # 64 - 60 = 4

        # Pad thêm 4 số 0 vào đầu chuỗi 60 ngày để có chuỗi 64 ngày (chia hết cho patch_size 16)
        x_padded = F.pad(x, (pad_len, 0))  # B x 64
        
        # Reshape thành dạng (B, seq_len_patched, patch_size_val) -> (B, 4, 16)
        target_16 = x_padded.view(batch_size, seq_len_patched, patch_size_val).to(self.device)
        
        if self.max_patch > patch_size_val:
            target = F.pad(target_16, (0, self.max_patch - patch_size_val))
            observed_mask = torch.zeros_like(target, dtype=torch.bool)
            observed_mask[:, :, :patch_size_val] = True
        else:
            target = target_16
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        
        sample_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)
        time_id = torch.arange(seq_len_patched, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        variate_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.model_type == 'moirai2':
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                input_tokens = torch.cat([scaled_target, observed_mask.to(torch.float32)], dim=-1)
                reprs = self.backbone.in_proj(input_tokens)
                attn_mask = self.packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

            elif self.model_type == 'moirai':
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                reprs = self.backbone.in_proj(scaled_target, patch_size_tensor)
                attn_mask = self.packed_attention_mask(sample_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

            elif self.model_type == 'moirai_moe':
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                in_reprs = self.backbone.in_proj(scaled_target, patch_size_tensor)
                in_reprs = F.silu(in_reprs)
                in_reprs = self.backbone.feat_proj(in_reprs, patch_size_tensor)
                res_reprs = self.backbone.res_proj(scaled_target, patch_size_tensor)
                reprs = in_reprs + res_reprs
                attn_mask = self.packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

        z = torch.mean(reprs, dim=1)
        return z



# %% CELL 7 (python) --- VolatilityRegressionModel
class VolatilityRegressionModel(nn.Module):
    def __init__(self, extractor, hidden_dim=256, output_dim=5):
        super().__init__()
        self.extractor = extractor
        self.mlp = nn.Sequential(
            nn.Linear(extractor.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        z = self.extractor(x)
        return self.mlp(z)


# %% CELL 8 (python) --- Training loop with Early Stopping
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        model.mlp.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        print('Epoch %02d | Train: %.4f | Val: %.4f' % (epoch + 1, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.mlp.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('-> Early stopping!')
                break

    model.mlp.load_state_dict(best_state)
    return model


# %% CELL 9 (python) --- Evaluation metrics
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics_summary = {}
    for idx, h in enumerate([1, 3, 5, 10, 21]):
        p_h = preds[:, idx]
        t_h = targets[:, idx]
        p_c = np.clip(p_h, 1e-5, None)
        t_c = np.clip(t_h, 1e-5, None)
        mse = np.mean((p_h - t_h) ** 2)
        mae = np.mean(np.abs(p_h - t_h))
        q_like = np.mean((t_c ** 2) / (p_c ** 2) - np.log((t_c ** 2) / (p_c ** 2)) - 1)
        metrics_summary['h_' + str(h)] = {'MSE': mse, 'MAE': mae, 'Q-LIKE': q_like}
    return metrics_summary, preds, targets


# %% CELL 10 (python) --- Full comparison pipeline
import os

def run_comparison_pipeline(csv_path, index_name, device='cuda', weights_dir=None, output_dir='/kaggle/working/predictions'):
    print('=== INDEX: ' + index_name + ' ===')
    train_ds, val_ds, test_ds = load_and_split_dataset(csv_path, index_name, SPLIT_INFO)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    results = {}
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    for m_type in ['moirai', 'moirai2', 'moirai_moe']:
        print('--- Model: ' + m_type.upper() + ' ---')
        extractor = VolatilityFeatureExtractor(model_type=m_type, size='small', device=device, weights_dir=weights_dir)
        model = VolatilityRegressionModel(extractor=extractor)
        model = train_model(model, train_loader, val_loader, epochs=30, device=device)
        metrics, preds, targets = evaluate_model(model, test_loader, device=device)
        results[m_type] = {'metrics': metrics, 'preds': preds, 'targets': targets}

        # Lưu file CSV kết quả cho từng mô hình và từng chỉ số
        csv_rows = []
        full_ds = test_ds.dataset
        for k, idx in enumerate(test_ds.indices):
            sample = full_ds.samples[idx]
            for h_idx, h in enumerate([1, 3, 5, 10, 21]):
                csv_rows.append({
                    'time': sample['time'],
                    'log_return': sample['log_return'],
                    'horizon': h,
                    'true_volatility': targets[k, h_idx],
                    'predict_volatility': preds[k, h_idx]
                })
        df_out = pd.DataFrame(csv_rows)
        csv_filename = os.path.join(output_dir, f"{index_name}_{m_type}_predictions.csv")
        df_out.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")

    return results



# %% CELL 11 (python) --- Visualization
import matplotlib.pyplot as plt


def plot_and_print_results(results, index_name):
    for metric_name in ['MSE', 'MAE', 'Q-LIKE']:
        print(metric_name + ':')
        for h in [1, 3, 5, 10, 21]:
            row = '  h=' + str(h) + '  '
            for m_type in results:
                val = results[m_type]['metrics']['h_' + str(h)][metric_name]
                row += m_type + ': %.6f  ' % val
            print(row)

    plt.figure(figsize=(15, 6))
    plt.plot(results['moirai']['targets'][:, 4], label='Actual', color='black', linewidth=1.5)
    for m_type, color in [('moirai', 'blue'), ('moirai2', 'green'), ('moirai_moe', 'red')]:
        plt.plot(results[m_type]['preds'][:, 4], label=m_type.upper(), color=color, alpha=0.7, linestyle='--')
    plt.title('Realized Volatility h=21 - ' + index_name)
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# %% CELL 12 (python) --- Run pipeline for ALL datasets
import os

# Duong dan thu muc weights offline ban da tai len Kaggle
WEIGHTS_DIR = '/kaggle/input/datasets/trnhngv/weights/weights'

# Thu muc chua tat ca file CSV
DATASET_DIR = '/kaggle/input/volatility-dataset/dataset/'

# Thu muc luu tat ca cac file output CSV du doan
OUTPUT_DIR = '/kaggle/working/predictions'

# Chay vong lap cho tat ca cac ma (index) trong SPLIT_INFO
for index_name in SPLIT_INFO.keys():
    csv_path = os.path.join(DATASET_DIR, f"{index_name}.csv")
    
    # Kiem tra xem file co ton tai khong de tranh loi
    if os.path.exists(csv_path):
        print(f"\n{'='*50}\nBắt đầu huấn luyện cho: {index_name}\n{'='*50}")
        results = run_comparison_pipeline(
            csv_path=csv_path, 
            index_name=index_name, 
            device='cuda', 
            weights_dir=WEIGHTS_DIR,
            output_dir=OUTPUT_DIR
        )
        plot_and_print_results(results, index_name)
    else:
        print(f"Không tìm thấy file: {csv_path}")

print("\nĐã hoàn thành chạy cho toàn bộ dataset!")
print(f"Các file dự đoán được lưu tại: {OUTPUT_DIR}")
