# Kế hoạch Triển khai Chi tiết trên Kaggle Notebook

Tài liệu này hướng dẫn chi tiết từng bước (Step-by-step) để bạn tự thiết lập và triển khai code dự báo realized volatility ($\sigma_{t,h}$) với 3 mô hình (**Moirai**, **Moirai2**, **MoiraiMoE**) trên **Kaggle Notebook**, bám sát theo các phân tích lý thuyết đã được duyệt.

---

## BƯỚC 1: Chuẩn bị và Tải lên Dataset lên Kaggle

Vì Kaggle không thể truy cập trực tiếp vào ổ đĩa nội bộ `D:\` của bạn, bạn cần tạo một Kaggle Dataset chứa các thư mục dữ liệu:

1.  **Nén thư mục dữ liệu:** Nén thư mục `dataset/` (chứa 9 tệp CSV của các chỉ số) thành tệp `dataset.zip`.
2.  **Tải lên Kaggle:**
    *   Truy cập vào [Kaggle Datasets](https://www.kaggle.com/datasets) -> Chọn **New Dataset**.
    *   Đặt tên Dataset (ví dụ: `volatility-dataset`).
    *   Tải tệp `dataset.zip` lên và nhấn **Create**.
    *   Kaggle sẽ tự động giải nén và tạo cấu trúc thư mục dạng:
        *   `/kaggle/input/volatility-dataset/` (hoặc `/kaggle/input/volatility-dataset/dataset/` tùy thuộc vào cách bạn nén).

---

## BƯỚC 2: Khởi tạo Notebook và Cài đặt Môi trường

1.  **Tạo Notebook:** Trong Kaggle, nhấn **New Notebook**.
2.  **Liên kết Dữ liệu:** Nhấn **+ Add Data** ở thanh menu bên phải, tìm kiếm dataset `volatility-dataset` và nhấn **Add**.
3.  **Bật GPU:** Ở phần **Settings** bên phải, mục **Accelerator** chọn **GPU T4 x2** (hoặc **GPU P100**). **Quan trọng:** Phải bật **Internet on** để có thể clone repo từ GitHub và tải weights từ Hugging Face.
4.  **Clone mã nguồn và Cài đặt thư viện (Cell 1, 2, 3):**

> [!IMPORTANT]
> **Quy trình cài đặt đã được kiểm chứng thực tế** (Verified Working Setup):
> Kaggle mặc định đã có sẵn: `torch`, `numpy 2.0.2`, `scipy`, `pandas`. Chúng ta **KHÔNG** cài `uni2ts` hay `gluonts` qua pip để tránh hạ cấp numpy. Thay vào đó: clone mã nguồn → patch `__init__.py` → mount `sys.path`.

*   *Cell 1 (bash): Clone repo và cài 2 thư viện thiếu:*
```bash
!git clone https://github.com/SalesforceAIResearch/uni2ts.git uni2ts

# module.py cần hydra-core và jaxtyping — 2 gói này không phụ thuộc gluonts, không hạ cấp numpy
!pip install hydra-core jaxtyping -q
```

*   *Cell 2 (python — GÕ TAY, không copy-paste): Patch 3 file `__init__.py`:*
```python
BASE = 'uni2ts/src/uni2ts/model'

open(BASE + '/moirai/__init__.py', 'w').write('from .module import MoiraiModule\n\n__all__ = ["MoiraiModule"]\n')
open(BASE + '/moirai2/__init__.py', 'w').write('from .module import Moirai2Module\n\n__all__ = ["Moirai2Module"]\n')
open(BASE + '/moirai_moe/__init__.py', 'w').write('from .module import MoiraiMoEModule\n\n__all__ = ["MoiraiMoEModule"]\n')

print("Patched: moirai, moirai2, moirai_moe __init__.py")
```

*   *Cell 3 (python): Nhúng đường dẫn và kiểm tra import:*
```python
import sys
import os

uni2ts_src = os.path.abspath('uni2ts/src')
if uni2ts_src not in sys.path:
    sys.path.insert(0, uni2ts_src)

from uni2ts.model.moirai import MoiraiModule
from uni2ts.model.moirai2 import Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEModule

print("-> Import thành công MoiraiModule, Moirai2Module, MoiraiMoEModule!")
print("-> Numpy version:", __import__('numpy').__version__)
print("-> Torch version:", __import__('torch').__version__)
```

**Output kỳ vọng (đã xác nhận):**
```
-> Import thành công MoiraiModule, Moirai2Module, MoiraiMoEModule!
-> Numpy version: 2.0.2
-> Torch version: 2.10.0+cu128
```

---

## BƯỚC 3: Tiền xử lý dữ liệu và Cửa sổ trượt (Cell 3 & 4)

Xây dựng class xử lý dữ liệu để tính toán log-return, realized volatility theo công thức từ bài báo, tạo sliding window và phân tách tập Train/Val/Test chính xác theo số lượng mẫu trong Table I. 

> [!TIP]
> **Giải pháp tối ưu cho Dữ liệu thô từ 2008:**
> Dữ liệu thô thực tế bắt đầu từ 2008/2009, trong khi thời gian thử nghiệm chính thức (benchmark) là từ 2010 đến 2025. 
> *   Chúng ta tính tỷ suất sinh lợi trên toàn bộ dữ liệu từ 2008 để phần dữ liệu 2008–2009 đóng vai trò là **Burn-in (khởi động)**.
> *   Các mẫu (samples) chỉ được tạo cho các ngày bắt đầu từ ngày **01/01/2010** trở đi. Nhờ đó, ngày đầu tiên của năm 2010 có đầy đủ 60 ngày tỷ suất sinh lợi lịch sử của năm 2009 làm cửa sổ lookback mà không bị khuyết thiếu (NaN).
> *   Số lượng mẫu được sinh ra sẽ trùng khớp hoàn toàn (100%) với tổng số mẫu thực tế trong Table I (ví dụ: 4059 mẫu đối với DAX 40).

*   *Cell 3: Khai báo lớp tiền xử lý dữ liệu:*
```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizons=[1, 3, 5, 10, 21]):
        self.lookback = lookback
        self.horizons = horizons
        
        # 1. Sắp xếp dữ liệu theo trình tự thời gian
        df = df.sort_values('time').reset_index(drop=True)
        close = df['close'].values
        
        # r_t = ln(P_t / P_{t-1}) * 100
        returns = np.diff(np.log(close)) * 100
        self.returns = np.concatenate([[0.0], returns])
        
        df['returns'] = self.returns
        self.times = df['time'].values
        
        # 2. Tìm tất cả các chỉ mục thuộc chu kỳ thử nghiệm (từ 2010-01-01 trở đi)
        # Phần dữ liệu trước 2010 đóng vai trò là burn-in lịch sử cho lookback
        self.valid_indices = df[df['time'] >= '2010-01-01'].index.tolist()
        self.samples = []
        n = len(self.returns)
        
        # 3. Tạo mẫu sliding window cho các chỉ mục hợp lệ
        for t in self.valid_indices:
            # x_t = [r_{t-60}, ..., r_{t-1}]
            # Đảm bảo causal setup hoàn toàn (returns quá khứ)
            x = self.returns[t - lookback : t] 
            
            y = []
            for h in horizons:
                # Trích xuất returns tương lai trong horizon h: [t, t+h-1]
                # Nếu vượt quá độ dài dữ liệu ở cuối tập test, ta chỉ lấy phần khả dụng còn lại
                end_idx = min(t + h, n)
                future_r = self.returns[t : end_idx]
                
                # Tính realized volatility: công thức (3) trong docs/data.md
                mean_r = np.mean(future_r)
                h_actual = len(future_r)
                if h_actual <= 1:
                    vol = 0.0
                else:
                    vol = np.sqrt(np.sum((future_r - mean_r) ** 2) / h_actual)
                y.append(vol)
                
            self.samples.append({
                'x': torch.tensor(x, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                'time': self.times[t]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['x'], self.samples[idx]['y']
```

*   *Cell 4: Hàm Helper chia tập Train/Val/Test theo đúng số lượng mẫu (Table I):*
```python
def load_and_split_dataset(csv_path, index_name, split_info):
    df = pd.read_csv(csv_path)
    full_ds = VolatilityDataset(df)
    
    # Số lượng mẫu quy định từ Table I (tương ứng chính xác số lượng dòng >= 2010-01-01)
    train_size = split_info[index_name]['train']
    val_size = split_info[index_name]['validation']
    test_size = split_info[index_name]['test']
    
    # Chia cắt trực tiếp theo chỉ mục liên tục để tránh rò rỉ chéo thời gian
    train_ds = torch.utils.data.Subset(full_ds, range(0, train_size))
    val_ds = torch.utils.data.Subset(full_ds, range(train_size, train_size + val_size))
    test_ds = torch.utils.data.Subset(full_ds, range(train_size + val_size, train_size + val_size + test_size))
    
    return train_ds, val_ds, test_ds

# Tra cứu kích thước mẫu từ Table I
SPLIT_INFO = {
    'DAX_40': {'train': 1983, 'validation': 1104, 'test': 972},
    'EuroNext_100': {'train': 2005, 'validation': 1115, 'test': 979},
    'IBEX_35': {'train': 1815, 'validation': 1362, 'test': 923},
    'KOSPI_index': {'train': 1944, 'validation': 1109, 'test': 881},
    'Nikkei_225': {'train': 1677, 'validation': 1174, 'test': 1062},
    'SMI': {'train': 1987, 'validation': 1135, 'test': 901},
    'snp500': {'train': 1988, 'validation': 1109, 'test': 927},
    'VN30_INDEX': {'train': 1971, 'validation': 1096, 'test': 925},
    'VN_INDEX': {'train': 1964, 'validation': 1103, 'test': 925}
}
```

---

## BƯỚC 4: Định nghĩa Mô hình & Trích xuất Đặc trưng (Cell 5)

Chúng ta xây dựng một class PyTorch Wrapper thực hiện:
1.  Tải weights mô hình Transformer nền tảng và đóng băng (freeze).
2.  Nhận chuỗi tỷ suất sinh lợi đầu vào và đẩy qua mô hình chuỗi thời gian để trích xuất embeddings ẩn của tokens quá khứ.
3.  Áp dụng Average Pooling trên chiều tokens để có biểu diễn cố định độ rộng.

*   *Cell 5: Khai báo Mô hình Trích xuất Đặc trưng:*
```python
import os
import torch.nn as nn
import torch.nn.functional as F

class VolatilityFeatureExtractor(nn.Module):
    def __init__(self, model_type='moirai', size='small', device='cuda', weights_dir=None):
        super().__init__()
        self.model_type = model_type
        self.device = device
        
        # Xác định đường dẫn tải mô hình (Online từ HF hoặc Offline từ thư mục cục bộ)
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
        
        # Tải mô hình tương ứng
        if model_type == 'moirai':
            from uni2ts.model.moirai import MoiraiModule
            self.backbone = MoiraiModule.from_pretrained(pretrained_path)
            self.d_model = self.backbone.d_model
        elif model_type == 'moirai2':
            from uni2ts.model.moirai2 import Moirai2Module
            self.backbone = Moirai2Module.from_pretrained(pretrained_path)
            self.d_model = self.backbone.d_model
        elif model_type == 'moirai_moe':
            from uni2ts.model.moirai_moe import MoiraiMoEModule
            self.backbone = MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{size}")
            self.d_model = self.backbone.d_model
        else:
            raise ValueError("Mô hình không hợp lệ!")
            
        # Đóng băng backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.backbone.to(device)

    def forward(self, x):
        # x: Batch_size x Lookback (B x 60)
        batch_size = x.shape[0]
        seq_len = x.shape[1]  # 60

        # Moirai 2.0 Small yêu cầu patch_size cố định là 16.
        # Moirai 1.0 và MoiraiMoE hỗ trợ đa patch_size [8, 16, 32, 64, 128] nên dùng 16 cũng hoàn toàn hợp lệ.
        patch_size_val = 16
        seq_len_patched = 4  # 64 / 16 = 4 tokens
        pad_len = 64 - seq_len  # 4

        # Pad thêm 4 số 0 vào đầu chuỗi 60 ngày để có chuỗi 64 ngày (chia hết cho patch_size 16)
        x_padded = nn.functional.pad(x, (pad_len, 0))  # B x 64
        
        # Reshape thành dạng (B, seq_len_patched, patch_size) -> (B, 4, 16)
        target = x_padded.view(batch_size, seq_len_patched, patch_size_val).to(self.device)
        observed_mask = torch.ones_like(target, dtype=torch.bool)
        
        sample_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)
        time_id = torch.arange(seq_len_patched, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        variate_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.model_type == 'moirai2':
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                input_tokens = torch.cat([scaled_target, observed_mask.to(torch.float32)], dim=-1)  # B x 4 x 32
                reprs = self.backbone.in_proj(input_tokens)
                from uni2ts.common.torch_util import packed_causal_attention_mask
                attn_mask = packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

            elif self.model_type == 'moirai':
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                reprs = self.backbone.in_proj(scaled_target, patch_size_tensor)
                from uni2ts.common.torch_util import packed_attention_mask
                attn_mask = packed_attention_mask(sample_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

            elif self.model_type == 'moirai_moe':
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                in_reprs = self.backbone.in_proj(scaled_target, patch_size_tensor)
                in_reprs = nn.functional.silu(in_reprs)
                in_reprs = self.backbone.feat_proj(in_reprs, patch_size_tensor)
                res_reprs = self.backbone.res_proj(scaled_target, patch_size_tensor)
                reprs = in_reprs + res_reprs
                from uni2ts.common.torch_util import packed_causal_attention_mask
                attn_mask = packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)

        # Average Pooling trên chiều tokens (seq_len_patched)
        z = torch.mean(reprs, dim=1) # B x d_model
        return z
```

---

## BƯỚC 5: Thiết lập và Huấn luyện đầu hồi quy MLP chung (Cell 6 & 7)

*   *Cell 6: Định nghĩa MLP Head và mạng ghép nối:*
```python
class VolatilityRegressionModel(nn.Module):
    def __init__(self, extractor, hidden_dim=256, output_dim=5):
        super().__init__()
        self.extractor = extractor
        
        # MLP Head chung nhận d_model làm đầu vào và ánh xạ ra 5 chân trời dự báo
        self.mlp = nn.Sequential(
            nn.Linear(extractor.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim) # Đầu ra 5 chiều
        )

    def forward(self, x):
        # 1. Trích xuất đặc trưng chuỗi thời gian
        z = self.extractor(x)
        # 2. Đi qua MLP Head để dự đoán volatility
        out = self.mlp(z)
        return out
```

*   *Cell 7: Vòng lặp huấn luyện chuẩn có Early Stopping:*
```python
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 7
    patience_counter = 0
    
    for epoch in range(epochs):
        # Mode train (chỉ cho phần MLP Head, backbone vẫn eval)
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
        
        # Mode validation
        model.mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Kiểm tra Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.mlp.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("-> Dừng sớm (Early stopping) do val loss không giảm!")
                break
                
    # Load lại MLP Head tốt nhất
    model.mlp.load_state_dict(best_state)
    return model
```

---

## BƯỚC 6: Đánh giá và Tính toán Metrics (Cell 8)

Định nghĩa metrics đánh giá bao gồm MSE, MAE, và đặc biệt là Q-LIKE Loss dành riêng cho độ biến động tài chính.

*   *Cell 8: Hàm đánh giá hiệu năng:*
```python
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
            
    preds = np.concatenate(all_preds, axis=0) # Mẫu x 5
    targets = np.concatenate(all_targets, axis=0) # Mẫu x 5
    
    horizons = [1, 3, 5, 10, 21]
    metrics_summary = {}
    
    for idx, h in enumerate(horizons):
        p_h = preds[:, idx]
        t_h = targets[:, idx]
        
        # Tránh chia cho 0 hoặc log số âm trong Q-LIKE
        # Ta clip các giá trị dự đoán nhỏ dưới ngưỡng 1e-5
        p_h_clipped = np.clip(p_h, 1e-5, None)
        t_h_clipped = np.clip(t_h, 1e-5, None)
        
        # 1. MSE
        mse = np.mean((p_h - t_h) ** 2)
        # 2. MAE
        mae = np.mean(np.abs(p_h - t_h))
        # 3. Q-LIKE (tính theo phương sai thực tế vs phương sai dự báo)
        # Q-LIKE = actual^2 / pred^2 - ln(actual^2 / pred^2) - 1
        q_like = np.mean((t_h_clipped**2) / (p_h_clipped**2) - np.log((t_h_clipped**2) / (p_h_clipped**2)) - 1)
        
        metrics_summary[f"h_{h}"] = {"MSE": mse, "MAE": mae, "Q-LIKE": q_like}
        
    return metrics_summary, preds, targets
```

---

## BƯỚC 7: Pipeline Chạy và Tổng hợp Kết quả (Cell 9)

Chạy huấn luyện và đánh giá lần lượt cho cả 3 mô hình trên từng chỉ số, lưu trữ kết quả để vẽ bảng so sánh tổng hợp.

*   *Cell 9: Hàm điều phối chạy so sánh 3 mô hình:*
```python
def run_comparison_pipeline(csv_path, index_name, device='cuda', weights_dir=None):
    print(f"================ CHẠY SO SÁNH TRÊN CHỈ SỐ: {index_name} ================")
    
    # 1. Load và chia tách dữ liệu
    train_ds, val_ds, test_ds = load_and_split_dataset(csv_path, index_name, SPLIT_INFO)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    models_to_test = ['moirai', 'moirai2', 'moirai_moe']
    results = {}
    
    for m_type in models_to_test:
        print(f"\n--- Đang huấn luyện mô hình: {m_type.upper()} ---")
        # Khởi tạo extractor và mô hình hồi quy
        extractor = VolatilityFeatureExtractor(model_type=m_type, size='small', device=device, weights_dir=weights_dir)
        model = VolatilityRegressionModel(extractor=extractor)
        
        # Huấn luyện
        model = train_model(model, train_loader, val_loader, epochs=30, device=device)
        
        # Đánh giá
        metrics, preds, targets = evaluate_model(model, test_loader, device=device)
        results[m_type] = {
            'metrics': metrics,
            'preds': preds,
            'targets': targets
        }

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
        csv_filename = f"{index_name}_{m_type}_predictions.csv"
        df_out.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")
        
    return results
```

---

## BƯỚC 8: Trực quan hóa và Kết luận (Cell 10 & 11)

Vẽ biểu đồ đường so sánh giữa Volatility thực tế vs Volatility dự báo của 3 mô hình trên tập Test tại chân trời $h=21$ để thấy rõ sự khác biệt.

*   *Cell 10: In bảng và vẽ đồ thị:*
```python
import matplotlib.pyplot as plt

def plot_and_print_results(results, index_name):
    # 1. In bảng so sánh
    print(f"\nBẢNG KẾT QUẢ SO SÁNH TRÊN CHỈ SỐ {index_name}:")
    for metric_name in ["MSE", "MAE", "Q-LIKE"]:
        print(f"\n-> Độ đo: {metric_name}")
        row_str = f"{'Horizon':<10}"
        for m_type in results.keys():
            row_str += f" | {m_type.upper():<12}"
        print(row_str)
        print("-" * len(row_str))
        
        for h in [1, 3, 5, 10, 21]:
            h_key = f"h_{h}"
            h_str = f"h = {h:<7}"
            for m_type in results.keys():
                val = results[m_type]['metrics'][h_key][metric_name]
                h_str += f" | {val:<12.6f}"
            print(h_str)

    # 2. Vẽ đồ thị so sánh bám đuổi (chọn h = 21)
    plt.figure(figsize=(15, 6))
    targets_21 = results['moirai']['targets'][:, 4] # target thực tế h=21
    plt.plot(targets_21, label='Actual Volatility', color='black', alpha=0.8, linewidth=1.5)
    
    colors = {'moirai': 'blue', 'moirai2': 'green', 'moirai_moe': 'red'}
    for m_type in results.keys():
        preds_21 = results[m_type]['preds'][:, 4]
        plt.plot(preds_21, label=f'Pred {m_type.upper()}', color=colors[m_type], alpha=0.6, linestyle='--')
        
    plt.title(f"Realized Volatility Forecasting Comparison (Horizon h=21) - {index_name}")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

*   *Cell 11: Thực thi chạy mô phỏng:*
```python
# Chỉnh sửa đường dẫn cho đúng với Kaggle input của bạn
CSV_PATH = '/kaggle/input/volatility-dataset/dataset/DAX_40.csv'
INDEX_NAME = 'DAX_40'

# Đường dẫn thư mục weights offline bạn đã tải lên Kaggle
WEIGHTS_DIR = '/kaggle/input/datasets/trnhngv/weights/weights'

results = run_comparison_pipeline(CSV_PATH, INDEX_NAME, device='cuda', weights_dir=WEIGHTS_DIR)
plot_and_print_results(results, INDEX_NAME)
```
