import os
from huggingface_hub import snapshot_download

# Tao thu muc chua weights
os.makedirs('weights', exist_ok=True)

models = {
    "moirai-1.1-R-small": "Salesforce/moirai-1.1-R-small",
    "moirai-2.0-R-small": "Salesforce/moirai-2.0-R-small",
    "moirai-moe-1.0-R-small": "Salesforce/moirai-moe-1.0-R-small"
}

print("=== BAT DAU TAI WEIGHTS VE MAY LOCAL ===")

for name, repo_id in models.items():
    print(f"\n-> Dang tai {name} tu repo {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=os.path.join('weights', name),
            # Chỉ tải các file cấu hình và trọng số cần thiết để tiết kiệm dung lượng
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.onnx", "*.pb"]
        )
        print(f"-> Hoan thanh tai {name}!")
    except Exception as e:
        print(f"-> Loi khi tai {name}: {e}")
        print("Goi y: Neu bi gioi han bang thong, ban co the dang nhap bang cach chay 'huggingface-cli login' truoc khi chay script nay.")

print("\n=== DA TAI XONG TOAN BO WEIGHTS ===")
print("Thu muc 'weights' da san sang.")
print("Buoc tiep theo: Nen zip thu muc 'weights' lai va upload len Kaggle Dataset!")
