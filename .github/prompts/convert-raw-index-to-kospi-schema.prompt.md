---
name: convert-raw-index-to-kospi-schema-sample
description: Chuyen CSV raw kieu Yahoo Finance (multi-header) thanh schema giong KOSPI_index.csv
---

# Muc tieu
Viet cac ham Python de xu ly du lieu chi so chung khoan co cau truc raw nhu DAX_40.csv (dang multi-header tu Yahoo Finance) va chuyen ve schema chuan:
`time,close,open,high,low,volume,return_1_day`

# Dau vao
- File nguon: `${input:source_csv}`
- File dich: `${input:target_csv}`
- Cot gia dong cua dung de tinh return: `close`
- Dinh dang ngay uu tien: `%Y-%m-%d`

# Yeu cau bat buoc
1. Ho tro 2 truong hop file nguon:
- Da la single header binh thuong.
- Multi-header 3 dong kieu Yahoo (`Price/...`, `Ticker/...`, `Date/...`).
2. Chuan hoa ten cot ve lowercase snake_case.
3. Mapping cot ve schema dich:
- Date -> time
- Close -> close - float
- Open -> open - float
- High -> high - float
- Low -> low - float
- Volume -> volume - float
4. Chuyen `time` sang datetime, sap xep tang dan theo `time`, bo dong `time` khong hop le.
5. Chuyen cac cot so (`close, open, high, low, volume`) ve numeric (coerce loi thanh NaN).
6. Tinh `return_1_day = np.log(df['close'] / df['close'].shift(1))`.
7. Luu CSV theo dung thu tu cot:
`time,close,open,high,low,volume,return_1_day`
8. Khong xoa du lieu hop le; khong hard-code ma chi so (`^GDAXI`, `^GSPC`, ...).

# Dau ra mong muon
- Tao cac ham ro rang, tai su dung duoc, toi thieu gom:
- `load_raw_index_csv(path)`
- `normalize_ohlcv_schema(df)`
- `add_return_1_day(df)`
- `convert_raw_to_kospi_schema(source_csv, target_csv)`
- Co doan usage mau (1-3 dong) de goi ham.
- Neu dang sua notebook, chi xuat code cell Python (khong giai thich dai).

# Tieu chi chat luong
- Code Python de doc, co xu ly loi co ban (`FileNotFoundError`, file rong, thieu cot bat buoc).
- Khong dung thu vien ngoai `pandas` va `numpy` (neu khong can thiet).
- Comment ngan gon cho logic kho (vi du phat hien multi-header).

# Kiem tra nhanh (bat buoc in ra)
- So dong truoc/sau khi xu ly.
- Min/max cua cot `time`.
- So gia tri null moi cot sau chuan hoa.



# Main prompt:
Do not create MD
Do not create other code for example or test
