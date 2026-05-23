"""
ダミー製造データ生成スクリプト
625件 × 62特徴量 × 目的変数2種類
"""
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
N = 625
N_FEATURES = 62

# 製造条件パラメータ名を生成
feature_names = (
    [f"temp_{i}" for i in range(1, 11)]       # 温度系 10個
    + [f"pressure_{i}" for i in range(1, 9)]   # 圧力系 8個
    + [f"speed_{i}" for i in range(1, 7)]       # 速度系 6個
    + [f"time_{i}" for i in range(1, 8)]        # 時間系 7個
    + [f"conc_{i}" for i in range(1, 9)]        # 濃度系 8個
    + [f"ratio_{i}" for i in range(1, 7)]       # 比率系 6個
    + [f"flow_{i}" for i in range(1, 7)]        # 流量系 6個
    + [f"param_{i}" for i in range(1, 12)]      # その他パラメータ 11個
)
assert len(feature_names) == N_FEATURES, f"特徴量数が{len(feature_names)}個です"

# 各特徴量の範囲を設定
feature_ranges = {}
for f in feature_names:
    if f.startswith("temp"):
        feature_ranges[f] = (100, 300)
    elif f.startswith("pressure"):
        feature_ranges[f] = (0.1, 10.0)
    elif f.startswith("speed"):
        feature_ranges[f] = (10, 500)
    elif f.startswith("time"):
        feature_ranges[f] = (5, 120)
    elif f.startswith("conc"):
        feature_ranges[f] = (0.01, 5.0)
    elif f.startswith("ratio"):
        feature_ranges[f] = (0.1, 2.0)
    elif f.startswith("flow"):
        feature_ranges[f] = (0.5, 50.0)
    else:
        feature_ranges[f] = (0, 100)

# 特徴量データを生成
data = {}
for f, (lo, hi) in feature_ranges.items():
    data[f] = np.random.uniform(lo, hi, N)

df = pd.DataFrame(data)

# 目的変数を特徴量の組み合わせから生成（非線形な関係）
t1 = df["temp_1"]
t2 = df["temp_2"]
p1 = df["pressure_1"]
s1 = df["speed_1"]
c1 = df["conc_1"]
r1 = df["ratio_1"]

df["target_property_1"] = (
    0.05 * t1
    - 0.03 * t2
    + 2.0 * p1
    - 0.002 * s1
    + 3.0 * c1
    + 1.5 * r1
    + 0.0001 * t1 * p1
    - 0.5 * np.log(s1 + 1)
    + np.random.normal(0, 0.5, N)
)

df["target_property_2"] = (
    0.02 * t1
    + 0.04 * t2
    + 1.2 * p1
    + 0.003 * s1
    + 2.0 * c1
    - 0.8 * r1
    + 0.0002 * t2 * c1
    + 0.3 * np.sqrt(np.abs(s1))
    + np.random.normal(0, 1.0, N)
)

out_dir = Path(__file__).parent / "data"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "dummy_data.csv", index=False)
print(f"ダミーデータ生成完了: {df.shape}")
print(df[["target_property_1", "target_property_2"]].describe())
