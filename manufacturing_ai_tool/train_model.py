"""
LightGBMによる物性予測モデル学習スクリプト
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "data" / "dummy_data.csv")

TARGET_COLS = ["target_property_1", "target_property_2"]
feature_cols = [c for c in df.columns if c not in TARGET_COLS]

X = df[feature_cols]
y1 = df["target_property_1"]
y2 = df["target_property_2"]

# 外れ値除去（IQR法）
q1, q3 = y1.quantile([0.25, 0.75])
iqr = q3 - q1
mask = (y1 >= q1 - 3 * iqr) & (y1 <= q3 + 3 * iqr)
X, y1, y2 = X[mask], y1[mask], y2[mask]
print(f"学習データ数: {len(X)}件")

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, test_size=0.2, random_state=42
)

# --- 特徴量選択 ---
selector_model = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
selector_model.fit(X_train, y1_train)

selector = SelectFromModel(selector_model, max_features=29, prefit=True)
best_features = X_train.columns[selector.get_support()].tolist()
print(f"選択された特徴量数: {len(best_features)}")

X_train_sel = X_train[best_features]
X_test_sel = X_test[best_features]

# --- モデル学習 ---
model_1 = LGBMRegressor(random_state=42, verbose=-1)
model_2 = LGBMRegressor(random_state=42, verbose=-1)

model_1.fit(X_train_sel, y1_train)
model_2.fit(X_train_sel, y2_train)

pred1 = model_1.predict(X_test_sel)
pred2 = model_2.predict(X_test_sel)

print("\n--- モデル精度 ---")
print(f"target_property_1: MSE={mean_squared_error(y1_test, pred1):.3f}, R2={r2_score(y1_test, pred1):.3f}")
print(f"target_property_2: MSE={mean_squared_error(y2_test, pred2):.3f}, R2={r2_score(y2_test, pred2):.3f}")

# --- 保存 ---
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)
joblib.dump(model_1, model_dir / "best_model_1_stable_29feat.pkl")
joblib.dump(model_2, model_dir / "best_model_2_stable_29feat.pkl")
joblib.dump(best_features, model_dir / "best_features_stable_29feat.pkl")
joblib.dump(X[best_features], model_dir / "X_base.pkl")
print("\nモデル・特徴量を保存しました。")
