"""
逆解析・Feasibility Check・近傍実績表示のユーティリティ
"""
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_target1_goal_predict_target2(
    model_1,
    model_2,
    best_features,
    base_df,
    target1_goal=15,
    fixed_conditions_light=None,
    n_trials=1000,
    random_seed=42,
):
    if fixed_conditions_light is None:
        fixed_conditions_light = {}

    search_features = [
        col for col in best_features if col not in fixed_conditions_light
    ]

    def objective(trial):
        params = {}
        for col in search_features:
            low = float(base_df[col].quantile(0.05))
            high = float(base_df[col].quantile(0.95))
            params[col] = trial.suggest_float(col, low, high)

        for k, v in fixed_conditions_light.items():
            params[k] = v

        X_trial = pd.DataFrame([params])[best_features]
        pred_1 = model_1.predict(X_trial)[0]
        loss = (pred_1 - target1_goal) ** 2
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    for k, v in fixed_conditions_light.items():
        best_params[k] = v

    X_best = pd.DataFrame([best_params])[best_features]
    return {
        "target1_goal": target1_goal,
        "pred_1": model_1.predict(X_best)[0],
        "target1_error": abs(model_1.predict(X_best)[0] - target1_goal),
        "pred_2_reference": model_2.predict(X_best)[0],
        "best_params": best_params,
        "study": study,
    }


def make_feature_range_table(X_base, features):
    X_ref = X_base[features].copy()
    return pd.DataFrame({
        "feature": features,
        "min": X_ref.min().values,
        "q05": X_ref.quantile(0.05).values,
        "q95": X_ref.quantile(0.95).values,
        "max": X_ref.max().values,
    })


def fit_knn_distance_reference(X_base, features, n_neighbors=5):
    X_ref = X_base[features].copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ref)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)

    distances, _ = nn.kneighbors(X_scaled)
    train_knn_dist = distances[:, 1:].mean(axis=1)

    return {
        "features": features,
        "scaler": scaler,
        "nn": nn,
        "knn_q95": np.quantile(train_knn_dist, 0.95),
        "knn_q99": np.quantile(train_knn_dist, 0.99),
        "knn_max": np.max(train_knn_dist),
    }


def judge_extrapolation_level(range_check_df, knn_value, knn_reference):
    minmax_out_count = int(range_check_df["minmax_out"].sum())
    q05q95_out_count = int(range_check_df["q05q95_out"].sum())

    if minmax_out_count > 0 or knn_value > knn_reference["knn_q99"]:
        level = "WARNING"
        message = "既存分布外の可能性が高いです。"
    elif q05q95_out_count >= 3 or knn_value > knn_reference["knn_q95"]:
        level = "CAUTION"
        message = "一部外れ気味です。条件妥当性を確認してください。"
    else:
        level = "SAFE"
        message = "既存実績に比較的近い条件です。"

    return level, message


def check_range(best_params, range_table):
    rows = []
    for _, row in range_table.iterrows():
        feat = row["feature"]
        val = best_params.get(feat, np.nan)
        rows.append({
            "feature": feat,
            "value": val,
            "min": row["min"],
            "max": row["max"],
            "q05": row["q05"],
            "q95": row["q95"],
            "minmax_out": not (row["min"] <= val <= row["max"]),
            "q05q95_out": not (row["q05"] <= val <= row["q95"]),
        })
    return pd.DataFrame(rows)


def get_knn_value(best_params, features, knn_reference):
    scaler = knn_reference["scaler"]
    nn = knn_reference["nn"]
    x_query = np.array([[best_params[f] for f in features]], dtype=float)
    x_query_scaled = scaler.transform(x_query)
    distances, _ = nn.kneighbors(x_query_scaled)
    return float(distances[0].mean())


def find_nearest_records(best_params, features, X_base, n_neighbors=5):
    X_ref = X_base[features].copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ref)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)

    x_query = np.array([[best_params[f] for f in features]], dtype=float)
    x_query_scaled = scaler.transform(x_query)
    distances, indices = nn.kneighbors(x_query_scaled)

    result_df = X_base.iloc[indices[0]].copy()
    result_df["distance"] = distances[0]
    return result_df
