"""
製造条件 逆解析支援ツール - Streamlit アプリ
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from analysis import (
    optimize_target1_goal_predict_target2,
    make_feature_range_table,
    fit_knn_distance_reference,
    judge_extrapolation_level,
    check_range,
    get_knn_value,
    find_nearest_records,
)

st.set_page_config(page_title="製造条件 逆解析支援ツール", layout="wide")
st.title("製造条件 逆解析支援ツール")


@st.cache_resource
def load_models():
    model_1 = joblib.load("models/best_model_1_stable_29feat.pkl")
    model_2 = joblib.load("models/best_model_2_stable_29feat.pkl")
    features = joblib.load("models/best_features_stable_29feat.pkl")
    X_base = joblib.load("models/X_base.pkl")
    return model_1, model_2, features, X_base


try:
    model_1, model_2, best_features, X_base = load_models()
except FileNotFoundError:
    st.error("モデルファイルが見つかりません。先に `train_model.py` を実行してください。")
    st.stop()

range_table = make_feature_range_table(X_base, best_features)
knn_reference = fit_knn_distance_reference(X_base, best_features)

# --- サイドバー: 入力 ---
st.sidebar.header("製造条件")

target1_goal = st.sidebar.number_input(
    "target_property_1 目標値",
    value=15.0,
    step=0.5,
    format="%.2f",
)

n_trials = st.sidebar.slider("Optunaトライアル数", 100, 2000, 500, step=100)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 固定条件（任意）")

fixed_conditions = {}
for feat in best_features[:5]:
    use_fixed = st.sidebar.checkbox(f"{feat} を固定", key=f"fix_{feat}")
    if use_fixed:
        lo = float(X_base[feat].quantile(0.05))
        hi = float(X_base[feat].quantile(0.95))
        val = st.sidebar.slider(feat, lo, hi, float(X_base[feat].median()), key=f"val_{feat}")
        fixed_conditions[feat] = val

run = st.sidebar.button("逆解析を実行", type="primary")

# --- メイン画面 ---
if run:
    with st.spinner("Optunaで最適条件を探索中..."):
        result = optimize_target1_goal_predict_target2(
            model_1=model_1,
            model_2=model_2,
            best_features=best_features,
            base_df=X_base,
            target1_goal=target1_goal,
            fixed_conditions_light=fixed_conditions,
            n_trials=n_trials,
        )

    best_params = result["best_params"]

    # 予測結果サマリー
    st.subheader("予測結果")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("target_property_1 目標値", f"{result['target1_goal']:.4f}")
    col2.metric("target_property_1 予測値", f"{result['pred_1']:.4f}")
    col3.metric("target1 誤差", f"{result['target1_error']:.4f}")
    col4.metric("target_property_2 参考予測", f"{result['pred_2_reference']:.4f}")

    # Feasibility Check
    st.subheader("Feasibility Summary")
    range_check_df = check_range(best_params, range_table)
    knn_value = get_knn_value(best_params, best_features, knn_reference)
    level, message = judge_extrapolation_level(range_check_df, knn_value, knn_reference)

    level_color = {"SAFE": "success", "CAUTION": "warning", "WARNING": "error"}
    getattr(st, level_color[level])(f"**{level}**: {message}")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    fcol1.metric("extrapolation_level", level)
    fcol2.metric("minmax_out_count", int(range_check_df["minmax_out"].sum()))
    fcol3.metric("q05q95_out_count", int(range_check_df["q05q95_out"].sum()))
    fcol4.metric("knn_distance", f"{knn_value:.3f}")

    with st.expander("特徴量ごとの範囲チェック詳細"):
        st.dataframe(range_check_df.style.apply(
            lambda row: ["background-color: #ffcccc" if row["minmax_out"] else "" for _ in row],
            axis=1
        ), use_container_width=True)

    # 近傍実績表示
    st.subheader("近傍実績（過去の類似条件）")
    nearest_df = find_nearest_records(best_params, best_features, X_base, n_neighbors=5)
    st.dataframe(nearest_df.reset_index(drop=True), use_container_width=True)

    # 提案条件との比較
    st.subheader("提案条件")
    proposed_df = pd.DataFrame([best_params]).T.rename(columns={0: "提案値"})
    proposed_df["提案値"] = proposed_df["提案値"].round(4)
    st.dataframe(proposed_df, use_container_width=True)

else:
    st.info("サイドバーで目標値を設定し、「逆解析を実行」ボタンを押してください。")

    st.subheader("学習データの分布（先頭10件）")
    st.dataframe(X_base[best_features].head(10), use_container_width=True)
