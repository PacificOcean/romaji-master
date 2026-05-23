"""
製造条件 逆解析支援ツール v2
- タブ切り替え: 逆解析モード / 順方向予測モード
- compare_df: 逆解析結果の蓄積・比較
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

st.set_page_config(page_title="製造条件 逆解析支援ツール v2", layout="wide")
st.title("製造条件 逆解析支援ツール v2")

# --- モデル読み込み ---
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

# --- session_state 初期化 ---
if "compare_results" not in st.session_state:
    st.session_state.compare_results = []
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

# ===========================
# タブ定義
# ===========================
tab_inv, tab_fwd, tab_compare = st.tabs([
    "🔄 逆解析モード",
    "🔮 順方向予測モード",
    "📊 結果比較 (compare_df)",
])

# ===========================
# タブ1: 逆解析モード
# ===========================
with tab_inv:
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        st.subheader("製造条件設定")
        target1_goal = st.number_input(
            "target_property_1 目標値",
            value=15.0, step=0.5, format="%.2f", key="inv_target"
        )
        n_trials = st.slider("Optunaトライアル数", 100, 2000, 500, step=100, key="inv_trials")

        st.markdown("---")
        st.markdown("**固定条件（任意）**")
        fixed_conditions = {}
        for feat in best_features[:5]:
            use_fixed = st.checkbox(f"{feat} を固定", key=f"inv_fix_{feat}")
            if use_fixed:
                lo = float(X_base[feat].quantile(0.05))
                hi = float(X_base[feat].quantile(0.95))
                val = st.slider(feat, lo, hi, float(X_base[feat].median()), key=f"inv_val_{feat}")
                fixed_conditions[feat] = val

        run_inv = st.button("逆解析を実行", type="primary", key="btn_inv")

    with col_main:
        if run_inv:
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
            range_check_df = check_range(best_params, range_table)
            knn_value = get_knn_value(best_params, best_features, knn_reference)
            level, message = judge_extrapolation_level(range_check_df, knn_value, knn_reference)

            # 予測結果
            st.subheader("予測結果")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("target_property_1 目標値", f"{result['target1_goal']:.4f}")
            c2.metric("target_property_1 予測値", f"{result['pred_1']:.4f}")
            c3.metric("target1 誤差", f"{result['target1_error']:.4f}")
            c4.metric("target_property_2 参考予測", f"{result['pred_2_reference']:.4f}")

            # Feasibility
            st.subheader("Feasibility Summary")
            level_color = {"SAFE": "success", "CAUTION": "warning", "WARNING": "error"}
            getattr(st, level_color[level])(f"**{level}**: {message}")

            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("extrapolation_level", level)
            fc2.metric("minmax_out_count", int(range_check_df["minmax_out"].sum()))
            fc3.metric("q05q95_out_count", int(range_check_df["q05q95_out"].sum()))
            fc4.metric("knn_distance", f"{knn_value:.3f}")

            with st.expander("特徴量ごとの範囲チェック詳細"):
                st.dataframe(range_check_df, use_container_width=True)

            # 近傍実績
            st.subheader("近傍実績（過去の類似条件）")
            nearest_df = find_nearest_records(best_params, best_features, X_base, n_neighbors=5)
            st.dataframe(nearest_df.reset_index(drop=True), use_container_width=True)

            # 提案条件
            with st.expander("提案条件（全特徴量）"):
                proposed_df = pd.DataFrame([best_params]).T.rename(columns={0: "提案値"})
                proposed_df["提案値"] = proposed_df["提案値"].round(4)
                st.dataframe(proposed_df, use_container_width=True)

            # compare_dfへ蓄積
            st.session_state.run_count += 1
            st.session_state.compare_results.append({
                "実行No": st.session_state.run_count,
                "目標値(p1)": round(target1_goal, 4),
                "予測値(p1)": round(result["pred_1"], 4),
                "誤差(p1)": round(result["target1_error"], 4),
                "参考予測(p2)": round(result["pred_2_reference"], 4),
                "Feasibility": level,
                "knn_distance": round(knn_value, 3),
                "Optunaトライアル数": n_trials,
                "固定条件": str(fixed_conditions) if fixed_conditions else "なし",
            })
            st.success(f"結果を compare_df に追加しました（実行No.{st.session_state.run_count}）。「📊 結果比較」タブで確認できます。")

        else:
            st.info("左の設定パネルで目標値を設定し、「逆解析を実行」を押してください。")
            st.subheader("学習データの分布（先頭10件）")
            st.dataframe(X_base[best_features].head(10), use_container_width=True)


# ===========================
# タブ2: 順方向予測モード
# ===========================
with tab_fwd:
    st.subheader("製造条件を入力して物性を予測")
    st.caption("各パラメータを設定すると、target_property_1 / 2 の予測値がリアルタイムで表示されます。")

    # 特徴量を3列に分けてスライダー表示
    input_vals = {}
    cols_per_row = 3
    feature_chunks = [best_features[i:i+cols_per_row] for i in range(0, len(best_features), cols_per_row)]

    with st.expander("製造条件スライダー（全特徴量）", expanded=True):
        for chunk in feature_chunks:
            cols = st.columns(cols_per_row)
            for col, feat in zip(cols, chunk):
                lo = float(X_base[feat].quantile(0.05))
                hi = float(X_base[feat].quantile(0.95))
                median = float(X_base[feat].median())
                with col:
                    input_vals[feat] = st.slider(
                        feat,
                        min_value=lo,
                        max_value=hi,
                        value=median,
                        key=f"fwd_{feat}",
                        format="%.3f",
                    )

    # 予測実行（リアルタイム）
    X_input = pd.DataFrame([input_vals])[best_features]
    pred1 = model_1.predict(X_input)[0]
    pred2 = model_2.predict(X_input)[0]

    # Feasibility
    range_check_fwd = check_range(input_vals, range_table)
    knn_val_fwd = get_knn_value(input_vals, best_features, knn_reference)
    level_fwd, message_fwd = judge_extrapolation_level(range_check_fwd, knn_val_fwd, knn_reference)

    st.subheader("予測結果")
    r1, r2 = st.columns(2)
    r1.metric("target_property_1 予測値", f"{pred1:.4f}")
    r2.metric("target_property_2 予測値", f"{pred2:.4f}")

    st.subheader("Feasibility")
    level_color_fwd = {"SAFE": "success", "CAUTION": "warning", "WARNING": "error"}
    getattr(st, level_color_fwd[level_fwd])(f"**{level_fwd}**: {message_fwd}")

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("extrapolation_level", level_fwd)
    fc2.metric("q05q95_out_count", int(range_check_fwd["q05q95_out"].sum()))
    fc3.metric("knn_distance", f"{knn_val_fwd:.3f}")

    with st.expander("範囲チェック詳細"):
        st.dataframe(range_check_fwd, use_container_width=True)


# ===========================
# タブ3: 結果比較 (compare_df)
# ===========================
with tab_compare:
    st.subheader("逆解析結果の比較")

    if not st.session_state.compare_results:
        st.info("まだ逆解析を実行していません。「逆解析モード」タブで実行すると、ここに結果が蓄積されます。")
    else:
        compare_df = pd.DataFrame(st.session_state.compare_results)

        # Feasibilityに応じた色付け
        def color_feasibility(val):
            colors = {"SAFE": "background-color: #d4edda", "CAUTION": "background-color: #fff3cd", "WARNING": "background-color: #f8d7da"}
            return colors.get(val, "")

        styled = compare_df.style.applymap(color_feasibility, subset=["Feasibility"])
        st.dataframe(styled, use_container_width=True)

        # サマリー統計
        st.subheader("サマリー統計")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("実行回数", len(compare_df))
        sc2.metric("平均誤差(p1)", f"{compare_df['誤差(p1)'].mean():.4f}")
        sc3.metric("SAFE率", f"{(compare_df['Feasibility']=='SAFE').mean()*100:.0f}%")

        # CSVダウンロード
        csv = compare_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "結果をCSVでダウンロード",
            data=csv,
            file_name="compare_results.csv",
            mime="text/csv",
        )

        if st.button("結果をリセット", type="secondary"):
            st.session_state.compare_results = []
            st.session_state.run_count = 0
            st.rerun()
