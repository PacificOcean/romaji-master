"""
製造条件 逆解析支援ツール v2
- タブ切り替え: 逆解析モード / 順方向予測モード / 結果比較
- compare_df: 逆解析結果の蓄積・比較
- SHAP: 予測の要因分解（waterfall plot + 全体特徴量重要度）
"""
import sys
import subprocess
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# このファイルの場所を基準にパスを解決（ローカル・Streamlit Cloud 両対応）
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

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


# --- モデル・SHAP読み込み（キャッシュ）---
def _ensure_models():
    """モデルが存在しない場合、データ生成→学習を自動実行する（初回デプロイ時）"""
    model_path = BASE_DIR / "models" / "best_model_1_stable_29feat.pkl"
    if not model_path.exists():
        with st.spinner("初回起動: ダミーデータ生成・モデル学習中（1〜2分）..."):
            (BASE_DIR / "models").mkdir(exist_ok=True)
            (BASE_DIR / "data").mkdir(exist_ok=True)
            subprocess.run(
                [sys.executable, str(BASE_DIR / "generate_data.py")],
                cwd=BASE_DIR, check=True
            )
            subprocess.run(
                [sys.executable, str(BASE_DIR / "train_model.py")],
                cwd=BASE_DIR, check=True
            )


@st.cache_resource
def load_models():
    _ensure_models()
    model_1 = joblib.load(BASE_DIR / "models" / "best_model_1_stable_29feat.pkl")
    model_2 = joblib.load(BASE_DIR / "models" / "best_model_2_stable_29feat.pkl")
    features = joblib.load(BASE_DIR / "models" / "best_features_stable_29feat.pkl")
    X_base   = joblib.load(BASE_DIR / "models" / "X_base.pkl")
    return model_1, model_2, features, X_base


@st.cache_resource
def load_shap_explainers(_model_1, _model_2):
    explainer_1 = shap.TreeExplainer(_model_1)
    explainer_2 = shap.TreeExplainer(_model_2)
    return explainer_1, explainer_2


@st.cache_data
def get_global_shap_values(_explainer_1, _explainer_2, _X_base, _features):
    """全学習データのSHAP値（グローバル特徴量重要度用）"""
    X_sample = _X_base[_features].sample(min(200, len(_X_base)), random_state=42)
    sv1 = _explainer_1(X_sample)
    sv2 = _explainer_2(X_sample)
    return sv1, sv2


def shap_waterfall_fig(explainer, X_input_df, title="", max_display=12):
    """SHAP waterfallプロットをmatplotlib figureとして返す"""
    sv = explainer(X_input_df)
    plt.close("all")
    shap.plots.waterfall(sv[0], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def shap_bar_fig(shap_values, title="", max_display=15):
    """SHAP barプロット（グローバル重要度）をmatplotlib figureとして返す"""
    plt.close("all")
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


try:
    model_1, model_2, best_features, X_base = load_models()
    explainer_1, explainer_2 = load_shap_explainers(model_1, model_2)
    shap_base_1, shap_base_2 = get_global_shap_values(
        explainer_1, explainer_2, X_base, best_features
    )
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
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

            # ---- SHAP ----
            st.subheader("🔍 SHAP による予測の説明")
            st.caption("各製造条件が予測値をどの方向にどれだけ動かしているかを可視化します（赤→予測値を上げる / 青→下げる）")

            X_best = pd.DataFrame([best_params])[best_features]
            shap_col1, shap_col2 = st.columns(2)
            with shap_col1:
                st.markdown("**target_property_1 の要因分解**")
                fig1 = shap_waterfall_fig(explainer_1, X_best, max_display=12)
                st.pyplot(fig1)
                plt.close("all")
            with shap_col2:
                st.markdown("**target_property_2 の要因分解（参考）**")
                fig2 = shap_waterfall_fig(explainer_2, X_best, max_display=12)
                st.pyplot(fig2)
                plt.close("all")

            with st.expander("📊 全体の特徴量重要度（学習データ全体のSHAP平均）"):
                gi_col1, gi_col2 = st.columns(2)
                with gi_col1:
                    st.markdown("**target_property_1**")
                    fig_g1 = shap_bar_fig(shap_base_1, max_display=15)
                    st.pyplot(fig_g1)
                    plt.close("all")
                with gi_col2:
                    st.markdown("**target_property_2**")
                    fig_g2 = shap_bar_fig(shap_base_2, max_display=15)
                    st.pyplot(fig_g2)
                    plt.close("all")

            # 近傍実績
            st.subheader("近傍実績（過去の類似条件）")
            nearest_df = find_nearest_records(best_params, best_features, X_base, n_neighbors=5)
            st.dataframe(nearest_df.reset_index(drop=True), use_container_width=True)

            with st.expander("提案条件（全特徴量）"):
                proposed_df = pd.DataFrame([best_params]).T.rename(columns={0: "提案値"})
                st.dataframe(proposed_df.round(4), use_container_width=True)

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
            st.success(f"結果を compare_df に追加しました（実行No.{st.session_state.run_count}）")

        else:
            st.info("左の設定パネルで目標値を設定し、「逆解析を実行」を押してください。")
            st.subheader("学習データの分布（先頭10件）")
            st.dataframe(X_base[best_features].head(10), use_container_width=True)


# ===========================
# タブ2: 順方向予測モード
# ===========================
with tab_fwd:
    st.subheader("製造条件を入力して物性を予測")
    st.caption("スライダーを動かすと予測値とSHAP解説がリアルタイムで更新されます。")

    input_vals = {}
    cols_per_row = 3
    feature_chunks = [best_features[i:i+cols_per_row] for i in range(0, len(best_features), cols_per_row)]

    with st.expander("製造条件スライダー（全特徴量）", expanded=True):
        for chunk in feature_chunks:
            cols = st.columns(cols_per_row)
            for col, feat in zip(cols, chunk):
                lo = float(X_base[feat].quantile(0.05))
                hi = float(X_base[feat].quantile(0.95))
                with col:
                    input_vals[feat] = st.slider(
                        feat, lo, hi, float(X_base[feat].median()),
                        key=f"fwd_{feat}", format="%.3f",
                    )

    X_input = pd.DataFrame([input_vals])[best_features]
    pred1 = model_1.predict(X_input)[0]
    pred2 = model_2.predict(X_input)[0]

    range_check_fwd = check_range(input_vals, range_table)
    knn_val_fwd = get_knn_value(input_vals, best_features, knn_reference)
    level_fwd, message_fwd = judge_extrapolation_level(range_check_fwd, knn_val_fwd, knn_reference)

    st.subheader("予測結果")
    r1, r2 = st.columns(2)
    r1.metric("target_property_1 予測値", f"{pred1:.4f}")
    r2.metric("target_property_2 予測値", f"{pred2:.4f}")

    level_color_fwd = {"SAFE": "success", "CAUTION": "warning", "WARNING": "error"}
    getattr(st, level_color_fwd[level_fwd])(f"**Feasibility: {level_fwd}** — {message_fwd}")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("extrapolation_level", level_fwd)
    fc2.metric("q05q95_out_count", int(range_check_fwd["q05q95_out"].sum()))
    fc3.metric("knn_distance", f"{knn_val_fwd:.3f}")

    # ---- SHAP ----
    st.subheader("🔍 SHAP による予測の説明")
    st.caption("現在のスライダー値がそれぞれの物性予測にどう影響しているかを表示します。")

    fwd_col1, fwd_col2 = st.columns(2)
    with fwd_col1:
        st.markdown("**target_property_1 の要因分解**")
        fig_f1 = shap_waterfall_fig(explainer_1, X_input, max_display=12)
        st.pyplot(fig_f1)
        plt.close("all")
    with fwd_col2:
        st.markdown("**target_property_2 の要因分解**")
        fig_f2 = shap_waterfall_fig(explainer_2, X_input, max_display=12)
        st.pyplot(fig_f2)
        plt.close("all")

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

        def color_feasibility(val):
            colors = {
                "SAFE": "background-color: #d4edda",
                "CAUTION": "background-color: #fff3cd",
                "WARNING": "background-color: #f8d7da",
            }
            return colors.get(val, "")

        styled = compare_df.style.map(color_feasibility, subset=["Feasibility"])
        st.dataframe(styled, use_container_width=True)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("実行回数", len(compare_df))
        sc2.metric("平均誤差(p1)", f"{compare_df['誤差(p1)'].mean():.4f}")
        sc3.metric("SAFE率", f"{(compare_df['Feasibility']=='SAFE').mean()*100:.0f}%")

        csv = compare_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("結果をCSVでダウンロード", data=csv,
                           file_name="compare_results.csv", mime="text/csv")

        if st.button("結果をリセット", type="secondary"):
            st.session_state.compare_results = []
            st.session_state.run_count = 0
            st.rerun()
