"""
用DBAASP真实实验数据训练ML预测模型（sklearn版本，无需xgboost）：
1. 溶血预测器（GradientBoosting回归log HC50 + 分类HC50≤50）
2. 活性预测器（GradientBoosting分类 MIC≤16为active）
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from descriptor_calculator import compute_descriptors_for_sequence, DESCRIPTOR_NAMES

EXCLUDE_COLS = {"gram_negative_active", "label", "sequence", "dbaasp_id", "name",
                "has_gram_negative_activity", "hemo_activity_value", "best_gn_mic",
                "hemolysis_risk", "toxicity_risk"}


def prepare_features(df, target_col):
    feature_rows = []
    targets = []
    for _, row in df.iterrows():
        seq = str(row["sequence"])
        try:
            desc = compute_descriptors_for_sequence(seq)
            features = {k: v for k, v in desc.items() if k not in EXCLUDE_COLS}
            feature_rows.append(features)
            targets.append(row[target_col])
        except Exception:
            pass
    if not feature_rows:
        return pd.DataFrame(), np.array([])
    X = pd.DataFrame(feature_rows)
    y = np.array(targets)
    feature_cols = [n for n in DESCRIPTOR_NAMES if n not in EXCLUDE_COLS]
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0)
    return X, y


def train_hemolysis_models():
    print("=" * 60)
    print("训练溶血预测模型（DBAASP真实HC50数据, sklearn GBDT）")
    print("=" * 60)

    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")
    df = pd.read_csv(raw_path)

    df["hemo_activity_value"] = pd.to_numeric(df["hemo_activity_value"], errors="coerce")
    hemo_df = df[df["has_hemolytic_data"] == True].copy()
    hemo_df = hemo_df.dropna(subset=["hemo_activity_value", "sequence"])
    hemo_df = hemo_df[(hemo_df["length"] >= 4) & (hemo_df["length"] <= 50)]
    hemo_df = hemo_df[hemo_df["hemo_activity_value"] < 1e5]
    hemo_df = hemo_df.reset_index(drop=True)

    print(f"有真实溶血数据的肽: {len(hemo_df)} 条")
    print(f"HC50范围: {hemo_df['hemo_activity_value'].min():.2f} - {hemo_df['hemo_activity_value'].max():.2f} µM")

    print("计算20维描述符特征...")
    X, y_hc50 = prepare_features(hemo_df, "hemo_activity_value")
    if len(X) == 0:
        print("特征提取失败"); return None, None

    y_log = np.log10(np.clip(y_hc50, 0.01, 10000))

    y_cls = np.full(len(y_hc50), -1)
    y_cls[y_hc50 <= 50] = 1
    y_cls[y_hc50 >= 200] = 0
    mask_cls = y_cls >= 0
    print(f"分类样本: 高风险{(y_cls==1).sum()} / 低风险{(y_cls==0).sum()}")

    # ---------------- 回归模型 ----------------
    print("\n--- 回归模型（预测log10(HC50)） ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    reg.fit(X_train_s, y_train)

    y_pred = reg.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE = {mae:.4f} (log10尺度)")
    print(f"  R²  = {r2:.4f}")
    print(f"  MAE(原始尺度) = {np.mean(np.abs(10**y_test - 10**y_pred)):.1f} µM")

    cv_scores = cross_val_score(
        GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        scaler.transform(X), y_log, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"  5-fold CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    scaler_reg_full = StandardScaler()
    X_all_s = scaler_reg_full.fit_transform(X)
    reg_full = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    reg_full.fit(X_all_s, y_log)

    # ---------------- 分类模型 ----------------
    print("\n--- 分类模型（HC50≤50为高风险） ---")
    X_cls = X[mask_cls]
    y_cls_valid = y_cls[mask_cls]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls_valid, test_size=0.2, random_state=42, stratify=y_cls_valid
    )
    scaler_c = StandardScaler()
    X_train_cs = scaler_c.fit_transform(X_train_c)
    X_test_cs = scaler_c.transform(X_test_c)

    cls = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    cls.fit(X_train_cs, y_train_c)

    y_prob = cls.predict_proba(X_test_cs)[:, 1]
    auc = roc_auc_score(y_test_c, y_prob)
    acc = accuracy_score(y_test_c, (y_prob >= 0.5).astype(int))
    print(f"  AUC-ROC = {auc:.4f}")
    print(f"  Accuracy = {acc:.4f}")

    # 与规则公式对比
    from descriptor_calculator import predict_hemolysis_risk as rule_predict
    rule_preds = []
    for _, row in hemo_df[mask_cls].iterrows():
        try:
            rule_preds.append(rule_predict(str(row["sequence"])))
        except Exception:
            rule_preds.append(0.5)
    rule_preds = np.array(rule_preds)
    rule_auc = roc_auc_score(y_cls_valid, rule_preds)
    print(f"\n  规则公式AUC对比:")
    print(f"    规则公式: AUC = {rule_auc:.4f}")
    print(f"    ML分类器: AUC = {auc:.4f} (提升 {(auc-rule_auc)*100:.1f}%)")

    # 5折CV
    cv_aucs = cross_val_score(
        GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        scaler_c.transform(X_cls), y_cls_valid, cv=5, scoring="roc_auc"
    )
    print(f"  5-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    scaler_cls_full = StandardScaler()
    X_cls_all_s = scaler_cls_full.fit_transform(X_cls)
    cls_full = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    cls_full.fit(X_cls_all_s, y_cls_valid)

    # 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "hemolysis_regressor.pkl"), "wb") as f:
        pickle.dump({"model": reg_full, "scaler": scaler_reg_full, "type": "regression"}, f)
    with open(os.path.join(MODEL_DIR, "hemolysis_classifier.pkl"), "wb") as f:
        pickle.dump({"model": cls_full, "scaler": scaler_cls_full, "type": "classification"}, f)
    print(f"\n模型已保存到 {MODEL_DIR}/")

    return reg_full, cls_full, scaler_reg_full, scaler_cls_full


def train_activity_model():
    print("\n" + "=" * 60)
    print("训练活性预测模型（DBAASP真实MIC数据, sklearn GBDT）")
    print("=" * 60)

    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")
    df = pd.read_csv(raw_path)

    df["best_gn_mic"] = pd.to_numeric(df["best_gn_mic"], errors="coerce")
    mic_df = df[df["best_gn_mic"].notna()].copy()
    mic_df = mic_df[(mic_df["length"] >= 4) & (mic_df["length"] <= 50)]

    y_label = np.full(len(mic_df), -1)
    y_label[mic_df["best_gn_mic"] <= 16] = 1
    y_label[mic_df["best_gn_mic"] >= 64] = 0
    mic_df["activity_label"] = y_label
    mic_df = mic_df[mic_df["activity_label"] >= 0].reset_index(drop=True)

    print(f"有MIC数据的肽（筛选后）: {len(mic_df)} 条")
    print(f"  Active (MIC≤16): {(mic_df['activity_label']==1).sum()}")
    print(f"  Inactive (MIC≥64): {(mic_df['activity_label']==0).sum()}")

    print("计算20维描述符特征...")
    X, y = prepare_features(mic_df, "activity_label")
    if len(X) == 0:
        print("特征提取失败"); return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cls = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    cls.fit(X_train_s, y_train)

    y_prob = cls.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))
    print(f"  AUC-ROC = {auc:.4f}")
    print(f"  Accuracy = {acc:.4f}")

    cv_aucs = cross_val_score(
        GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        scaler.transform(X), y, cv=5, scoring="roc_auc"
    )
    print(f"  5-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X)
    cls_full = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    cls_full.fit(X_all_s, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "activity_classifier.pkl"), "wb") as f:
        pickle.dump({"model": cls_full, "scaler": scaler_full, "type": "classification"}, f)
    print(f"模型已保存: {os.path.join(MODEL_DIR, 'activity_classifier.pkl')}")
    return cls_full, scaler_full


def main():
    print("=" * 70)
    print("  ML预测模型训练（sklearn版本）")
    print("=" * 70)

    hemo_reg, hemo_cls, hemo_sr, hemo_sc = train_hemolysis_models()
    act_cls, act_sc = train_activity_model()

    print("\n" + "=" * 70)
    print("  训练完成！")
    print("=" * 70)
    print("  下一步：修改descriptor_calculator.py集成ML模型推理")
    print("  然后重新生成候选肽、重跑验证脚本")


if __name__ == "__main__":
    main()
