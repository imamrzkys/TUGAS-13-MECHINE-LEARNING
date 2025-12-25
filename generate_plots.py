"""
Script untuk generate dan menyimpan semua visualisasi dari TA13.ipynb
Menjalankan script ini akan membuat semua plot dan menyimpannya ke folder static/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, precision_recall_curve, auc, average_precision_score
)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import joblib

# Setup folder untuk menyimpan plot
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plot akan disimpan ke: {PLOTS_DIR}")

# Path dataset
DATA_PATH = os.path.join(BASE_DIR, "Penipuan Transaksi Digital.csv")
MODEL_PATH = os.path.join(BASE_DIR, "svm_fraud_pipeline.pkl")

print("\n" + "="*60)
print("GENERATE VISUALISASI - FRAUD DETECTION SVM")
print("="*60 + "\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("1. Loading dataset...")
usecols = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud"
]

dtypes = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8"
}

df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes, low_memory=True, memory_map=True)
print(f"   Dataset shape: {df.shape}")

# ============================================================================
# 2. PLOT 1: DISTRIBUSI KELAS
# ============================================================================
print("\n2. Generating class_distribution.png...")
counts = df["isFraud"].value_counts()
pct = df["isFraud"].value_counts(normalize=True) * 100

plt.figure(figsize=(6, 4))
sns.barplot(x=counts.index.astype(str), y=counts.values)
plt.title("Distribusi Kelas isFraud (0=Normal, 1=Fraud)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: class_distribution.png")

# ============================================================================
# 3. PLOT 2: DISTRIBUSI AMOUNT (KDE)
# ============================================================================
print("\n3. Generating amount_kde.png...")
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df, x="amount", hue="isFraud", common_norm=False)
plt.title("Distribusi Amount (Data Asli - Imbalanced)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "amount_kde.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: amount_kde.png")

# ============================================================================
# 4. PLOT 3: CORRELATION HEATMAP
# ============================================================================
print("\n4. Generating corr_heatmap.png...")
num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(9, 6))
sns.heatmap(num_df.corr(), annot=False)
plt.title("Correlation Heatmap (Numerik)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "corr_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: corr_heatmap.png")

# ============================================================================
# 5. BALANCE DATA & SPLIT
# ============================================================================
print("\n5. Balancing dataset...")
fraud = df[df["isFraud"] == 1]
normal = df[df["isFraud"] == 0]
n_fraud = len(fraud)
normal_down = normal.sample(n=n_fraud, random_state=42)
df_bal = pd.concat([fraud, normal_down]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Balanced shape: {df_bal.shape}")

X = df_bal.drop(columns=["isFraud"])
y = df_bal["isFraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = ["type"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ============================================================================
# 6. LOAD ATAU TRAIN MODEL
# ============================================================================
print("\n6. Loading/Training model...")
if os.path.exists(MODEL_PATH):
    print("   Loading existing model...")
    best_linear = joblib.load(MODEL_PATH)
    y_pred_best = best_linear.predict(X_test)
    print("   ✓ Model loaded successfully")
else:
    print("   Model tidak ditemukan, training model baru...")
    svm_linear = Pipeline(steps=[
        ("prep", preprocess),
        ("clf", LinearSVC(max_iter=10000))
    ])
    
    param_grid = {"clf__C": [0.1, 1, 10]}
    grid = GridSearchCV(
        svm_linear,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    best_linear = grid.best_estimator_
    y_pred_best = best_linear.predict(X_test)
    
    joblib.dump(best_linear, MODEL_PATH)
    print(f"   ✓ Model trained and saved to {MODEL_PATH}")

# ============================================================================
# 7. PLOT 4 & 5: CONFUSION MATRIX
# ============================================================================
print("\n7. Generating confusion_matrix_count.png dan confusion_matrix_percent.png...")
cm = confusion_matrix(y_test, y_pred_best)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Count)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_count.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: confusion_matrix_count.png")

plt.figure(figsize=(6, 5))
sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues")
plt.title("Confusion Matrix (Row %)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_percent.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: confusion_matrix_percent.png")

# ============================================================================
# 8. PLOT 6 & 7: ROC & PR CURVE
# ============================================================================
print("\n8. Generating roc_curve.png dan pr_curve.png...")
scores = best_linear.decision_function(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - LinearSVC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: roc_curve.png")

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, scores)
ap = average_precision_score(y_test, scores)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"AP = {ap:.4f}")
plt.title("Precision-Recall Curve - LinearSVC")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: pr_curve.png")

# ============================================================================
# 9. PLOT 8 & 9: PCA 2D & 3D
# ============================================================================
print("\n9. Generating pca_2d.png dan pca_3d.png...")
# Sampling untuk performa
N = min(8000, len(X_test))
idx = np.random.RandomState(42).choice(len(X_test), size=N, replace=False)
X_s = X_test.iloc[idx]
y_s = y_test.iloc[idx]

X_s_trans = preprocess.fit_transform(X_s)
X_s_dense = X_s_trans.toarray() if hasattr(X_s_trans, "toarray") else X_s_trans

# PCA 2D
pca2 = PCA(n_components=2, random_state=42)
X2 = pca2.fit_transform(X_s_dense)
score_s = best_linear.decision_function(X_s)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X2[:, 0], X2[:, 1], c=score_s, s=18, alpha=0.75, cmap='viridis')
plt.colorbar(scatter, label="Decision Score")
plt.title("PCA 2D + Decision Score (Gradasi Kepercayaan Model)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_2d.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: pca_2d.png")

# PCA 3D
pca3 = PCA(n_components=3, random_state=42)
X3 = pca3.fit_transform(X_s_dense)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=score_s, s=18, alpha=0.75)
fig.colorbar(p, ax=ax, shrink=0.6, label="Decision Score")
ax.set_title("PCA 3D + Decision Score (Gradasi Kepercayaan Model)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_3d.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: pca_3d.png")

# ============================================================================
# 10. PLOT 10: FEATURE IMPORTANCE
# ============================================================================
print("\n10. Generating feature_importance.png...")
prep = best_linear.named_steps["prep"]
clf = best_linear.named_steps["clf"]

num_features = num_cols
cat_features = list(prep.named_transformers_["cat"].get_feature_names_out(["type"]))
feature_names = num_features + cat_features

weights = clf.coef_.ravel()
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "weight": weights,
    "abs_weight": np.abs(weights)
}).sort_values("abs_weight", ascending=False)

top = feat_imp.head(15).sort_values("abs_weight")
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["weight"])
plt.title("Top 15 Feature Weights (Linear SVM)")
plt.xlabel("Weight (+ mendukung FRAUD, - mendukung NORMAL)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: feature_importance.png")

# ============================================================================
# 11. PLOT 11 & 12: DECISION SCORE DISTRIBUTION
# ============================================================================
print("\n11. Generating decision_score_kde.png dan decision_score_histogram.png...")
plot_df = pd.DataFrame({"score": scores, "isFraud": y_test.values})

plt.figure(figsize=(9, 4))
sns.kdeplot(data=plot_df, x="score", hue="isFraud", common_norm=False)
plt.title("Decision Score Distribution (Normal vs Fraud)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "decision_score_kde.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: decision_score_kde.png")

th = np.median(scores)
plt.figure(figsize=(9, 4))
sns.histplot(data=plot_df, x="score", hue="isFraud", bins=60, element="step", stat="density")
plt.axvline(th, linestyle="--", label=f"Threshold = {th:.3f}")
plt.title(f"Score Histogram + Threshold (th={th:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "decision_score_histogram.png"), dpi=200, bbox_inches="tight")
plt.close()
print("   ✓ Saved: decision_score_histogram.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SELESAI! Semua visualisasi telah disimpan ke:")
print(f"  {PLOTS_DIR}")
print("="*60)
print("\nDaftar file yang dihasilkan:")
plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
for i, f in enumerate(plot_files, 1):
    print(f"  {i:2d}. {f}")

