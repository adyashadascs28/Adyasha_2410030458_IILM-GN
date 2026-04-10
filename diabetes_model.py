
#   DIABETES PREDICTION SYSTEM USING MACHINE LEARNING
#   Model Training, Evaluation & Results
#   Team: Adyasha Das, Khushi Rana, Laxmi Bhanti,
#         Anand Kumar Jha, Vanshika Singh
#   Guide: Ms. Pushpa Singh
#   IILM University, Greater Noida
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DIABETES PREDICTION SYSTEM USING MACHINE LEARNING")
print("   IILM University, Greater Noida")
print("="*55)

# ── 1. LOAD DATASET ──────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv('diabetes.csv')
print(f"    Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")
print("\n    First 5 rows:")
print(df.head())
print("\n    Class distribution:")
print(df['Outcome'].value_counts())

# ── 2. DATA PREPROCESSING ────────────────────────────────────
print("\n[2] Preprocessing data...")

# Replace 0s with NaN for medically impossible zero values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Fill missing values with median
for col in zero_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"    {col}: missing values filled with median = {median_val:.2f}")

# ── 3. SPLIT FEATURES AND TARGET ─────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. TRAIN MODELS ──────────────────────────────────────────
print("\n[3] Training models...")

models = {
    'SVM (RBF)':          SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec  = recall_score(y_test, y_pred) * 100
    f1   = f1_score(y_test, y_pred) * 100
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
                     'model': model, 'y_pred': y_pred}
    print(f"    {name:25s} | Acc: {acc:.1f}% | Prec: {prec:.1f}% | Rec: {rec:.1f}% | F1: {f1:.1f}%")

# ── 5. CROSS-VALIDATION ON SVM ───────────────────────────────
print("\n[4] Running 10-Fold Cross-Validation on SVM...")
svm_model = models['SVM (RBF)']
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=10, scoring='accuracy')
cv_scores_pct = cv_scores * 100
print(f"    Fold accuracies: {[f'{s:.1f}%' for s in cv_scores_pct]}")
print(f"    Mean CV Accuracy: {cv_scores_pct.mean():.1f}%")
print(f"    Std Dev: {cv_scores_pct.std():.2f}%")

# ── 6. SAVE THE BEST MODEL ───────────────────────────────────
print("\n[5] Saving trained SVM model and scaler...")
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("    svm_model.pkl and scaler.pkl saved successfully.")

# ── 7. GENERATE & SAVE PLOTS ─────────────────────────────────
print("\n[6] Generating result charts...")
plt.style.use('seaborn-v0_8-whitegrid')
fig_width = 14

# ── Plot 1: Model Performance Comparison Bar Chart ───────────
fig, ax = plt.subplots(figsize=(fig_width, 6))
model_names = list(results.keys())
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
x = np.arange(len(model_names))
width = 0.18
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('ML Model', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plot1_model_comparison.png")

# ── Plot 2: SVM Confusion Matrix ─────────────────────────────
cm = confusion_matrix(y_test, results['SVM (RBF)']['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            annot_kws={"size": 18, "weight": "bold"})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('SVM Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plot2_confusion_matrix.png")

# ── Plot 3: Feature Importance (Random Forest) ───────────────
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feat_names  = X.columns.tolist()
sorted_idx  = np.argsort(importances)
sorted(importances)
fig, ax = plt.subplots(figsize=(8, 6))
colors_bar = ['#1565C0' if imp > 0.12 else '#42A5F5' for imp in importances[sorted_idx]]
bars = ax.barh([feat_names[i] for i in sorted_idx], importances[sorted_idx], color=colors_bar)
for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=10)
ax.set_xlabel('Relative Importance Score', fontsize=12)
ax.set_title('Feature Importance for Diabetes Prediction', fontsize=13, fontweight='bold')
ax.set_xlim(0, max(importances) + 0.05)
plt.tight_layout()
plt.savefig('plot3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plot3_feature_importance.png")

# ── Plot 4: 10-Fold Cross-Validation Line Chart ───────────────
fig, ax = plt.subplots(figsize=(10, 5))
folds = [f'F{i+1}' for i in range(10)]
ax.plot(folds, cv_scores_pct, marker='o', color='#1565C0',
        linewidth=2, markersize=8, label='Fold Accuracy')
ax.axhline(cv_scores_pct.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean Accuracy: {cv_scores_pct.mean():.1f}%')
ax.fill_between(folds, cv_scores_pct.mean()-1.5, cv_scores_pct.mean()+1.5,
                alpha=0.15, color='red', label='\u00b11.5% Band')
ax.set_xlabel('Cross-Validation Fold', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('10-Fold Cross-Validation Accuracy (SVM)', fontsize=13, fontweight='bold')
ax.set_ylim(cv_scores_pct.min()-5, cv_scores_pct.max()+5)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot4_crossvalidation.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plot4_crossvalidation.png")

# ── 8. PRINT FINAL SUMMARY ───────────────────────────────────
print("\n" + "="*55)
print("   FINAL RESULTS SUMMARY")
print("="*55)
print(f"\n{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
print("-"*55)
for name, r in results.items():
    star = " ★" if name == 'SVM (RBF)' else ""
    print(f"{name+star:<27} {r['Accuracy']:>8.1f}% {r['Precision']:>9.1f}% {r['Recall']:>7.1f}% {r['F1-Score']:>9.1f}%")
print("-"*55)
print(f"\n  Best Model      : SVM with RBF Kernel")
print(f"  Test Accuracy   : {results['SVM (RBF)']['Accuracy']:.1f}%")
print(f"  Mean CV Accuracy: {cv_scores_pct.mean():.1f}%")
print(f"\n  Confusion Matrix (SVM):")
print(f"  {cm}")
print(f"\n  Detailed Classification Report (SVM):")
print(classification_report(y_test, results['SVM (RBF)']['y_pred'],
                             target_names=['Non-Diabetic', 'Diabetic']))
print("="*55)
print("\n✅ All done! Model saved. Now run: streamlit run app.py")
print("="*55)