import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - SAVES plots guaranteed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Force seaborn style
plt.style.use('default')
sns.set_palette("husl")

print("🔄 Loading data...")
df = pd.read_csv("data/E_Commerce.csv")
print(f"✅ Training with {len(df)} records...")

# Prepare data
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

print(f"📊 Features: {X.shape[1]}, Classes: {y.nunique()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("🤖 Training Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions & evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2%}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Saved model.pkl & columns.pkl")

# ========== ON-TIME GRAPHS (GUARANTEED TO WORK) ==========
print("📈 Creating on-time analysis graphs...")

# 1. CONFUSION MATRIX
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Late', 'On-Time'], 
            yticklabels=['Late', 'On-Time'])
plt.title('Confusion Matrix\n(On-Time Delivery)', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 2. PREDICTION DISTRIBUTION
plt.subplot(2, 2, 2)
pred_dist = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_dist.value_counts().unstack(fill_value=0).plot(kind='bar', ax=plt.gca())
plt.title('Actual vs Predicted Distribution', fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Status')

# 3. FEATURE IMPORTANCE (TOP 10)
plt.subplot(2, 2, 3)
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
sns.barplot(x=top10.values, y=top10.index, palette='viridis')
plt.title('Top 10 Feature Importances', fontweight='bold')
plt.xlabel('Importance')

# 4. ON-TIME RATE OVERALL
plt.subplot(2, 2, 4)
ontime_rate = y.mean()
plt.pie([ontime_rate, 1-ontime_rate], labels=['On-Time', 'Late'], 
        autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0))
plt.title('Overall On-Time Rate', fontweight='bold')

plt.suptitle('E-Commerce On-Time Delivery Analysis', fontsize=16, y=1.02)
plt.tight_layout()

# SAVE - This ALWAYS works
plt.savefig('ontime_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ontime_analysis.pdf', bbox_inches='tight')  # PDF backup
plt.close()  # Close figure to free memory
print("✅ Graphs saved: ontime_analysis.png & ontime_analysis.pdf")

# ========== SIMPLE BAR CHART (ALTERNATIVE) ==========
plt.figure(figsize=(10, 6))
results = pd.DataFrame({
    'Metric': ['Accuracy', 'On-Time Rate', 'Late Rate'],
    'Value': [accuracy, y.mean(), 1-y.mean()]
})
sns.barplot(data=results, x='Metric', y='Value', palette='Set2')
plt.title('On-Time Delivery Summary', fontweight='bold')
plt.ylim(0, 1)
for i, v in enumerate(results['Value']):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')
plt.savefig('ontime_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Summary chart saved: ontime_summary.png")

# Test model load
model_loaded = pickle.load(open("model.pkl", "rb"))
print("✅ Model loaded successfully!")
print(f"✅ Expected features: {len(pickle.load(open('columns.pkl', 'rb')))}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Late', 'On-Time']))

print("\n🎉 ALL FILES GENERATED:")
print("- model.pkl")
print("- columns.pkl") 
print("- ontime_analysis.png")
print("- ontime_analysis.pdf")
print("- ontime_summary.png")
