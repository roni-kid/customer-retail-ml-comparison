# Customer Retail ML Model Comparison
# Models: Logistic Regression, Decision Tree, KNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ── 1. Load Data ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('customer_retail csv file.csv', encoding='latin-1')
print(f"Dataset shape: {df.shape}")

# ── 2. Clean Data ─────────────────────────────────────────────
print("\nMissing values:")
print(df.isnull().sum())
df.dropna(inplace=True)
print(f"After cleaning: {df.shape}")

# ── 3. Create Target Label ────────────────────────────────────
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
df['HighValue'] = (df['TotalSpend'] > df['TotalSpend'].median()).astype(int)

# ── 4. Encode Country ─────────────────────────────────────────
le = LabelEncoder()
df['Country_Encoded'] = le.fit_transform(df['Country'])

# ── 5. Visualize Customer Distribution ───────────────────────
country_counts = df['Country'].value_counts().head(10)
plt.figure(figsize=(12, 5))
country_counts.plot(kind='bar', color='steelblue')
plt.title('Top 10 Countries by Number of Transactions')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('customer_distribution.png')
plt.show()
print("Customer distribution chart saved.")

# ── 6. Split Data ─────────────────────────────────────────────
X = df[['Quantity', 'UnitPrice', 'Country_Encoded']]
y = df['HighValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining size: {X_train.shape}")
print(f"Testing size:  {X_test.shape}")

# ── 7. Train Models ───────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Training KNN (this may take a moment)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# ── 8. Evaluate Models ────────────────────────────────────────
lr_acc  = accuracy_score(y_test, lr_pred)
dt_acc  = accuracy_score(y_test, dt_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("\n── Model Accuracies ──")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Decision Tree:       {dt_acc:.4f}")
print(f"KNN:                 {knn_acc:.4f}")

# ── 9. Confusion Matrices ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, pred) in zip(axes, [
    ('Logistic Regression', lr_pred),
    ('Decision Tree',       dt_pred),
    ('KNN',                 knn_pred)
]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
print("Confusion matrices saved.")

# ── 10. Accuracy Comparison Chart ────────────────────────────
models     = ['Logistic Regression', 'Decision Tree', 'KNN']
accuracies = [lr_acc, dt_acc, knn_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['steelblue', 'seagreen', 'tomato'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.05)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()
print("Accuracy comparison chart saved.")

print("\nDone. All models trained, evaluated, and charts saved.")