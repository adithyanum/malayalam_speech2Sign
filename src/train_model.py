import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
import joblib

# 1. Load Dataset
# The dataset contains 1,248 features (MFCC + Delta + Delta-Delta)
df = pd.read_csv('data/features/malayalam_features.csv')

# 2. Feature-Target Separation
X = df.drop('label', axis=1)
y = df['label']

# 3. Label Encoding
# Converting Malayalam strings into categorical numerical indices
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Stratified Train-Test Split
# Stratification ensures an equal distribution of each word in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Feature Scaling
# Standardizing features to Zero Mean and Unit Variance (Critical for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Dimensionality Reduction (Feature Selection)
# Using ANOVA F-test (f_classif) to select the Top 300 most discriminative features.
# This mitigates the 'Curse of Dimensionality' and prevents the model from overfitting to noise.
selector = SelectKBest(score_func=f_classif, k=300)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"Optimized Feature Space: Reduced from {X_train_scaled.shape[1]} to {X_train_selected.shape[1]} features.")

# 7. Hyperparameter Optimization+
# Searching for the optimal 'C' (Penalty) and 'Gamma' (Kernel Coefficient).
# probability=True enables Platt Scaling for confidence-based inference.
param_grid = {
    'C': [0.1, 1, 5, 10], 
    'gamma': ['scale', 'auto', 0.001],
    'kernel': ['rbf']
}

print("Executing Grid Search Cross-Validation...")
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train_selected, y_train)

# 8. Evaluation
best_model = grid.best_estimator_
predictions = best_model.predict(X_test_selected)

print("\n--- Model Optimization Results ---")
print(f"Best Parameters: {grid.best_params_}")

print("\n--- Performance Metrics ---")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# 9. Persistence (Saving the Pipeline)
# We save the Selector so the Inference script knows which features to extract live.
joblib.dump(best_model, 'models/malayalam_svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(selector, 'models/selector.pkl') 

print("\n✅ Training Pipeline Complete. Assets saved to models/ folder!")
