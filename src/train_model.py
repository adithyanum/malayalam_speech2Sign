import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the features
df = pd.read_csv('data/features/malayalam_features.csv')

# 2. Separate Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']

# 3. Encode the Malayalam text into numbers
# "സന്തോഷം" -> 0, "വീട്" -> 1, etc.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

print("Searching for the best C and Gamma...")
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train_scaled, y_train)

# 7. Evaluate
best_model = grid.best_estimator_
predictions = best_model.predict(X_test_scaled)

print("\n--- Best Parameters ---")
print(grid.best_params_)

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# 8. Save the "Brain" for your friend (Mediapipe lead)
joblib.dump(best_model, 'models/malayalam_svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("\n✅ Model and Scalers saved to models/ folder!")