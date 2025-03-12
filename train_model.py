import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_excel("C:/Internshala Project/data/dummy_npi_data.xlsx")

# Data Cleaning
data.dropna(inplace=True)

# Feature Engineering
data['Login Hour'] = pd.to_datetime(data['Login Time']).dt.hour
data['Logout Hour'] = pd.to_datetime(data['Logout Time']).dt.hour
data['Session Duration'] = data['Logout Hour'] - data['Login Hour']
data['Is Weekend'] = pd.to_datetime(data['Login Time']).dt.weekday >= 5

def assign_active_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

data['Active Period'] = data['Login Hour'].apply(assign_active_period)

# Encode categorical data
le_region = LabelEncoder()
le_speciality = LabelEncoder()
le_active_period = LabelEncoder()

data['Region'] = le_region.fit_transform(data['Region'])
data['Speciality'] = le_speciality.fit_transform(data['Speciality'])
data['Active Period'] = le_active_period.fit_transform(data['Active Period'])

# Bin 'Usage Time (mins)' to reduce its dominance
data['Usage Time (bins)'] = pd.qcut(data['Usage Time (mins)'], q=3, labels=[0, 1, 2])

# Define features and target
X = data[['Login Hour', 'Logout Hour', 'Session Duration', 'Is Weekend', 'Active Period', 'Usage Time (bins)', 'Region', 'Speciality']]
y = (data['Usage Time (mins)'] > data['Usage Time (mins)'].median()).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training with XGBoost and Hyperparameter Tuning
model = XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_model = grid_search.best_estimator_

# Evaluate Model
y_pred = best_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
importances = best_model.feature_importances_
feature_names = X.columns
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# Save model and scaler
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
