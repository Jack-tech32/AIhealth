import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load processed tabular data
df = pd.read_csv("../data/processed/clean_healthcare.csv")


# Classification

# Target
y_clf = df['outcome']
X_clf = df.drop(columns=['outcome','length_of_stay_days'])

# Encode target
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf_enc, test_size=0.2, random_state=42)

# Train XGBoost Classifier
clf = XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
pred = clf.predict(X_test)
print(classification_report(y_test, pred))

# Save classification model and label encoder
joblib.dump(clf, "../models/classifier_xgb.pkl")
joblib.dump(le, "../models/label_encoder.pkl")
print("Classification Model Saved Successfully")


# Regression Model (Length of Stay)


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import pandas as pd

# Load processed clean tabular data
df = pd.read_csv("../data/processed/clean_healthcare.csv")

# Target and features
y_reg = df['length_of_stay_days']
X_reg = df.drop(columns=['length_of_stay_days', 'outcome'])  

# Train-test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
reg = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

reg.fit(X_train_r, y_train_r)

# Predict
y_pred = reg.predict(X_test_r)

# Evaluation
mae = mean_absolute_error(y_test_r, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))  
r2 = r2_score(y_test_r, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Save regression model
joblib.dump(reg, "../models/regressor_xgb.pkl")

print("Regression Model Saved Successfully")
