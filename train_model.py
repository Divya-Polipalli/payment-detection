import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data/fraud.csv")

# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Features
X = df[['type', 'amount', 'oldbalanceOrg',
        'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest']]

# Target
y = df['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=10,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, 'model/model.pkl')

print("Model Saved Successfully!")