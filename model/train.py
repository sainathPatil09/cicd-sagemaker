# model/train.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
