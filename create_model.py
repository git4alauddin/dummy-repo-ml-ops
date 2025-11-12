from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

print("Training a simple Iris model...")
iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, 'iris_model.pkl')
print("Model saved to iris_model.pkl!")
