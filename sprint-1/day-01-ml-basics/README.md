# Day 01 – ML Basics

## 1️⃣ Concept (Intuition)
- ML = learn from data → predict output
- Feature (X) = input
- Label (y) = output
- Model = algorithm → learns X→y
- Engineer mindset: ML is a component in system, not magic

## 2️⃣ Tools / Libraries Used
- **Pandas** – data manipulation
- **Scikit-learn** – ML algorithms
- **Matplotlib** – visualization
- **Jupyter Notebook** – interactive coding

## 3️⃣ Code (Hands-on)
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    "size": [1200, 1500, 1800, 2000],
    "bedrooms": [3, 4, 3, 5],
    "price": [200000, 260000, 300000, 400000]
}
df = pd.DataFrame(data)

# Inspect Data
print(df)

# Visualize
plt.scatter(df["size"], df["price"])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Prices")
plt.show()

# Train Model
X = df[["size","bedrooms"]]
y = df["price"]
model = LinearRegression()
model.fit(X, y)

# Predict
new_houses = [[1700,3], [2100,4]]
predictions = model.predict(new_houses)
print(predictions)
