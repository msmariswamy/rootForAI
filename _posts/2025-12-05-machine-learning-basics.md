---
layout: post
title: "Machine Learning Basics: Supervised vs Unsupervised"
date: 2025-12-05
path_type: beginner
categories:
  - machine-learning
  - fundamentals
read_time: 12
---

Learn the two main approaches to machine learning and when to use each one.

## Supervised Learning

Supervised learning uses labeled data to train models. The algorithm learns to map inputs to known outputs.

### Common Supervised Learning Algorithms

| Algorithm | Use Case | Example |
|-----------|----------|---------|
| Linear Regression | Predicting continuous values | House price prediction |
| Logistic Regression | Classification | Spam detection |
| Decision Trees | Classification & Regression | Customer churn prediction |
| Support Vector Machines | Classification | Image recognition |

### Real-world Example

```python
from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]  # Features
y = [100, 200, 300, 400, 500]  # Target values

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])
print(f"Predicted value: ${prediction[0]}")
```

## Unsupervised Learning

Unsupervised learning discovers patterns in data without labeled examples.

### Common Unsupervised Learning Algorithms

- **K-Means Clustering**: Grouping similar data points
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Association**: Finding relationships between variables

## When to Use Each

| Scenario | Approach |
|----------|----------|
| You have labeled data | Supervised |
| Discovering hidden patterns | Unsupervised |
| Predicting known outcomes | Supervised |
| Market segmentation | Unsupervised |

---

*Reading time: 12 minutes | Difficulty: Beginner*
