'''##Stage 01 — Understand and inspect'''
# Import Libararies

# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load the dataset using Pandas

df = pd.read_csv("/content/drive/MyDrive/DataSet/student_performance.csv")

# Display the first five rows
df.head()

# Shape and column of data
print("Dataset Shape:", df.shape)

# number of rows and columns
print("Rows:    ",df.shape[0])
print("Columns: ",df.shape[1])


# name of columns
print("\nColumn Names:")
print(df.columns)

# Check all column names and data types
df.info()

# Find the missing values in each column
df.isnull().sum()

# Check whether duplicate rows exist.
print("Duplicate rows:", df.duplicated().sum())

# Generate descriptive statistics for the numerical columns.
df.describe()

"""##Stage 2 — Data cleaning




"""

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where the target is missing
df = df.dropna(subset=["FinalMarks"])

# Check the cleaned dataset
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

"""##Stage 3 — Exploratory Data Analysis (EDA)"""

# distribution of FinalMarks Histogram
plt.figure(figsize=(4,6))
plt.hist(df["FinalMarks"], bins=20)
plt.title("Distribution of FinalMarks")
plt.xlabel("Final Mark")
plt.show()

# Scatter plots
plt.figure(figsize= (4,6))
plt.scatter(df["StudyHours"],df["FinalMarks"])
plt.title("Relation Between Study Hours and Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.show()

plt.figure(figsize= (4,6))
plt.scatter(df["AttendancePct"],df["FinalMarks"])
plt.title("Relation Between AttendancePct and Final Marks")
plt.xlabel("Attendance")
plt.ylabel("Final Marks")
plt.show()

plt.figure(figsize= (4,6))
plt.scatter(df["PreviousScore"],df["FinalMarks"])
plt.title("Relation Between PreviousScore and Final Marks")
plt.xlabel("Previous Score")
plt.ylabel("Final Marks")
plt.show()

"""Correlation Matrix and Heatmap"""

# Correlation Matrix for the Entire DataFrame
df_corr = df.corr()
df_corr

# Correlation Between Two Specific Columns
correlation = df["StudyHours"].corr(df["FinalMarks"])
print("Correlation between Study Hours and Final Marks:", correlation)

# correlation matrix for numerical columns

"""Seaborn heatmap"""

# Seaborn heatmap
plt.figure(figsize=(4,4))
sns.heatmap(df_corr, annot = True, cmap="coolwarm", fmt=".2f",
vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

"""##Stage 4 — Separate Features (X) and Target (y)"""

# To split data into training and testing sets

X = df[["StudyHours", "AttendancePct", "PreviousScore"]]
y = df["FinalMarks"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=40)


print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""## Stage 5 — First Regression

Model: **Linear Regression**
"""

# Create a pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", LinearRegression())
])

# Fit (train) the model
model.fit(X_train,y_train)

# Predict the output for the test data
y_pred = model.predict(X_test)

comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison_df)

# MAE — Mean Absolute Error

mae = mean_absolute_error(y_test,y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# MSE — Mean Squared Error

mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# RMSE — Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# R² Score (Coefficient of Determination)

r2 = r2_score(y_test, y_pred)
print(f"R² Score (Variance Explained): {r2:.4f}")

"""Model Validation"""

# Create 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state = 42)

# Run cross-validation

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="neg_mean_absolute_error"
)

mae_scores = -cv_scores

print("Fold MAE Scores:", mae_scores)
print("Average CV MAE:", mae_scores.mean())

"""Stage 8 — Residual Analysis

"""

# Calculate residuals

residuals = y_test - y_pred
print(residuals.head())

# Residual scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Final Marks")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()

residuals.mean()

plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()

print("Mean Residual:", residuals.mean())

# Model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.named_steps["regressor"].coef_
})

print(coefficients)

# Model intercept
print("Intercept:", model.named_steps["regressor"].intercept_)

"""Baseline model"""

# Calculate average marks from training data
baseline_value = y_train.mean()

# Predict the same average for every test student
baseline_pred = np.full(len(y_test), baseline_value)

# Evaluate baseline
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("Baseline MAE:", baseline_mae)
print("Regression MAE:",mae)

baseline_r2 = r2_score(y_test, baseline_pred)

print("Baseline R2:", baseline_r2)
print("Linear Regression R2:", r2)

"""Model: **Decision Tree Regression**"""

# Create Decision Tree pipeline
tree_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# Train the model
tree_model.fit(X_train, y_train)

# Make predictions
tree_pred = tree_model.predict(X_test)

comparison_tree = pd.DataFrame({"Actual": y_test, "Predicted": tree_pred})
print(comparison_tree)

tree_mae = mean_absolute_error(y_test, tree_pred)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(y_test, tree_pred)

print("MAE:", tree_mae)
print("MSE:", tree_mse)
print("RMSE:", tree_rmse)
print("R2:", tree_r2)

"""Conclusion: For this dataset and this train/test split, Linear Regression performs better.

"""

# compare its training and testing performance

tree_train_pred = tree_model.predict(X_train)

tree_train_mae = mean_absolute_error(
    y_train,
    tree_train_pred
)

print("Training MAE:", tree_train_mae)
print("Testing MAE:", tree_mae)

"""Reduce Decision Tree complexity"""

# Decision Tree with maximum depth 3
tree_model2 = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", DecisionTreeRegressor(max_depth=3, random_state=42))
])

tree_model2.fit(X_train, y_train)

# Generate predictions
tree2_train_pred = tree_model2.predict(X_train)
tree2_test_pred = tree_model2.predict(X_test)

# Calculate MAE for both

tree2_train_mae = mean_absolute_error(y_train, tree2_train_pred)
tree2_test_mae = mean_absolute_error(y_test, tree2_test_pred)

print("Training MAE:", tree2_train_mae)
print("Testing MAE:", tree2_test_mae)

# Decision Tree with maximum depth 5
tree_model3 = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", DecisionTreeRegressor(max_depth=5, random_state=42))
])

tree_model3.fit(X_train, y_train)

# Generate predictions
tree3_train_pred = tree_model3.predict(X_train)
tree3_test_pred = tree_model3.predict(X_test)

# Calculate MAE for both

tree3_train_mae = mean_absolute_error(y_train, tree3_train_pred)
tree3_test_mae = mean_absolute_error(y_test, tree3_test_pred)

print("Training MAE:", tree3_train_mae)
print("Testing MAE:", tree3_test_mae)

plt.figure(figsize=(12, 8))

plot_tree(
    tree_model2.named_steps["regressor"],
    feature_names=X.columns,
    filled=True,
    rounded=True
)

plt.show()

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree (Unlimited)",
        "Decision Tree (Depth 3)",
        "Decision Tree (Depth 5)"
    ],
    "Test MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, tree_pred),
        mean_absolute_error(y_test, tree2_test_pred),
        mean_absolute_error(y_test, tree3_test_pred)
    ]
})

model_comparison

