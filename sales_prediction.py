# ============================================================
# Task 5 - Sales Prediction Using Python
# Internship Project | Machine Learning with Linear Regression
# ============================================================

# ----- Import Libraries -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ----- 1. Load Dataset -----
df = pd.read_csv('advertising.csv')

print("=" * 50)
print("SALES PREDICTION USING PYTHON")
print("=" * 50)

print("\n[1] Dataset Overview")
print(f"Shape: {df.shape}")
print(df.head())

# ----- 2. Exploratory Data Analysis -----
print("\n[2] Statistical Summary")
print(df.describe())

print("\n[3] Checking for Null Values")
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Correlation Heatmap - Advertising vs Sales')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print(">> Saved: correlation_heatmap.png")

# Pairplot
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales',
             height=4, aspect=0.8, kind='scatter')
plt.suptitle('Advertising Spend vs Sales', y=1.02)
plt.tight_layout()
plt.savefig('pairplot.png', dpi=150)
plt.show()
print(">> Saved: pairplot.png")

# ----- 3. Feature Engineering -----
X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']                        # Target

# ----- 4. Train/Test Split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[4] Data Split")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ----- 5. Train the Model -----
model = LinearRegression()
model.fit(X_train, y_train)

print("\n[5] Model Coefficients")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coef_df)
print(f"Intercept: {model.intercept_:.4f}")

# ----- 6. Predictions -----
y_pred = model.predict(X_test)

# ----- 7. Model Evaluation -----
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n[6] Model Performance")
print(f"Mean Absolute Error  (MAE)  : {mae:.4f}")
print(f"Mean Squared Error   (MSE)  : {mse:.4f}")
print(f"Root Mean Sq. Error  (RMSE) : {rmse:.4f}")
print(f"R² Score                    : {r2:.4f}")

# ----- 8. Actual vs Predicted Plot -----
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='darkorange', edgecolors='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'b--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print(">> Saved: actual_vs_predicted.png")

# ----- 9. Residual Plot -----
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, color='steelblue', edgecolors='black', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=150)
plt.show()
print(">> Saved: residual_plot.png")

# ----- 10. Predict New Sales -----
print("\n[7] Predict Sales for New Input")
sample = pd.DataFrame({
    'TV': [200],
    'Radio': [30],
    'Newspaper': [50]
})
predicted_sales = model.predict(sample)
print(f"TV=200, Radio=30, Newspaper=50 → Predicted Sales: {predicted_sales[0]:.2f} units")

print("\n" + "=" * 50)
print("Task 5 Completed Successfully!")
print("=" * 50)
