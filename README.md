# 📊 Task 5 – Sales Prediction Using Python

> **Internship Project** | Machine Learning | Linear Regression

---

## 📌 Overview

Sales prediction means predicting how much of a product people will buy based on factors such as:
- The **amount spent on advertising** (TV, Radio, Newspaper)
- The **segment of people** being targeted
- The **platform** being used for advertising

This project uses **Machine Learning (Linear Regression)** to predict future sales based on advertising spend data.

---

## 📁 Project Structure

```
sales_prediction/
│
├── advertising.csv          # Dataset (TV, Radio, Newspaper → Sales)
├── sales_prediction.py      # Main Python script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## 📂 Dataset

**File:** `advertising.csv`

| Column | Description |
|--------|-------------|
| `TV` | Budget spent on TV advertising (in thousands) |
| `Radio` | Budget spent on Radio advertising (in thousands) |
| `Newspaper` | Budget spent on Newspaper advertising (in thousands) |
| `Sales` | Units of product sold (in thousands) — **Target Variable** |

- **Rows:** 100 records
- **Source:** Classic Advertising Dataset

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Heatmaps & pairplots |
| Scikit-learn | ML model (Linear Regression) |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Yashsompura07/sales-prediction.git
cd sales-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python sales_prediction.py
```

---

## 📈 ML Workflow

```
Load Dataset
     ↓
Exploratory Data Analysis (EDA)
     ↓
Feature Selection (TV, Radio, Newspaper)
     ↓
Train / Test Split (80% / 20%)
     ↓
Train Linear Regression Model
     ↓
Predict Sales
     ↓
Evaluate (MAE, RMSE, R² Score)
     ↓
Visualize Results
```

---

## 📊 Outputs & Visualizations

The script generates the following plots:

| File | Description |
|------|-------------|
| `correlation_heatmap.png` | Correlation between all features and sales |
| `pairplot.png` | Scatter plots of each ad channel vs sales |
| `actual_vs_predicted.png` | How closely predictions match real values |
| `residual_plot.png` | Distribution of prediction errors |

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | ~1.3 |
| RMSE (Root Mean Squared Error) | ~1.7 |
| R² Score | ~0.90 |

> R² Score of ~0.90 means the model explains **90% of the variance** in sales — a strong result!

---

## 💡 Key Insight

**TV advertising** has the strongest correlation with Sales, followed by Radio. Newspaper advertising has comparatively little impact on sales predictions.

---

## 👤 Author

**Yash Sompura**
- GitHub: https://github.com/Yashsompura07

---
