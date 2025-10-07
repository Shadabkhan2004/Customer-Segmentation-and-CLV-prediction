# Customer Segmentation + Customer Lifetime Value (CLV) Prediction

## Project Overview
This project demonstrates an end-to-end workflow for **segmenting customers** and **predicting their 6-month CLV** using historical transactional data from the [UCI Online Retail II dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).  
The goal is to provide actionable business insights, identify high-value customers, and forecast future revenue.

---

## Dataset
- **Source:** Kaggle - mashlyn/online-retail-ii-uci
- **Columns used:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
- **Transactions filtered:** Removed negative/zero quantities, and missing Customer IDs

---

## Methodology

### 1. Preprocessing
- Cleaned raw transaction data
- Calculated **SalesAmount = Quantity × UnitPrice**
- Removed canceled/invalid transactions
- Defined a **snapshot date** as 6 months before the last transaction (`last_date - 6 months`)  
  - **df_snapshot:** all transactions on or before snapshot → used to calculate RFM features  
  - **df_future:** transactions in the 6 months after snapshot → used to compute actual 6-month CLV

### 2. RFM Feature Engineering
- **Recency:** Days since last purchase before snapshot
- **Frequency:** Number of purchases before snapshot
- **Monetary:** Total spending before snapshot
- **AvgMonetary:** Average spend per transaction
- Log-transformed all RFM features for better modeling

### 3. Customer Segmentation
- Applied **KMeans clustering (k=6)** on log-RFM features
- Named segments based on behavior:
  - Best Customers
  - Loyal Customers
  - Promising
  - At Risk
  - Low Value
  - Lost
- Summarized **segment counts and average CLV**

### 4. CLV Calculation
- Calculated **actual 6-month CLV (`CLV_6M`)** from transactions after snapshot date
- Transformed with `log1p` for modeling: `CLV_6M_log`
- Optional features: cluster, AvgMonetary_log

### 5. CLV Prediction Model
- Model: **XGBoost Regressor**
- Features: `Recency_log`, `Frequency_log`, `Monetary_log`, `AvgMonetary_log`, `Cluster`
- Target: `CLV_6M_actual_log`
- Train/test split: 80%/20%
- Metrics (log scale):
  - RMSE: 2.70
  - R²: 0.34
- Feature importance analyzed to identify key drivers of CLV

---

## Results & Insights
- **Cluster analysis** shows:
  - Best Customers: high frequency & monetary, high predicted CLV
  - Loyal Customers: frequent but moderate spending
  - At Risk: high recency, low frequency, low predicted CLV
- **RFM features** are strong predictors of future customer value
- Model captures **general trends**, but high variance in CLV is expected due to noisy behavior

---

## Why Predictive Performance is Moderate
While the model captures general trends in customer value, the predictive accuracy is moderate (R² ≈ 0.34) due to the following reasons:

1. **Inherent variability in customer behavior**  
   - Customers’ future spending is influenced by promotions, seasonality, personal circumstances, and external factors that are **not captured by RFM features**.  

2. **Limited feature set**  
   - The model uses only **historical RFM metrics, average monetary value, and cluster labels**.  
   - Additional features like inter-purchase times, purchase trends, or product preferences could improve predictions but are not included in this project.  

3. **Skewed and heavy-tailed CLV distribution**  
   - CLV is naturally **highly skewed**: a few customers spend much more than average.  
   - Log transformation helps, but extreme values still make exact prediction difficult.  

4. **Prediction horizon**  
   - We are predicting **6 months into the future**. Longer horizons increase uncertainty as behavior may change unexpectedly.  

> Despite these limitations, the model is useful for **identifying high-value segments** and **capturing relative differences between customers**, which is often the primary goal in business analytics.

---

## Visualizations
- **Feature Importance:** Shows which RFM features influence CLV prediction
- Predicted vs Actual CLV scatter plot (log scale)  

---

## How to Run
1. Clone the repo
2. Create a conda environment:
   ```bash
   conda create -n clv_env python=3.10
   conda activate clv_env
   pip install -r requirements.txt
