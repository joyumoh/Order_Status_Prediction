# Order_Status_Prediction

A multiclass classification project that predicts whether a customer order will be **Delivered**, **Cancelled**, or **Returned** — helping e-commerce businesses reduce losses, flag risky transactions early, and improve fulfilment operations.

---

## The Problem

Returns and cancellations are costly. By predicting order outcomes before they happen, businesses can intervene proactively — whether that means adjusting inventory, reaching out to customers, or flagging suspicious transactions for review.

---

## Dataset

1,200 customer orders from `Online-Store-Orders.xlsx`, with the following key features:

| Feature | Description |
|---|---|
| `OrderStatus` | Target — Delivered, Cancelled, Returned, Shipped, Pending |
| `Product` | Item ordered |
| `Quantity` / `UnitPrice` / `TotalPrice` | Order financials |
| `PaymentMethod` | Payment type |
| `CouponCode` | Discount code used (if any) |
| `ReferralSource` | How the customer arrived |
| `Date` | Order date (parsed into Day, Month, DayOfWeek) |

Identifier columns (`OrderID`, `CustomerID`, `ShippingAddress`, `TrackingNumber`) were dropped before modeling.

---

## Methodology

**Cleaning** — Missing `CouponCode` values filled with `'Unknown'`; `Date` parsed and decomposed into temporal features.

**Encoding** — Categorical features label-encoded; numerical features scaled with `StandardScaler`.

**Class imbalance** — Addressed on the training set using SMOTE (Synthetic Minority Over-sampling Technique).

**Models trained** — Decision Tree, Logistic Regression, Random Forest, XGBoost. Data split 80/20 with stratification.

**Tuning** — Random Forest tuned via `GridSearchCV` optimizing for `f1_macro`.

---

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 0.170 |
| Logistic Regression | 0.175 |
| XGBoost | 0.208 |
| **Random Forest** | **0.225** |

Random Forest was selected for hyperparameter tuning. Best configuration: `n_estimators=300`, `max_depth=10`, `min_samples_split=2` — achieving a cross-validated **F1 score of 0.25**.

The low scores across all models suggest the prediction task is genuinely difficult given the available features, and point toward areas for future improvement (see below).

---

## What's Next

The current models establish a baseline, but there's meaningful room to grow:

- Richer feature engineering — customer history, product return rates, delivery zone data
- Advanced models — LightGBM, CatBoost, or neural networks
- Better imbalance handling — cost-sensitive learning or alternative resampling strategies
- More data — 1,200 rows is a relatively small sample for a five-class problem

---

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `imbalanced-learn` · `xgboost`
