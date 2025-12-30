# Employee Attrition Prediction – Production-Ready ML Pipeline

## 🔗 Live Deployment
**API Endpoint:**  
- Root Link: *https://ibm-hr-analytics.onrender.com*
- Testing Link: *https://ibm-hr-analytics.onrender.com/docs*
- Endpoint Link: *https://ibm-hr-analytics.onrender.com/predict*

---

## 📌 Project Overview

Employee attrition is a critical HR problem where **missing a potential resignation is more costly than raising a false alert**.  
This project builds an **end-to-end, deployment-ready machine learning pipeline** to predict employee attrition using structured HR data.

The focus of this project is not only model accuracy, but:
- Handling class imbalance correctly
- Building honest probability estimates
- Selecting an optimal decision threshold
- Designing a pipeline safe for production deployment

---

## 📊 Dataset Summary

- Total records: **1470**
- Target variable: **Attrition**
  - `Yes`: 237 (~16%)
  - `No`: 1233 (~84%)

This is a **highly imbalanced binary classification problem**, where accuracy alone is misleading.

---

## 🧩 Feature Categorization & Handling

### 1️⃣ Categorical (Nominal)
BusinessTravel, Department, EducationField, Gender,
JobRole, MaritalStatus, OverTime


**Handling:**
- One-Hot Encoding
- `drop="first"` to avoid multicollinearity
- `handle_unknown="ignore"` for deployment safety

---

### 2️⃣ Ordinal Features
Education, EnvironmentSatisfaction, JobInvolvement,
JobSatisfaction, PerformanceRating, RelationshipSatisfaction,
WorkLifeBalance, JobLevel, StockOptionLevel


**Decision:**
- Kept as numeric (ordering is meaningful)
- No one-hot encoding applied
- Scaled along with numeric features

---

### 3️⃣ Numeric (Continuous)
Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany,
YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, etc.


**Outlier Handling Decision:**
- Outliers are **true business values**, not data errors
- Rows were **not dropped**
- Values were **not capped**
- Used **RobustScaler** to reduce outlier influence safely

---

## 🔧 Preprocessing Pipeline

All preprocessing is handled using a single `ColumnTransformer`:

- OneHotEncoder → categorical features
- RobustScaler → numeric + ordinal features

This ensures:
- No data leakage
- Consistent behavior during inference
- Clean deployment with serialized pipeline

---

## 🤖 Model Selection

### Logistic Regression (Baseline Model)

Chosen because:
- Interpretable
- Stable
- Strong baseline for tabular HR data
- Works well with calibrated probabilities

**Configuration:**
- `class_weight="balanced"`
- `solver="liblinear"`
- `max_iter=1000`

---

## 🧪 Train / Validation / Test Split

- Train: **70%**
- Validation: **15%**
- Test: **15%**

**Stratified splitting** was used to preserve class distribution.

### Why Validation Data?
- Probability calibration
- Threshold selection
- Prevents optimistic bias
- Ensures honest evaluation

---

## 🎯 Probability Calibration

Raw model probabilities are often **over-confident**, especially in imbalanced datasets.

**Solution:**
- `CalibratedClassifierCV` with **Isotonic Regression**
- Model trained on training data
- Calibration learned on validation data

This ensures:
- Reliable probabilities
- Stable threshold behavior in production

---

## 🔢 Threshold Selection Strategy (F1-Score Based)

### Why threshold tuning is required
- Default threshold (0.5) is rarely optimal for imbalanced data
- Business requires a balance between:
  - Capturing attrition cases (Recall)
  - Avoiding excessive false alerts (Precision)

### Final Decision
The decision threshold was selected by **maximizing the F1-score** on the validation set.

**Reason:**
- F1-score provides a balanced trade-off between precision and recall
- Simple, interpretable, and commonly accepted baseline strategy
- Suitable when explicit business costs are not yet defined

---

## 📈 Model Performance (Test Set)

**Attrition = Yes (Positive Class):**

- Recall ≈ **0.69**
- Precision ≈ **0.41**
- F1-score ≈ **0.52**

**Overall Accuracy:** ≈ **0.79**

These results are:
- Realistic for HR attrition data
- Achieved without data leakage
- Stable after calibration and threshold tuning

---

## 🚀 Deployment Details

- Entire pipeline (preprocessing + model) is serialized
- Final model returns:
  - `1` → Attrition risk
  - `0` → No attrition risk
- Integrated with FastAPI for real-time inference
- Ready for production usage

---

## ✅ Key Design Principles Followed

- No data leakage
- Honest probability estimation
- Proper handling of class imbalance
- Pipeline-based preprocessing
- Threshold-aware decision making
- Deployment-safe architecture

---

## 🔮 Future Improvements

- Business cost-based threshold optimization
- Gradient Boosting / Tree-based models
- Model monitoring & drift detection
- Explainability (SHAP)
- Feedback loop from HR outcomes

---

## 📝 Final Note

This project prioritizes **decision quality over metric chasing**.  
The goal is to build a model that behaves **reliably in real-world conditions**, not just one that performs well on paper.

---

## 👤 Author
### Subir Kumar Behera
Aspiring Data Analyst | Machine Learning Enthusiast

