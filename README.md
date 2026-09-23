# Employee Attrition Prediction & HR Analytics

An end-to-end Machine Learning system built to predict employee turnover and identify key operational drivers behind attrition. By leveraging HR data—such as job satisfaction, monthly income, tenure, and work-life balance—this project enables corporate decision-makers to proactively retain top talent and reduce costly hiring overhead.

---

## Executive Summary

Employee turnover is a major cost driver in corporate operations. This project builds and evaluates three classification models (**Logistic Regression**, **Random Forest**, and **Gradient Boosting**) on HR workforce data to predict whether an employee is likely to leave (**Attrition: Yes/No**). 

By addressing severe class imbalance and evaluating models through business-critical metrics (**Recall** and **F1-Score**), we isolate the top 10 workplace factors influencing exit rates—delivering clear, actionable insights for HR teams.

---

## Project Workflow

### **Task 1 — Data Loading & Exploration**
* **Initial Inspection:** Loaded the dataset using `pandas` and examined the first 10 rows to inspect schema and feature types.
* **Target Identification:** Identified the target variable (`Attrition`: `Yes` vs. `No`).
* **Attrition Rate:** Computed overall employee turnover rate and identified class distribution.
* **Feature Analysis:** Segregated columns into numeric attributes (e.g., `MonthlyIncome`, `YearsAtCompany`) and categorical variables (e.g., `Department`, `JobRole`).
* **Observation on Class Imbalance:** Noted a significant class imbalance (approx. 84% stayed vs. 16% left), requiring balanced loss weighting during model training.

### **Task 2 — Data Cleaning & Preprocessing**
* **Null Value Handling:** Checked for missing values across all columns and imputed/cleaned missing records.
* **Feature Elimination:** Dropped non-predictive, zero-variance, or static administrative columns (e.g., `EmployeeNumber`, `Over18`, `StandardHours`).
* **Target Encoding:** Converted binary target `Attrition` into numerical values (`Yes` $\rightarrow$ `1`, `No` $\rightarrow$ `0`).
* **Categorical Encoding:** Applied **One-Hot Encoding** (`pd.get_dummies` / `OneHotEncoder`) to convert multi-class attributes (`Department`, `JobRole`, `MaritalStatus`, `BusinessTravel`) into binary vectors.
* **Feature Scaling:** Applied `StandardScaler` to numerical features to ensure equitable distance/gradient calculations across algorithms.

---

## Task 3 — Exploratory Data Analysis & Business Insights

* **Department Analysis:** `Sales` and `Human Resources` departments exhibit significantly higher attrition percentages compared to `Research & Development`.
* **Job Role Exit Rates:** Frontline operational roles like `Sales Executive`, `Laboratory Technician`, and `Research Scientist` show elevated turnover.
* **Income Disparity:** Employees with lower `MonthlyIncome` show a noticeably higher density of exit cases compared to higher-salaried peers.
* **Work-Life Balance Impact:** Employees rating their `WorkLifeBalance` as `1` (Poor) or `2` (Fair) exhibit higher exit rates, confirming burnout as a major driver.
* **Tenure Vulnerability:** Attrition peaks between **Years 1 and 3** of tenure, showing that early-career employees represent the highest exit risk.

### **Key Business Takeaways for HR**
1. **Early Career Flight Risk:** Retention efforts must target employees within their first 3 years of tenure to curb early exit patterns.
2. **Competitive Compensation at Base Levels:** Lower-income bands correlate directly with higher turnover; market salary adjustments at lower job levels yield immediate retention benefits.
3. **Targeted Role Interventions:** Sales Executives and Lab Technicians require structured career progression paths to mitigate high churn.
4. **Work-Life Balance Warning Indicator:** Low work-life balance scores serve as a reliable early warning signal for voluntary resignation.

---

## Task 4 & 5 — Model Building, Evaluation & Selection

Data was split into an **80/20 train-test ratio**. To account for class imbalance, models were trained using `class_weight='balanced'`. Three classifiers were built and evaluated using **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **Confusion Matrices**:

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.46 | 0.78 | 0.58 | 0.82 |
| **Random Forest Classifier** | 0.62 | 0.68 | 0.65 | 0.85 |
| **Gradient Boosting Classifier (Best)** | **0.65** | **0.74** | **0.69** | **0.88** |

### **Best Performing Model: Gradient Boosting Classifier**
* **Why it won:** In HR attrition modeling, **Recall** is critical (minimizing False Negatives where an at-risk employee is missed). **Gradient Boosting** achieved the highest overall **F1-Score (0.69)** and **ROC-AUC (0.88)** while maintaining a strong balance between Precision and Recall.

---

## Top 10 Features Driving Attrition

Extraction of feature importances from the top model highlights the following key predictors:

1. **`OverTime_Yes`** — Working overtime is the single strongest predictor of employee departure.
2. **`MonthlyIncome`** — Lower compensation levels directly accelerate exit decisions.
3. **`TotalWorkingYears`** — Overall career experience influences stability and job mobility.
4. **`YearsAtCompany`** — Tenure length dictates organizational attachment.
5. **`StockOptionLevel`** — Lack of equity/stock options reduces long-term incentive alignment.
6. **`YearsWithCurrManager`** — Managerial relationships heavily sway retention outcomes.
7. **`Age`** — Younger employees demonstrate higher voluntary mobility.
8. **`JobSatisfaction`** — Low satisfaction scores directly correlate with active job searches.
9. **`EnvironmentSatisfaction`** — Negative workplace culture drives turnover.
10. **`WorkLifeBalance`** — Poor work-life integration accelerates burnout exits.

---

## Task 6 — Visualizations

The project generates five distinct visualizations:
* **Chart 1:** Categorical Bar Chart — Attrition rate breakdown across Departments and Job Roles.
* **Chart 2:** Box Plot — Distribution comparison of `MonthlyIncome` for stayed vs. left employees.
* **Chart 3:** Heatmap — Confusion Matrix visual for the primary model.
* **Chart 4:** Horizontal Bar Chart — Ranking of the Top 10 Feature Importances.
* **Chart 5 (Bonus):** Combined ROC Curve — Overlay plot comparing Receiver Operating Characteristic curves for all 3 models.

---

## Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Data Processing & Analytics:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `StandardScaler`, `OneHotEncoder`)
* **Data Visualization:** `matplotlib`, `seaborn`

---
