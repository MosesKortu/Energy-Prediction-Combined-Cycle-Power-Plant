# Combined Cycle Power Plant - Energy Output Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

> **Author:** Moses Bargue Kortu Jr. <br>
> **Task:** Supervised Machine Learning (Regression)

## 📖 Project Overview
This project builds a robust machine learning pipeline to predict the **net hourly electrical energy output (PE)** of a Combined Cycle Power Plant (CCPP). The plant uses a combination of gas turbines, steam turbines, and heat recovery steam generators. 

By accurately predicting energy output based on ambient environmental conditions, plant operators can optimize grid dispatching and fuel scheduling, leading to greater efficiency and cost savings.

## 📊 The Dataset
The model is trained on a dataset containing **9,568 hourly average ambient environmental readings** collected from sensors at the power plant. 

### Features (Inputs)
| Feature | Code | Unit | Range | Description |
| :--- | :---: | :---: | :--- | :--- |
| **Temperature** | `AT` | °C | 1.81 - 37.11 | Ambient external temperature. |
| **Exhaust Vacuum** | `V` | cmHg | 25.36 - 81.56 | Vacuum pressure in the steam turbine. |
| **Ambient Pressure** | `AP` | mbar | 992.89 - 1033.30 | Environmental atmospheric pressure. |
| **Relative Humidity**| `RH` | % | 25.56 - 100.16 | Moisture in the air. |

### Target (Output)
* **Net Electrical Energy Output (`PE`)**: Measured in Megawatts (MW). Range: 420.26 - 495.76 MW.

---

## 🛠️ Modeling Approach

### 1. Feature Engineering
To capture complex environmental relationships, I engineered several new features before modeling:
* **Interaction Terms:** `AT * RH` (Heat index/moisture effect) and `V * AP` (Pressure differential).
* **Polynomial Features:** Squared terms (`AT²`, `V²`) to capture non-linear degradation in plant efficiency at extreme conditions.
* **Feature Scaling:** Applied `StandardScaler` to ensure all numerical inputs contributed equally to gradient calculations.

### 2. Model Selection & Validation
* **Data Split:** 80% Training, 20% Hold-out Test Set.
* **Validation Strategy:** 5-Fold Cross-Validation on the training set to prevent data leakage and overfitting.
* **Algorithms Compared:** Linear Regression (Baseline) vs. Random Forest Regressor (Ensemble).

---

## 🏆 Results & Evaluation
The models were evaluated using **Root Mean Squared Error (RMSE)**—which measures the average prediction error directly in Megawatts (MW)—and **R-squared ($R^2$)**.

| Model | Cross-Validation RMSE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 4.31 MW | ~4.31 MW | - |
| **Random Forest** 🌟 | **3.40 MW** | **~3.40 MW** | **~ 0.96** |


### Interpretation
The **Random Forest model** outperformed the baseline by **21%**. Because it is an ensemble of hundreds of decision trees, it inherently captured the non-linear interactions between temperature, humidity, and vacuum pressure that the rigid Linear Regression model missed. 

An R-squared of **0.96** indicates that our model explains 96% of the variance in the power plant's output. Furthermore, an average error of just **3.40 MW** (on a scale of 420-495 MW) proves this model is highly reliable for production-grade forecasting.

---

## 🚀 Next Steps & Future Work
1. **Hyperparameter Tuning:** Utilize `GridSearchCV` to optimize `n_estimators`, `max_depth`, and `min_samples_split` to push RMSE below 3.0 MW.
2. **Deployment:** Containerize the model using Docker and expose it via a FastAPI REST endpoint to connect to live plant sensor data streams.
3. **Model Monitoring:** Track prediction drift over time and schedule quarterly retraining pipelines to adapt to shifting seasonal baselines.

---

## 💻 How to Run This Project Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/ccpp-energy-prediction.git](https://github.com/YourUsername/ccpp-energy-prediction.git)
   cd ccpp-energy-prediction

2. ** Set up a virtual environment (Recommended):** 

Bash
```python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

3. ** Bash **
```pip install pandas numpy scikit-learn
Run the model script:

4. ** Bash ** 
``` python model.py
Developed by Moses Bargue Kortu Jr for the Machine Learning Modeling Project.
