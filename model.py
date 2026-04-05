import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


#Load the dataset 
data = pd.read_csv('Dataset/CCPP_data.csv')

# Data Exploration 

## Display the first few rows of the dataset
#print(data.head())
## Check for missing values 
#print(data.info()) 
## Check for duplicates
#print(data.duplicated().sum())

### Summary satistics
#print(data.describe())

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# A. Creating Interaction terms 
data['AT_RH'] = data['AT'] * data['RH'] # Interaction between Ambient Temperature and Relative Humidity
data['V_AP'] = data['V'] * data['AP'] # Interaction between Air Velocity and Atmospheric Pressure

## Exploring the new featuers 


# Creating Polynonmial features (Non-linear relationships)
data['AT_squared'] = data['AT'] ** 2 # Square of Ambient Temperature
data['V_Squared'] = data['V'] ** 2 # Square of Air Velocity

# 3. Define features and target variable
X = data.drop('PE', axis=1) # Features (all columns except 'PE)
y = data['PE'] # Target variable (Power Output)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. FEATURE SCALING
# ==========================================

# Initialize the StandardScaler
scaler = StandardScaler() 
# Fit the scaler on the training data to prevent data leakage  and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- FINAL MODEL EVALUATION ---")

# 5. Train and evaluate the Linear Regression model
lr_model = LinearRegression()
lr_scores = cross_val_score(lr_model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
lr_rmse = np.mean(np.sqrt(-lr_scores))
print(f"Linear Regression Cross-Validation RMSE: {lr_rmse:.2f} MW")


# 6. Train and evaluate the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_scores = cross_val_score(rf_model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
rf_rmse = np.mean(np.sqrt(-rf_scores))
print(f"Random Forest Cross-Validation RMSE: {rf_rmse:.2f} MW")

# 7. Select the Best Model and Train it
print("\n--- FINAL MODEL EVALUATION ---")
if rf_rmse < lr_rmse:
    print("Random Forest performed better. Proceeding with Random Forest.")
    final_model = rf_model
else:
    print("Linear Regression performed better. Proceeding with Linear Regression.")
    final_model = lr_model

# Train the final model on the full scaled training set
final_model.fit(X_train_scaled, y_train)

# 9. Evaluate on the Test Set
y_pred = final_model.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f"Final Model Test RMSE: {test_rmse:.2f} MW")
print(f"Final Model Test R-squared: {test_r2:.4f}")
