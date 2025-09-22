import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

df = pd.read_csv("data.csv",low_memory=False)

month_map = {
    'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
    'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12
}
df['Month_num'] = df['Month'].map(month_map)

df['Activity_Period'] = pd.to_datetime(df['Activity Period'].astype(str), format='%Y%m')

drop_cols = ['Passenger Count', 'Adjusted Activity Type Code', 'Activity Period', 'Month', 'Year',
             'Operating Airline IATA Code', 'Published Airline', 'Published Airline IATA Code', 'Activity_Period','Terminal', 'Boarding Area']
df_model = df.drop(columns=drop_cols)

categorical_cols = ['Operating Airline', 'GEO Summary', 'GEO Region',
                    'Activity Type Code', 'Price Category Code']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)


X=df_model.drop(columns=['Adjusted Passenger Count'])
y=df_model['Adjusted Passenger Count']

train_size=int(0.8*len(df_model))
X_train,X_test=X[:train_size],X[train_size:]
y_train,y_test=y[:train_size],y[train_size:]

model=xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.25,
    max_depth=10,
    random_state=69
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
print(f"RMSE:{rmse}")
print(f"R_2 score:{r2}")

model_filename='final_xgboost_model.joblib'
joblib.dump(model, model_filename)

print(f"Trained model saved to {model_filename}")



print("\n--- Interactive Passenger Count Prediction ---")
print("Please enter the values for a single flight event.")


month_num = int(input("Enter month (1-12): "))
op_airline = input("Enter Operating Airline (e.g., 'United Airlines'): ")
geo_summary = input("Enter GEO Summary (e.g., 'Domestic'): ")
geo_region = input("Enter GEO Region (e.g., 'US'): ")
activity_type = input("Enter Activity Type Code (e.g., 'Deplaned'): ")
price_category = input("Enter Price Category Code (e.g., 'Other'): ")


user_input = pd.DataFrame([{
    'Month_num': month_num,
    'Operating Airline': op_airline,
    'GEO Summary': geo_summary,
    'GEO Region': geo_region,
    'Activity Type Code': activity_type,
    'Price Category Code': price_category
}])


user_input_encoded = pd.get_dummies(user_input, columns=categorical_cols, drop_first=True)
user_input_aligned = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)


predicted_count = model.predict(user_input_aligned)


print(f"\nPredicted Adjusted Passenger Count: {int(predicted_count[0])}")
