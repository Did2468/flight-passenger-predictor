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



model=xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.25,
    max_depth=10,
    random_state=69
)
model.fit(X,y)


model_filename='final_xgboost_model.joblib'
joblib.dump(model, model_filename)

print(f"Trained model saved to {model_filename}")


training_cols = X.columns.tolist()
joblib.dump(training_cols, "training_cols.joblib")
print("Feature columns saved to training_cols.joblib")

