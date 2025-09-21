from flask import Flask,render_template,request
import joblib
import pandas as pd

model_xg = joblib.load("models/model-xg.joblib")
model_linear=joblib.load("models/model-linear.joblib")
model_polynomial=joblib.load("models/model-polynomial.joblib")
model_rf=joblib.load("models/model-rf.joblib")
feature_columns=joblib.load("models/features.joblib")

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
    prediction=None
    if request.method=='POST':
        month = int(request.form.get('month'))
        airline = request.form.get('operatingAirline')
        geo_summary = request.form.get('geoSummary')
        geo_region = request.form.get('geoRegion')
        activity_type = request.form.get('activityTypeCode')
        price_category = request.form.get('priceCategoryCode')
        model = request.form.get('model')


        input_df = pd.DataFrame({
            "Month_num": [month],
            "Operating Airline": [airline],
            "GEO Summary": [geo_summary],
            "GEO Region": [geo_region],
            "Activity Type Code": [activity_type],
            "Price Category Code": [price_category]
        })

        
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        if model=='XGBoost':
            prediction = abs(int(model_xg.predict(input_encoded)[0]))
        elif model=='Linear':
            prediction = abs(int(model_linear.predict(input_encoded)[0]))
        elif model=='Polynomial':
            prediction = abs(int(model_polynomial.predict(input_encoded)[0]))
        elif model=='Random Forest':
            prediction = abs(int(model_rf.predict(input_encoded)[0]))
        elif model=='All':
            xg=abs(int(model_xg.predict(input_encoded)[0]))
            l=abs(int(model_linear.predict(input_encoded)[0]))
            p=abs(int(model_polynomial.predict(input_encoded)[0]))
            r=abs(int(model_rf.predict(input_encoded)[0]))
            prediction=abs(int(((0.7*xg)+(0.54*p)+(0.49*l)+(0.11*r))/(0.70+0.54+0.49+0.11)))
    return render_template("index.html",prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True, port=3000)

