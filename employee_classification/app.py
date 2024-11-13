from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

# Load the saved model
model = joblib.load('best_model_MLP.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # # Map Gender and EverBenched to integers
    # gender_map = {'Male': 0, 'Female': 1}
    # benched_map = {'Yes': 1, 'No': 0}
    # education_map={'Bachelors':1,'Masters':2}

    features = {
    'Age': [int(data['age'])],
    'Gender': [(data['gender'])],
    'PaymentTier': [int(data['payment_tier'])],
    'ExperienceInCurrentDomain': [int(data['experience'])],
    'Education': [(data['education'])],
    'City': [(data['city'])], # for the some time using integer for this 
    'JoiningYear': [int(data['joining_year'])],
    'EverBenched': [(data['ever_benched'])]

}
    # Convert the features to a DataFrame
    features_df = pd.DataFrame(features)
    # # Ensure the correct data types for each column
    # features_df['City'] = features_df['City'].astype(str)
     
    features_df=features_df.dropna()
    # Apply one-hot encoding on the specified columns (ensure to pass a list of column names)
    # features_df1 = pd.get_dummies(features_df, columns=['Gender', 'Education', 'City', 'EverBenched'])
    


    # Make prediction
    prediction = model.predict(features_df)[0]
    
    prediction_text = 'Stay' if prediction == 1 else 'Leave'

    # prediction_text = {1: 'Leave', 0: 'Stay'}.get(prediction, 'Stay')


    # Render result.html and pass the prediction result
    return render_template('result.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

