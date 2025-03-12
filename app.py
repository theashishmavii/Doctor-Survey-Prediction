from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load original data for doctor details
data = pd.read_excel("dummy_npi_data.xlsx")

# Prediction function
def predict_best_doctors(input_time):
    input_hour = int(input_time)  # Fix for hour-only format
    input_features = pd.DataFrame({
        'Login Hour': [input_hour],
        'Logout Hour': [input_hour + 1],
        'Session Duration': [1],
        'Is Weekend': [0],
        'Active Period': [1],  
        'Usage Time (bins)': [1],
        'Region': [0],
        'Speciality': [0]
    })

    scaled_features = scaler.transform(input_features)
    predicted_labels = model.predict(scaled_features)

    # Simulate doctor filtering logic
    selected_doctors = data.sample(10)  # Test logic: Select random 10 doctors for now

    if selected_doctors.empty:
        return pd.DataFrame(), None

    output_file = "recommended_doctors.csv"
    selected_doctors[['NPI', 'State', 'Speciality', 'Region']].to_csv(output_file, index=False)

    return selected_doctors, output_file


# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_time = request.form.get('time')
        if input_time:
            doctors, file_path = predict_best_doctors(input_time)
            doctor_list = doctors[['NPI', 'State', 'Speciality', 'Region']].to_dict(orient='records')
            return render_template('index.html', doctor_list=doctor_list, file_path=file_path, input_time=input_time)
        else:
            return render_template('index.html', error="Please enter a valid time.")
    return render_template('index.html')

@app.route('/download')
def download_file():
    file_path = "recommended_doctors.csv"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found."

if __name__ == '__main__':
    app.run(debug=True)