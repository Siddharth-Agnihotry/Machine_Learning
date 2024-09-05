# AL_Internship.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load data and train model
df = pd.read_csv(r"C:\Users\Siddharth Agnihotry\Downloads\traffic_test.csv")

# Convert date_time to datetime
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['month_day'] = df['date_time'].dt.day
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year

# Define features and target
features = ['is_holiday', 'temperature', 'weather_type', 'month_day', 'weekday', 'month', 'year']
target = 'traffic_volume'

X = df[features]
y = df[target]

# Preprocessing pipelines
numeric_features = ['temperature']
categorical_features = ['is_holiday', 'weather_type', 'month_day', 'weekday', 'month', 'year']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = pd.DataFrame([data])
    
    # Predict
    predicted_volume = model.predict(user_input)[0]
    
    # Log the input and predicted volume
    print(f"User Input: {data}")
    print(f"Predicted Volume: {predicted_volume}")
    
    # Set threshold for traffic volume
    threshold = 4000  # Example threshold value
    result = 'yes' if predicted_volume > threshold else 'no'
    
    return jsonify({'prediction': result, 'traffic_volume': predicted_volume})

if __name__ == '__main__':
    app.run(debug=True)