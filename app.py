from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
model = pickle.load(open('model/regressor.pkl', 'rb'))

# Load each label encoder
fuel_encoder = pickle.load(open('model/label_encoder_3.pkl', 'rb'))
seller_type_encoder = pickle.load(open('model/label_encoder_4.pkl', 'rb'))
transmission_encoder = pickle.load(open('model/label_encoder_5.pkl', 'rb'))
owner_encoder = pickle.load(open('model/label_encoder_6.pkl', 'rb'))

# Load the dataset for finding similar cars
df = pd.read_csv('car_data.csv')  # Replace with the path to your dataset

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']

        # Encode the categorical features
        fuel_type_encoded = fuel_encoder.transform([fuel_type])[0]
        seller_type_encoded = seller_type_encoder.transform([seller_type])[0]
        transmission_encoded = transmission_encoder.transform([transmission])[0]
        owner_encoded = owner_encoder.transform([owner])[0]

        # Prepare the data for prediction
        # Include all 7 features, assuming 'name' is not used in prediction
        input_data = np.array([[year, km_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner_encoded, 0]])

        # Predict the price
        prediction = model.predict(input_data)[0]

        # Render the result template with the prediction
        return render_template('index.html', prediction_text=f'Estimated Car Price: ৳{prediction:.2f}')

    except ValueError as e:
        return f"Error in prediction: {e.args[0]}", 400
    except KeyError as e:
        return f"Missing form field: {e.args[0]}", 400
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
