from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extracting features from the JSON payload
        Age = data['Age']
        Irregular_Periods = data['Irregular_Periods']
        Excessive_Hair_Thinning = data['Excessive_Hair_Thinning']
        Oily_Skin = data['Oily_Skin']
        Weight_Gain = data['Weight_Gain']
        Dark_Skin_Patches = data['Dark_Skin_Patches']
        Pelvic_Pain = data['Pelvic_Pain']
        
        # Creating the input data array
        input_data = np.array([Age, Irregular_Periods, Excessive_Hair_Thinning, Oily_Skin, Weight_Gain, Dark_Skin_Patches, Pelvic_Pain]).reshape(1, -1)
        
        # Standardizing the input data
        std_data = scaler.transform(input_data)
        
        # Making a prediction
        prediction = model.predict(std_data)
        
        # Preparing the response
        result = "She has PCOD" if prediction[0] == 1 else "She doesn't have PCOD"
        response = {
            'prediction': int(prediction[0]),
            'result': result
        }
        
        return jsonify(response)
    
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
