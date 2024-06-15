import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

def scale_new_data(data, features_to_scale, min_train_values, max_train_values):
    """
    Scales specific features in the data using min-max scaling.

    Args:
        data (list): The list of features for a new data point.
        features_to_scale (list): A list of feature indices to scale.
        min_train_values (list): A list of minimum values for each feature to scale (corresponding to features_to_scale).
        max_train_values (list): A list of maximum values for each feature to scale (corresponding to features_to_scale).

    Returns:
        list: The scaled data with specific features scaled.
    """

    scaled_data = data.copy()
    for i, feature_index in enumerate(features_to_scale):
        scaled_data[feature_index] = (data[feature_index] - min_train_values[i]) / (max_train_values[i] - min_train_values[i])
    return scaled_data

def rescale_prediction(prediction, min_train_value, max_train_value):
    """
    Rescales the predicted value back to the original scale using min-max scaling.

    Args:
        prediction (float): The predicted value in scaled space.
        min_train_value (float): The minimum value of the target variable in training data.
        max_train_value (float): The maximum value of the target variable in training data.

    Returns:
        float: The rescaled predicted value in the original scale.
    """

    return prediction * (max_train_value - min_train_value) + min_train_value

app = Flask(__name__)
model = pickle.load(open('Salary.pkl', 'rb'))

features_to_scale = ['AGE','PAST EXP','Days Worked']  # Replace with feature names to scale
min_train_values = [21.0, 0.0, 341.0]  # Replace with minimum values for corresponding features
max_train_values = [45.0, 23.0, 2540.0]  # Replace with maximum values for corresponding features
min_train_salary = 40001.0  # Minimum value of salary in training data
max_train_salary = 388112.0  # Maximum value of salary in training data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    
    features_to_scale_indices = [i for i, feature in enumerate(int_features) if feature in features_to_scale]
    # Scale specific features
    scaled_features = scale_new_data(int_features, features_to_scale_indices, min_train_values, max_train_values)
    
    # Prepare final features (considering scaling and non-scaled features)
    final_features = np.array([scaled_features]).tolist()
    for i, feature in enumerate(int_features):
        if i not in features_to_scale_indices:
            final_features[0][i] = feature

    prediction = model.predict(final_features)
    output = rescale_prediction(prediction[0], min_train_salary, max_train_salary)
    output = round(output, 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data_list = list(data.values())
    
    features_to_scale_indices = [i for i, feature_name in enumerate(features_to_scale) if feature_name in data.keys()]
    
    # Scale specific features
    scaled_data = scale_new_data(data_list, features_to_scale_indices, min_train_values, max_train_values)
    
    # Prepare final features (considering scaling and non-scaled features)
    for i, feature in enumerate(data_list):
        if i not in features_to_scale_indices:
            scaled_data[i] = feature

    # Make prediction using the scaled data
    prediction = model.predict([np.array(scaled_data)])
    output = rescale_prediction(prediction[0], min_train_salary, max_train_salary)
    output = round(output, 2)
    
    # Return the prediction as JSON
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

