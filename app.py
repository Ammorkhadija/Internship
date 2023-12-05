from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the machine learning model
with open('task6.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the index route
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input1 = float(input_data['input1'])
    input2 = float(input_data['input2'])
    input3 = float(input_data['input3'])
    input4 = float(input_data['input4'])

    prediction = model.predict([[input1, input2, input3, input4]])[0]

    class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    predicted_class = class_labels[int(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
