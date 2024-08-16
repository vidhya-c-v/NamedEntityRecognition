from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and other necessary components
with open('class.pkl', 'rb') as file:
    clf = pickle.load(file)

# List of predicted entities to filter
entity_types = ["B-DISEASE", "I-DISEASE", "B-CARDINAL", "B-GPE", "B-DATE", "I-DATE"]

# Define a function to perform predictions and extract entities
def extract_entities(sentences, predictions, entity_types):
    data = []
    for sentence, prediction in zip(sentences, predictions):
        entities = []
        for token in prediction:
            word, label = list(token.items())[0]
            if label in entity_types:
                entities.append((word, label))
        data.append({"Sentence": sentence, "Entities": entities})
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentences_input = request.form['sentences']
    
    # Split input into individual sentences
    sentences = [s.strip() for s in sentences_input.split('\n') if s.strip()]

    # Perform predictions on the input sentences
    prediction, _ = clf.predict(sentences)

    # Create a dataframe with filtered entity predictions
    df_predictions = extract_entities(sentences, prediction, entity_types)

    # Pass the predictions to the result page
    return render_template('result.html', predictions=df_predictions)

@app.route('/extracted_entities')
def extracted_entities():
    # Get the predictions from the session or redirect if not available
    predictions = request.args.get('predictions')
    if not predictions:
        return redirect(url_for('index'))

    # Parse the predictions from the URL parameter
    predictions = eval(predictions)

    # Render the extracted entities page with the predictions
    return render_template('extracted_entities.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
