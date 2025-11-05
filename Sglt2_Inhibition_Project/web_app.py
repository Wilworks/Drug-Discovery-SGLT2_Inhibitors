from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os

app = Flask(__name__)

# Load models and encoder
model_path = 'catboost_model.pkl'
encoder_path = 'label_encoder.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
else:
    label_encoder = None

def smiles_to_fingerprint(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            # Use MorganGenerator for modern RDKit API
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            fingerprint = generator.GetFingerprint(mol)
            return np.array(fingerprint)
        else:
            return None
    except:
        return None

def predict_activity(smiles, cid, sid):
    if model is None or label_encoder is None:
        return "Error: Model or encoder not loaded."

    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return "Error: Could not generate fingerprint from SMILES."

    fingerprint = fingerprint.reshape(1, -1)
    predicted_encoded = model.predict(fingerprint)[0]
    predicted_activity = label_encoder.inverse_transform([predicted_encoded])[0]

    return f"Prediction for Compound (CID: {cid}, SID: {sid}): {predicted_activity}"

@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SGLT2 Inhibition Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            form { margin-bottom: 20px; }
            input { margin: 5px; padding: 8px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>SGLT2 Inhibition Predictor</h1>
        <form id="predictionForm">
            <input type="text" id="cid" placeholder="PubChem CID" required><br>
            <input type="text" id="sid" placeholder="PubChem SID" required><br>
            <input type="text" id="smiles" placeholder="SMILES string" required><br>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const cid = document.getElementById('cid').value;
                const sid = document.getElementById('sid').value;
                const smiles = document.getElementById('smiles').value;

                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({cid: cid, sid: sid, smiles: smiles}),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerText = data.result;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('result').innerText = 'An error occurred.';
                });
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    cid = data.get('cid')
    sid = data.get('sid')
    smiles = data.get('smiles')

    if not cid or not sid or not smiles:
        return jsonify({'result': 'Error: Missing input data.'})

    try:
        cid = int(cid)
        sid = int(sid)
    except ValueError:
        return jsonify({'result': 'Error: CID and SID must be integers.'})

    result = predict_activity(smiles, cid, sid)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
