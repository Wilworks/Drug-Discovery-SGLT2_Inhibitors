from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os

# Load the trained CatBoost model and the label encoder
# Assuming the CatBoost model was chosen as it worked for SHAP calculations
save_dir = 'trained_models_and_plots'
cat_model = joblib.load(os.path.join(save_dir, 'catboost_model.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Initialize the SHAP explainer for CatBoost with training data
# Assuming X_train is available from previous successful executions
explainer_cat = shap.TreeExplainer(cat_model, data=X_train)


def smiles_to_fingerprint(smiles_string):
    """Converts a SMILES string to an RDKit molecular fingerprint."""
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

def predict_activity(fingerprint, model, label_encoder):
    """
    Predicts the activity of a compound based on its molecular fingerprint.

    Args:
        fingerprint (np.ndarray): The fingerprint of a single molecule, shaped as (1, n_features).
        model: The trained machine learning model.
        label_encoder: The LabelEncoder used to encode the target variable.

    Returns:
        str: The predicted activity outcome ('Active' or 'Inactive').
    """
    # Ensure the fingerprint is a 2D array for prediction
    if fingerprint.ndim == 1:
        fingerprint = fingerprint.reshape(1, -1)

    # Predict the encoded label
    predicted_encoded_label = model.predict(fingerprint)[0]

    # Decode the predicted label back to the original string
    predicted_activity = label_encoder.inverse_transform([predicted_encoded_label])[0]

    return predicted_activity

def calculate_single_shap(explainer, fingerprint):
    """
    Calculates SHAP values for a single input instance.

    Args:
        explainer: The SHAP explainer for the model.
        fingerprint (np.ndarray): The fingerprint of a single molecule, shaped as (1, n_features).

    Returns:
        np.ndarray: The calculated SHAP values for the single instance.
    """
    # Ensure the fingerprint is a 2D array with shape (1, n_features)
    if fingerprint.ndim == 1:
        fingerprint = fingerprint.reshape(1, -1)
    elif fingerprint.ndim > 2 or fingerprint.shape[0] != 1:
        raise ValueError("Fingerprint must be a 1D array or a 2D array with shape (1, n_features).")

    # Calculate SHAP values
    shap_values = explainer.shap_values(fingerprint)

    # Handle different explainer output formats
    if isinstance(explainer, shap.TreeExplainer) and isinstance(shap_values, list) and len(shap_values) > 1:
        # For tree explainers with multiple classes, return values for the positive class (class 1)
        return np.array(shap_values[1])
    else:
        # For other explainers or binary tree explainers, return the single array
        return np.array(shap_values)

def plot_shap_values(top_features: pd.Series, title: str):
    """
    Visualizes the top features and their SHAP values as colored progress bars.

    Args:
        top_features (pd.Series): A Series containing feature names and their SHAP values.
        title (str): The title for the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Sort features by absolute SHAP value for consistent plotting order
    top_features = top_features.sort_values(key=abs, ascending=True)

    features = top_features.index
    shap_values = top_features.values

    colors = ['green' if val > 0 else 'red' for val in shap_values]

    ax.barh(features, shap_values, color=colors)
    ax.set_title(title)
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('Features')

    plt.tight_layout()
    plt.show()


# Main interactive loop
while True:
    try:
        # 1. Interactive input prompting
        cid_input = input("Enter PubChem CID: ")
        sid_input = input("Enter PubChem SID: ")
        smiles = input("Enter SMILES string: ")

        # Convert CID and SID to integers with error handling
        cid = int(cid_input)
        sid = int(sid_input)

        # 2. Generate molecular fingerprint
        fingerprint = smiles_to_fingerprint(smiles)

        if fingerprint is None:
            print("Error: Could not generate fingerprint from SMILES. Please enter a valid SMILES string.")
            continue # Continue the loop for invalid SMILES

        # 3. Predict activity
        predicted_activity = predict_activity(fingerprint, cat_model, label_encoder)

        # 4. Print prediction
        print(f"\nPrediction for Compound (CID: {cid}, SID: {sid}): {predicted_activity}")

        # 5. Calculate SHAP values for the single instance
        single_shap_values = calculate_single_shap(explainer_cat, fingerprint)

        # 6. Calculate absolute SHAP values
        single_shap_values_abs = np.abs(single_shap_values)

        # 7. Create a pandas Series from the absolute SHAP values
        feature_indices = [f'feature_{i}' for i in range(single_shap_values_abs.shape[-1])]
        single_shap_series = pd.Series(single_shap_values_abs[0], index=feature_indices)

        # 8. Sort the Series in descending order and select the top N
        top_n = 20
        top_features_single_instance = single_shap_series.sort_values(ascending=False).head(top_n)

        # 9. Visualize the top features and their SHAP values
        plot_shap_values(top_features_single_instance, f'Top {top_n} Feature Contributions to Prediction (CID: {cid}, SID: {sid})')

        # 10. Ask user if they want to analyze another compound
        another = input("Analyze another compound? (yes/no): ").lower()
        if another != 'yes':
            break # Break the loop if the user does not enter 'yes'

    except ValueError:
        print("Invalid input for CID or SID. Please enter integer values.")
        continue # Continue the loop for invalid CID/SID
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        continue

print("Inference script finished.")