# Machine Learning-Based QSAR Modelling of SGLT2 Inhibitors

## Overview

This project implements a comprehensive machine learning framework for predicting the inhibitory activity of Sodium-Glucose Cotransporter 2 (SGLT2) compounds using molecular fingerprints and explainable AI. The system includes a Django web application for interactive predictions, standalone inference scripts, and detailed model evaluation.

## Features

- **Molecular Fingerprint Generation**: ECFP4 fingerprints from SMILES strings using RDKit
- **Multiple ML Models**: Random Forest, Gradient Boosting, XGBoost, CatBoost, and SVM
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Explainability**: SHAP analysis for feature importance and model interpretability
- **Web Interface**: Django-based application with user authentication
- **Interactive Prediction**: Web forms and command-line tools for compound screening

## Project Structure

```
Sglt2_Inhibition_Project/
├── index.html                    # Landing page with project overview
├── .gitignore                    # Git ignore file
├── README.md                     # This file
├── projec.md                     # Detailed project report
├── pipeline.py                   # Complete ML pipeline (Colab notebook)
├── Inference.py                  # Command-line prediction tool
├── web_app.py                    # Standalone Flask web app
├── manage.py                     # Django management script
├── db.sqlite3                    # SQLite database
├── Wilfred.xlsx                 # Original dataset
├── sglt2_project/               # Django project directory
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── predictor/                    # Django app directory
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   ├── migrations/
│   ├── static/
│   │   └── predictor/
│   │       └── WhatsApp Image 2025-11-05 at 02.54.04_4641df7a.jpg
│   └── templates/
│       └── predictor/
│           ├── landing.html
│           ├── about.html
│           ├── login.html
│           └── predict.html
├── trained_models_and_plots/     # Model files and plots (if saved)
├── *.pkl                         # Trained model files
├── *.png                         # Evaluation plots
└── requirements.txt              # Python dependencies (to be created)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Wilworks/Drug-Discovery-SGLT2_Inhibitors.git
   cd Drug-Discovery-SGLT2_Inhibitors
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Django migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Create superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

## Usage

### Django Web Application

1. **Start the server**:
   ```bash
   python manage.py runserver
   ```

2. **Access the application**:
   - Landing page: http://127.0.0.1:8000/
   - About page: http://127.0.0.1:8000/about/
   - Login: http://127.0.0.1:8000/login/
   - Prediction tool: http://127.0.0.1:8000/predict/

### Standalone Flask App

```bash
python web_app.py
```

Access at: http://127.0.0.1:5000/

### Command-Line Inference

```bash
python Inference.py
```

Follow the prompts to enter CID, SID, and SMILES for prediction.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.8846 | 0.8400 | 0.9130 | 0.8750 | 0.9799 |
| SVM | 0.8846 | 0.8400 | 0.9130 | 0.8750 | 0.9775 |
| CatBoost | 0.8750 | 0.8367 | 0.8913 | 0.8632 | 0.9779 |
| Gradient Boosting | 0.8750 | 0.8367 | 0.8913 | 0.8632 | 0.9781 |
| Random Forest | 0.8654 | 0.8200 | 0.8913 | 0.8542 | 0.9642 |

## Dependencies

- Python 3.8+
- Django 5.2+
- RDKit
- scikit-learn
- XGBoost
- CatBoost
- SHAP
- Flask
- pandas
- numpy
- matplotlib
- seaborn

## Dataset

The project uses molecular data from PubChem, processed to include:
- SMILES strings for molecular representation
- Activity outcomes (Active/Inactive)
- ECFP4 fingerprints (2048 bits) as features

## Authors

- **Asumboya Wilfred Ayine**
  - Biomedical Engineering Student, Level 300
  - University of Ghana
  - Internship: AI and ML with Institute of Applied Sciences and Technology

## Supervisors

- **Nunana Kingsley** (Tutor)
- **Prof. Samuel Kwofie** (Supervisor)

## License

This project is for academic and research purposes.

## Acknowledgments

- PubChem database for molecular data
- RDKit for cheminformatics tools
- scikit-learn, XGBoost, CatBoost for ML implementations
- SHAP for explainable AI
