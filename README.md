disease-prediction-ml/
├── data/
│   ├── raw/             ← Original Kaggle CSV files (never modify)
│   └── processed/       ← Cleaned, encoded datasets
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   ← Cleaning & feature engineering
│   └── 03_model_training.ipynb  ← Training + evaluation + comparison
├── models/
│   ├── model.pkl        ← Best trained model (Voting Ensemble)
│   ├── label_encoder.pkl
│   └── model_metrics.json  ← Accuracy scores for all models
├── app/
│   ├── app.py           ← Main Streamlit application
│   ├── predict.py       ← Prediction logic (loads model, returns output)
│   └── utils.py         ← Helper functions (symptom list, etc.)
├── deployment/
│   ├── requirements.txt
│   └── README.md        ← Hugging Face Spaces setup guide
├── reports/
│   ├── architecture_diagram.png
│   ├── confusion_matrix.png
│   ├── model_comparison_chart.png
│   └── final_report.pdf
└── README.md            
