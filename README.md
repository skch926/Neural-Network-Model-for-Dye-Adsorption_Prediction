Adsorption Capacity Prediction Using ANN
=======================================

This project uses an Artificial Neural Network (ANN) to predict Adsorption Capacity (%) using three experimental input parameters:

- Dye concentration (mg/L)
- Catalyst dosage (mg)
- Time (min)

The project includes model training, a saved ANN model, and an interactive Gradio web application for real-time predictions.

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

train_model.py       → Trains the ANN and saves the model  
app.py               → Runs Gradio web app for prediction  
all_datacsv.csv      → Dataset used for training  
adsorption_ann.h5    → Trained ANN model  
requirements.txt     → Dependencies  
README.txt           → Project documentation  

------------------------------------------------------------
HOW TO RUN THE PROJECT
------------------------------------------------------------

1. Install dependencies:
   pip install -r requirements.txt

2. Run the web app:
   python app.py
   A Gradio link will open in your browser for predictions.

3. (Optional) Retrain the model:
   python train_model.py

------------------------------------------------------------
MODEL DETAILS
------------------------------------------------------------

- Neural Network built using TensorFlow/Keras
- Architecture:
  Dense(32) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(1)
- Scaling: MinMaxScaler (range 0.1 to 0.9)
- Output: Adsorption capacity (%)
- Evaluation: R² score used to measure prediction accuracy

------------------------------------------------------------
WEB APP (GRADIO)
------------------------------------------------------------

The web interface contains:

- Slider for dye concentration
- Slider for catalyst dosage
- Slider for time
- Output box showing predicted adsorption capacity

This allows instant predictions without retraining the model.

------------------------------------------------------------
TECHNOLOGIES USED
------------------------------------------------------------

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Gradio

------------------------------------------------------------
APPLICATIONS
------------------------------------------------------------

- Photocatalytic dye removal prediction
- Wastewater treatment modeling
- Adsorption science & optimization
- Material and environmental research

------------------------------------------------------------
AUTHOR
------------------------------------------------------------

Sujeet Kumar  
If you like this project, please star the repository!
