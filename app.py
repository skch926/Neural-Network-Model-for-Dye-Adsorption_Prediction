
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

df = pd.read_csv("all_datacsv.csv")

X = df.drop(columns=["Adsorption capacity (%)"])
y = df["Adsorption capacity (%)"]

X_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaler.fit_transform(X)

y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
y_scaler.fit_transform(y.values.reshape(-1, 1))

model = tf.keras.models.load_model(
    "adsorption_ann.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)

def predict_adsorption(dye, catalyst, time):
    arr = np.array([[dye, catalyst, time]])
    scaled = X_scaler.transform(arr)
    pred_scaled = model.predict(scaled)
    pred = y_scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])

interface = gr.Interface(
    fn=predict_adsorption,
    inputs=[
        gr.Slider(20, 50, step=1, label="Dye Concentration (mg/L)"),
        gr.Slider(15, 30, step=1, label="Catalyst Dosage (mg)"),
        gr.Slider(10, 60, step=1, label="Time (min)")
    ],
    outputs=gr.Number(label="Adsorption Capacity (%)"),
    title="Adsorption Predictor"
)

if __name__ == "__main__":
    interface.launch()
