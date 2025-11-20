import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("all datacsv.csv")

# Define X and y
X = df.drop(columns=['Adsorption capacity (%)'])
y = df['Adsorption capacity (%)']

# Normalize
X_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaled = X_scaler.fit_transform(X)

y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Build model
model = Sequential([
    Dense(32, activation='relu', input_dim=X.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(0.001), loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=150, batch_size=5,
                    validation_data=(X_test, y_test), verbose=1)

# Evaluate R²
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_actual = y_scaler.inverse_transform(y_test)
r2 = r2_score(y_test_actual, y_pred)
print("R²:", r2)

# Save model to the new directory
model.save("adsorption-ann-project/adsorption_ann.h5")

# Save loss plot to the new directory
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.savefig("adsorption-ann-project/training_loss.png")
plt.show()