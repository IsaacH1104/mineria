import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Suponiendo que tienes imágenes de microestructura y mapas de esfuerzo correspondientes
# X: imágenes de entrada (n_samples, 32, 32, 1)
# Y: mapas de esfuerzo (n_samples, 32, 32, 1)
# Aquí usaremos datos aleatorios para simularlo
X = np.random.rand(5321, 32, 32, 1).astype(np.float32)
Y = np.random.rand(5321, 32, 32, 1).astype(np.float32)

# División de los datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalización
X_mean, X_std = X_train.mean(), X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Arquitectura de la red basada en el paper
def build_model():
    inputs = tf.keras.Input(shape=(32, 32, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 8x8

    # Residual blocks con squeeze-excitation simplificados
    for _ in range(5):
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)  # 16x16
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)  # 32x32
    outputs = layers.Conv2D(1, (3, 3), padding='same')(x)

    return models.Model(inputs, outputs)

model = build_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Entrenamiento
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

# Evaluación
loss, mae = model.evaluate(X_test, Y_test)
print(f"MAE: {mae}")

# Predicciones
preds = model.predict(X_test[:5])

# Visualización de resultados
for i in range(5):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Microestructura")
    plt.imshow(X_test[i].squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Esfuerzo Real")
    plt.imshow(Y_test[i].squeeze(), cmap='viridis')

    plt.subplot(1, 3, 3)
    plt.title("Predicción CNN")
    plt.imshow(preds[i].squeeze(), cmap='viridis')
    plt.show()
