# ----------------------------
# Red Neuronal Convolucional Multicapa (CNN) - CIFAR-10
# ----------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Cargar y preprocesar los datos
# ----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizar las imágenes (de 0-255 a 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Ver dimensiones
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# ----------------------------
# Definir la arquitectura de la CNN
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases
])

# ----------------------------
# Compilar el modelo
# ----------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# Entrenar el modelo
# ----------------------------
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# ----------------------------
# Evaluar el modelo
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión en el conjunto de prueba:", test_acc)

# ----------------------------
# Graficar precisión del entrenamiento
# ----------------------------
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión del modelo')
plt.legend()
plt.grid(True)
plt.show()
