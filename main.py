import tensorflow as tf
from tensorflow import keras
import argparse
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", help="Plik z obrazkiem cyfry")
args = parser.parse_args()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(10, activation="softmax"),
#     ]
# )
# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# model.fit(x_train, y_train, epochs=5)  # użyj verbose=0 jeśli jest problem z konsolą
# model.evaluate(x_test, y_test)

# model.save("lab1-model.keras")

new_model = tf.keras.models.load_model("lab1-model.keras")

# Show the model architecture
new_model.summary()


# Krzywa uczenia
history = new_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Rysowanie krzywej uczenia
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.title('Krzywa uczenia - strata')
plt.show()

plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.title('Krzywa uczenia - dokładność')
plt.show()
