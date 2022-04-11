import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
np.set_printoptions(precision=3)
from sklearn.metrics import r2_score
import seaborn as sns
import joblib
sns.set_theme()
sns.set_palette("rocket")

# data retrieval
file = f"data/NN/data_5_1.csv"
data = np.genfromtxt(file, delimiter=",")[1:,1:]

# data formatting
x_train, x_test, y_train, y_test = train_test_split(data[:,:3], data[:,3:], random_state = 1, test_size = 0.2)

Scaler = StandardScaler()
x_train_scaled = Scaler.fit_transform(x_train)
x_test_scaled = Scaler.transform(x_test)

# network
NN = tf.keras.models.Sequential()
NN.add(tf.keras.layers.Dense(np.size(x_train_scaled[0]),
                                          activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(300, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(300, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(300, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(300, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(300, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.add(tf.keras.layers.Dense(np.size(y_train[0]),
                                          activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN.compile(optimizer="adam",
           loss="MeanSquaredError",
           metrics=["mean_absolute_error"])

# train
hist = NN.fit(x_train_scaled, y_train, epochs=20, validation_split = 0.1,
              batch_size = 4000)

y_test_predict = NN.predict(x_test_scaled)
errors = y_test - y_test_predict
# NN.save("data/model/model_999")
# joblib.dump(Scaler, "data/model/scaler_999")