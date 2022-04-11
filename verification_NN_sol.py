import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
np.set_printoptions(precision=3)
from sklearn.metrics import r2_score
import seaborn as sns
import joblib
import itertools as it
sns.set_theme()
sns.set_palette("rocket")

NN_sol = tf.keras.models.load_model("data/model/model_4")
Scaler = joblib.load("data/model/scaler_4")

file = f"data/NN/data_10_0.csv"
data = np.genfromtxt(file, delimiter=",")[1:,1:]

x_train, x_test, y_train, y_test = train_test_split(data[:,:3], data[:,3:], random_state = 1, test_size = 0.2)

x_train_scaled = Scaler.fit_transform(x_train)
x_test_scaled = Scaler.transform(x_test)

y_test_predict = NN_sol.predict(x_test_scaled)

R2 = r2_score(y_test, y_test_predict)

fig, ax = plt.subplots(1,1)
ax.scatter(y_test_predict, y_test, c = "k", alpha = 0.6)
ax.set_xlabel("Network prediction values [AU]", fontsize=32)
ax.set_ylabel("True verification values [AU]", fontsize=32)

meanerror = np.mean(np.abs(y_test-y_test_predict))
varerror = np.var(np.abs(y_test-y_test_predict))

# cone = np.linspace(-np.pi/2, np.pi/2, 100)
# time_a = np.linspace(2.4588500e+06, 2.4661550e+06, 1000)
# time_v = np.linspace(0, 6*np.pi, 1000)
#
# grid = np.zeros(np.size(cone)*np.size(time_a)*np.size(time_v))
# perms = it.product(cone, time_a, time_v)
# for i,perm in enumerate(perms):
#     grid[i] = perm

omega = 2*np.pi/(365*24*3600)
au = 149597870.7E3
