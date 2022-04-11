import numpy as np
import pandas as pd

file = f"data/NN/data_0.csv"
# file = r"data\emp_ast\2021_QM1_xyPlane.dat"

stop = False
rows = 6E7
i = 0
step = 10
data_new = np.zeros((1,4))
while not stop:
    data = np.genfromtxt(file, delimiter=",", skip_header=int(1+i*rows), max_rows=rows)[:,1:]

    if i == 0 and not stop:
        data_new = data[np.arange(0, np.size(data[:, 0]), step)]
    elif i != 0 and not stop:
        data_new = np.vstack((data_new, data[np.arange(0, np.size(data[:, 0]), step)]))

    stop = np.shape(data)[0] != rows
    print(i)
    i += 1


data_new = pd.DataFrame(data_new)
# data_new.to_csv(f"data/NN/data_{step}_999.csv")
