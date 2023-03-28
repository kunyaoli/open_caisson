from matplotlib import pyplot as plt
import pandas as pd

pred = pd.read_csv("../Mlknn/result_pred.csv")
true = pd.read_csv("../Mlknn/result_true.csv")


x = [i for i in range(38)]

plt.scatter(x, pred.iloc[68,])
plt.scatter(x, true.iloc[68,1:])
plt.show()