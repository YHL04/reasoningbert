import pandas as pd
import matplotlib.pyplot as plt


data1 = pd.read_csv("logs/bert", names=["Loss", "Predicted Loss", "Real Loss"])
data2 = pd.read_csv("logs/block_bert", names=["Loss", "Predicted Loss", "Real Loss"])

plt.subplot(2, 1, 1)

plt.plot(data1["Loss"], label="bert")
plt.plot(data1["Loss"].rolling(500).mean(), label="bert")
# plt.plot(data2["Loss"], label="block_bert")

plt.yscale("log")

leg = plt.legend(loc="upper right")

plt.subplot(2, 1, 2)

plt.plot(data1["Predicted Loss"])
plt.plot(data1["Real Loss"])

plt.yscale("log")

plt.show()
