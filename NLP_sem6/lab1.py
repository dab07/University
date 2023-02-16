import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/hs/Downloads/archive (1)/Employee_Salary_Dataset.csv")
df.sort_values(by=['Age'], ascending=True)

department_1 = np.array(df)[0:8]
department_2 = np.array(df)[9:16]
department_3 = np.array(df)[16:24]
department_4 = np.array(df)[24:32]
# SALARY
mean1 = np.mean(department_1[:, -1])
mean2 = np.mean(department_2[:, -1])
mean3 = np.mean(department_3[:, -1])
mean4 = np.mean(department_4[:, -1])
print("Average salary in dep1 = ", mean1)
print("Average salary in dep2 = ", mean2)
print("Average salary in dep3 = ", mean3)
print("Average salary in dep4 = ", mean4)

ma = df.iloc[0,-1]
mi = ma
sum = 0
for i in range(0, 32):
    ma = max(ma, df.iloc[i, -1])
    mi = min(mi, df.iloc[i, -1])
    sum = sum + df.iloc[i, -1]

print("Maximum Salary: ", ma)
print("Manimum Salary: ", mi)
print("Average Salary: ", sum / 32)

plt.plot(department_1[:,-1], color="red")
plt.plot(department_2[:,-1], color="blue")
plt.plot(department_3[:,-1], color="green")
plt.plot(department_4[:,-1], color="orange")
plt.show()
