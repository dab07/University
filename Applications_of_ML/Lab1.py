import numpy as np
import pandas as pd

dataset1 = pd.read_csv("/Users/hs/Downloads/student-mat-pass-or-fail.csv")
dataset2 = pd.read_csv("/Users/hs/Downloads/CSV File-1.csv")

target_1 = np.array(dataset1.iloc[:, 29:30]).flatten()
attribute_1 = np.array(dataset1.iloc[:, :29])

target = np.array(dataset2)[:, -1]
attribute = np.array(dataset2)[:, :-1]
print("attributes\n", attribute)
print("\ntarget\n", target)


def Find_S(att, tar):
    for i, j in enumerate(tar):
        if j == "yes" or j == 1:
            hypo = att[i].copy()
            break

    for i, j in enumerate(att):
        if tar[i] == "yes" or tar[i] == 1:
            for x in range(len(hypo)):
                if j[x] != hypo[x]:
                    hypo[x] = '?'
                else:
                    pass

    return hypo

def Find_S1(att, tar):
    for i, j in enumerate(tar):
        if j == "yes" or j == 1:
            hypo = att[i].copy()
            break

    for i, j in enumerate(att):
        if tar[i] == "yes" or tar[i] == 1:
            for x in range(len(hypo)):
                if j[x] != hypo[x]:
                    hypo[x] = -1
                else:
                    pass

    return hypo
print(Find_S(attribute, target))
print(Find_S1(attribute_1, target_1))