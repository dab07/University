import pandas as pd
import numpy as np

df = pd.read_csv("/Users/hs/Downloads/CSV File-1.csv")
target = np.array(df)[:, -1]
attribute = np.array(df)[:, :-1]

training_data = np.array(df)

specific_hypo = attribute[0]
general_hypo = [['?' for i in range(len(specific_hypo))] for i in range(len(specific_hypo))]

for k, input_vector in enumerate(attribute):
    if target[k] == 'yes':
        for i in range(len(specific_hypo)):
            if input_vector[i] != specific_hypo[i]:
                specific_hypo[i] = ['?']
                general_hypo[i][i] = ['?']
    elif target[k] == 'no':
        for i in range(len(specific_hypo)):
            if input_vector[i] != specific_hypo[i]:
                general_hypo[i][i] = specific_hypo[i]
            else:
                general_hypo[i][i] = ['?']

indices = [i for i, val in enumerate(general_hypo) if val == ['?', '?', '?', '?', '?', '?']]
for i in indices:
    general_hypo.remove(['?', '?', '?', '?', '?', '?'])

print("specific Hypothesis: ", specific_hypo)
print("general Hypothesis: ", general_hypo)



