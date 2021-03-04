import pandas as pd
import datetime
from colorama import Fore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split




# Read .dat file
df = pd.read_table("marambio/marambio_2007.dat", sep="\s+")

# print(df)

variables = df.columns.values
print("Variables:")
for v in variables:
    print(f'{Fore.BLUE}{v}')


#date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

#df.hist()
print(df.median())
print(df)
df.boxplot()
plt.savefig("prueba_ML")

train, test = train_test_split(df, test_size=0.2)

print(train.shape)
print(test)


