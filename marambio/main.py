import pandas as pd
import datetime
from colorama import Fore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split




# Read .dat file
df = pd.read_table("./marambio_2007.dat", sep=r"\s+")

# print(df)

variables = df.columns.values
# print(variables)
print("Variables:")
for v in variables:
    print(f'{Fore.BLUE}{v}')

# Minima y maxima fecha
# Este codigo ignora el year, asi que si se incluyen en el dataset habria q cambiarlo
dates = pd.to_datetime(df.index,format="%m%d")
minDate = min(dates)
maxDate = max(dates)
print("Date from: {} to {}".format(minDate.strftime("%d/%m"), maxDate.strftime("%d/%m")))

# Mediana
print(f'{Fore.RED}{df.median()}')

# Generacion de histogramas:
for v in variables:
    df.hist(v)
    plt.savefig("images/hist_{}.png".format(v))
    plt.clf()

# Generacion de boxplot
plt.clf()
df.boxplot()
plt.savefig("images/boxplot.png")


#  TODO: Ver como mostramos la division de test y train en el ppt
train, test = train_test_split(df, test_size=0.2)
# print(train.shape)
# print(test)


