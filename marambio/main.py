import pandas as pd
from colorama import Fore

# Read .dat file
df = pd.read_table("marambio/marambio_2007.dat", sep="\s+")

# print(df)

variables = df.columns.values
print("Variables:")
for v in variables:
    print(f'{Fore.BLUE}{v}')