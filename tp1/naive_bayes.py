import pandas as pd
import numpy as np

class NaiveBayesClassifier:

    def __init__(self):
        self.trained = False

    def train(self, df, group_column_name, apply_laplace=True):
        self.group_column_name = group_column_name
        # Columna que indica la clasificacion del registro
        group_column = df[group_column_name]
        
        # Clases para clasificacion. Ej: I, E
        groups = group_column.unique()

        # Cuantos registros hay para cada clase
        group_count = group_column.value_counts()
       
        group_frequency = group_count / len(df)
        
        # Agregacion por cada atributo
        agg = df.groupby(group_column_name).agg('sum')
        
        group_count.index.name = group_column_name

        # Frecuencias relativas (SIN LAPLACE)
        relative_freqs = agg.div(group_count, axis='index')

        # Frecuencias relativas (CON LAPLACE)
        # TODO: Cambiar el "+2" y el "+1" por el 'k' de la formula
        agg2 = agg + 1
        relative_freqs_laplace = agg2.div(group_count + 2, axis='index')

        self.trained = True
        self.relative_freqs = relative_freqs_laplace if apply_laplace else relative_freqs
        self.group_frequency = group_frequency

    def predict(self, df):
        assert self.trained, "Training was not performed"

        for index, row in df.iterrows():
            conditional_probabilities = self.relative_freqs * row + (1-row) * (1-self.relative_freqs)
            productorial = conditional_probabilities.product(axis=1)
            h = productorial * self.group_frequency

            df.at[index, self.group_column_name] = h.idxmax()
            for i,r in h.items():
                    df.at[index, "H({})".format(i)] = r

                        
        return df
