from naive_bayes import NaiveBayesClassifier
import pandas as pd

train_data = pd.read_excel("data/PreferenciasBritanicos.xlsx")
nb = NaiveBayesClassifier()

test_data = [
    [1,0,1,1,0], 
    [0,1,1,0,1]
]
test_data = pd.DataFrame(test_data, columns=["scones", "cerveza", "wiskey","avena","futbol"])


print("Train data\n", train_data)
print("Test data\n", test_data)

nb.train(train_data, "Nacionalidad")
prediction = nb.predict(test_data)

print("Prediction\n", prediction)