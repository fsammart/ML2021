import math
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt

from plots import plot_confusion_matrix

# This mode computes only if the word appeared in the news or not
APPEARANCES = 0
# This mode computes how many times the word appeared for that category
QUANTITY = 1

CROSSED_VALIDATION = 0
BOOTSTRAP = 1
RANDOM = 2


class Category:

    def __init__(self, name, mode=APPEARANCES):
        self.name = name
        self.mode = mode
        self.count = 0
        self.probability = 0.0  # prior probability of category
        self.words = {}
        self.relative_frequencies = {}  # will be used to identify category of test news
        self.non_existing_word_probability = 0 # will be used in case word is not in dict

    # Receives headline of news and tokenizes it.
    def add_news(self, headline):
        self.count += 1
        tokens = headline.lower().split()
        distinct_tokens = list(set(tokens))

        if self.mode == APPEARANCES:
            processed = {token:1 for token in distinct_tokens}
        else:
            processed = {token: tokens.count(token) for token in distinct_tokens}

        [self.add_token(token, quantity) for token, quantity in processed.items()]
    
    # token MUST BE lowercase
    def add_token(self, token, quantity):
        if token not in self.words.keys():
            self.words[token] = 0
        self.words[token] = self.words.get(token) + quantity

    def get_rf(self, quantity):
        total = sum(self.words.values())

        if self.mode == APPEARANCES:
            # 2 stands for "appears in headline vs. does not appear in headline"
            return (quantity + 1) / (total + 2)

        # This in case mode == QUANTITY
        k = len(self.words.keys())
        return (quantity + 1) / (total + k)

    def compute_rf_for_word(self, word, quantity):
        self.relative_frequencies[word] = self.get_rf(quantity)

    def conclude_learning(self, total_registers):
        # 1. We want to calculate the probability of the category
        self.probability = self.count / total_registers

        # 2. We want to compute relative frequencies for the words that appeared
        [self.compute_rf_for_word(word, quantity) for word, quantity in self.words.items()]

        # 3. Computes non existing word probability for words in test headlines that are
        # not in dictionary
        self.non_existing_word_probability = self.get_rf(0)

    # TODO: replace this when deciding what to do with probabilities
    def get_prod(self, headline):
        tokens = headline.lower().split()
        distinct_tokens = list(set(tokens))

        prod = 1

        if self.mode == APPEARANCES:
            not_in_category = 0

            for word in distinct_tokens:
                probability = self.relative_frequencies.get(word) if word in self.relative_frequencies.keys() else 1
                if probability == 1: not_in_category += 1
                prod = prod * probability

            # words that are not in this category
            prod = prod * math.pow(
                self.non_existing_word_probability, not_in_category
            )

        # If self.mode === QUANTITY
        else:
            for token in tokens:
                prod = prod * self.non_existing_word_probability if token not in self.relative_frequencies.keys() \
                    else self.relative_frequencies.get(token)

        return prod * self.probability

    def __str__(self):
        return f'{self.name} - P(x=self)={self.probability}'


class Bayes:

    def __init__(self, mode=APPEARANCES):
        # General
        self.categories = {}
        self.mode = mode

        # Metrics
        self.confusion_matrix = None
        self.roc_data = None

    def process(self, headline, category):
        if category not in self.categories.keys():
            self.categories[category] = Category(category, mode=self.mode)
        self.categories[category].add_news(headline)

    def process_data(self, dataframe):
        # Processes news and will dynamically add categories to dictionary
        [self.process(headline, category) for headline, category in
         zip(dataframe['titular'], dataframe['categoria'])]
        print(f"Finished processing {len(dataframe.index)} news.")

    def conclude_learning(self):
        [category.conclude_learning(len(argentine_news.index)) for category in self.categories.values()]
        print("Naive bayes algorithm learning concluded.")

    def test(self, headline, real_category):
        prod_results = {name: category.get_prod(headline) for name, category in self.categories.items()}

        winner = max(prod_results.items(), key=operator.itemgetter(1))[0]

        probability_of_register = sum(prod_results.values())
        probability_results = {name: prod/probability_of_register for name, prod in prod_results.items()}

        # both prod results and probability results are dictionary
        return prod_results, probability_results, winner, winner == real_category

    def test_batch(self, test_df):
        hits = 0
        verbose = ''

        # key is real category and values has columns with results
        self.confusion_matrix = {
            category: {category: 0 for category in self.categories.keys()}
            for category in self.categories.keys()
        }

        self.roc_data = []

        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            headline = row.titular
            real_category = row.categoria

            results, probabilities, winner, success = self.test(headline, real_category)
            self.confusion_matrix[real_category][winner] = self.confusion_matrix.get(real_category).get(winner) + 1

            probabilities["real"] = row.categoria
            self.roc_data.append(probabilities)


            hits += 1 if success else 0
            verbose += f'** Winner: {winner}, Real: {real_category} **\n'
        return hits, verbose

    
    def __str__(self):
        desc = ''
        for name, category in self.categories.items():
            desc += f'{name}: {category.probability}\n'
        return desc


def split_dataset(dataframe, mode=RANDOM, percentage=0.8):
    train_df = None
    test_df = None
    if mode == CROSSED_VALIDATION:
        pass  # TODO: implement
    elif mode == BOOTSTRAP:
        pass  # TODO: implement
    else:
        msk = np.random.rand(len(dataframe)) < percentage
        train_df = dataframe[msk]
        test_df = dataframe[~msk]

    return train_df, test_df

def get_metrics(confusion_matrix):
    metrics = {k:{} for k in confusion_matrix.keys()}

    for metric in metrics.keys():
        tp = 0
        fp = 0
        tn = 0
        fn = 0


        for real,row in confusion_matrix.items():
            for predicted, value in row.items():
                if real == predicted == metric:
                    tp += value
                elif predicted == metric and real != metric: # se predijo true pero el real es distinto
                    fp += value
                elif real == metric and predicted != metric: #se predijo false pero era true
                    fn += value
                elif real != metric and predicted != metric: # se predijo false y era false
                    tn += value
 
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)

        f1 = (2*precision*recall) / (precision + recall)
        tvp = tp /  (tp+fn)
        tfp = fp / (fp + tn)

        metrics[metric] = {
            "acc": accuracy,
            "prc": precision,
            "rec": recall,
            "f1": f1,
            "tvp": tvp,
            "tfp": tfp,

            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }

    avgAcc = 0
    avgPrc = 0
    avgRec = 0
    avgF1 = 0
    avgTvp = 0
    avgTfp = 0
    for k,v in metrics.items():
        avgAcc += v["acc"]  / len(metrics.items())
        avgPrc += v["prc"] / len(metrics.items())
        avgRec += v["rec"] / len(metrics.items())
        avgF1  += v["f1"] / len(metrics.items())
        avgTvp += v["tvp"] / len(metrics.items())
        avgTfp += v["tfp"] / len(metrics.items())

    metrics["Promedio"]  = {
        "acc": avgAcc,
        "prc": avgPrc,
        "rec": avgRec,
        "f1": avgF1,
        "tvp": avgTvp,
        "tfp": avgTfp,
        "tp": "",
        "tn": "",
        "fp": "",
        "fn": ""

    }

    return metrics


def print_metrics(metrics):
    cols = [x for x in metrics[next(iter(metrics))].keys()]

    max_category_size = max(len(x) for x in metrics.keys())
    max_header_size = max(len(x) for x in cols)

    print("".ljust(max_category_size + 5), end="")
    for c in cols:
        print("\t", c.ljust(max_header_size),  end="")
    print()
    
    for cat,cat_metrics in metrics.items():
        if cat != "Promedio":
            print(cat.ljust(max_category_size), "\t", end="")
            for metric in cols:
                if(type(cat_metrics[metric]) == float):
                    print("{:.3f}".format(cat_metrics[metric]).ljust(max_header_size), "\t", end='')
                else:
                    print(str(cat_metrics[metric]).ljust(max_header_size), "\t", end='')
        print()
    print()
    print("Promedio".ljust(max_category_size), "\t", end="")
    for metric in cols:
        if(type(cat_metrics[metric]) == float):
            print("{:.3f}".format(metrics["Promedio"][metric]).ljust(max_header_size), "\t", end='')
        else:
            print(str(metrics["Promedio"][metric]).ljust(max_header_size), "\t", end='')
    print()

def plot_roc(roc_data):
    categories = list(roc_data[0].keys())
    categories.remove("real")
    

    # Cada categoria va a ser un array donde el valor 0 corresponde a un distinto umbral
    # posicion 0 => u = 0.1, posicion 1 => u = 0.2 .... posicion 8 => u = 0.9
    roc_curve = {
        x: {
            "tfp": [],
            "tvp": []
        } for x in categories
    }

    for category in categories:
        for u in range(0, 11):
            u = u / 10
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for entry in roc_data:
                if entry["real"] == category:
                    if entry[category] > u: # es la categoria y predije true
                        tp += 1
                    else:
                        fn +=1 # es la categoria y predije false
                else:
                    if entry[category] > u: #no es la categoria y predije true
                        fp += 1
                    else: # no es la categoria y predije false
                        tn +=1


            tfp = fp / (fp + tn)
            tvp = tp /  (tp+fn)
            roc_curve[category]["tfp"].append(tfp)
            roc_curve[category]["tvp"].append(tvp)

    for cat in categories:
        plt.plot(roc_curve[cat]["tfp"], roc_curve[cat]["tvp"], "-o", label=cat)

    line = [x/10 for x in range(0, 11)]
    plt.plot(line,line, "--", color="blue")
    axes = plt.gca()
    axes.set_xlim([-0.01,1.09])
    axes.set_ylim([-0.01,1.09])
    plt.xlabel("Taza de Falsos Positivos")
    plt.ylabel("Taza de Verdaderos Positivos")
    plt.legend()
    plt.show()
        

########################
# RUN LEARNING PROCESS #
########################

bayes = Bayes(mode=QUANTITY)

# Pre processing Argentine News dataset

argentine_news = pd.read_excel('/home/marina/ML2021/tp1/data/Noticias_argentinas.xlsx')
# we want to filter news without category
argentine_news = argentine_news.loc[argentine_news['categoria'].notnull()]

# when commenting this, success rate will be much lower
selected_categories = [
    'Deportes',
    'Salud',
    'Economia',
    'Entretenimiento',
    'Nacional',
    'Internacional',
    'Ciencia y Tecnologia',
]
argentine_news = argentine_news.loc[argentine_news['categoria'].isin(selected_categories)]

print("Finished pre processing.")

# Split into train and test
train, test = split_dataset(argentine_news, percentage=0.99)

print("Finished train-test splitting.")

# Processes all news
bayes.process_data(train)

# Method MUST BE called when finishing processing. Computes information for prediction with naives bayes
bayes.conclude_learning()

# print(bayes)

##################
# RUN TEST CASES #
##################

print("Running some tests for bayes algorithm...")

result_hits, result_verbose = bayes.test_batch(test)

print(f'\nHITS: {result_hits}/{len(test)}')
print(f'Success rate: {result_hits/len(test)}\n')

# print(result_verbose)
    


metrics = get_metrics(bayes.confusion_matrix)
print_metrics(metrics)

# Dejamos los ploteos comentados, pero descomentar en caso de necesitar alguno
# plot_roc(bayes.roc_data)
# plot_confusion_matrix(bayes.confusion_matrix)
