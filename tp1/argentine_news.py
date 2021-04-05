import math
import numpy as np
import operator
import pandas as pd

# This mode computes only if the word appeared in the news or not
APPEARANCES = 0
# This mode computes how many times the word appeared for that category
QUANTITY = 1


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
        self.non_existing_word_probability =  self.get_rf(0)

    # TODO: replace this when deciding what to do with probabilities
    def get_prod(self, headline):
        tokens = headline.lower().split()

        prod = 1

        if self.mode == APPEARANCES:
            distinct_tokens = list(set(tokens))
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
        self.categories = {}
        self.mode = mode

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
        results = {name: category.get_prod(headline) for name, category in self.categories.items()}
        winner = max(results.items(), key=operator.itemgetter(1))[0]
        return results, winner, winner == real_category

    def test_batch(self, test_df):
        hits = 0
        verbose = ''
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            headline = row.titular
            real_category = row.categoria
            results, winner, success = self.test(headline, real_category)
            hits += 1 if success else 0
            verbose += f'** Winner: {winner}, Real: {real_category} **\n'
        return hits, verbose

    def __str__(self):
        desc = ''
        for name, category in self.categories.items():
            desc += f'{name}: {category.probability}\n'
        return desc


########################
# RUN LEARNING PROCESS #
########################

# TODO: for now use appearances until resolving problems with probabilities
bayes = Bayes(mode=APPEARANCES)

# Pre processing Argentine News dataset

argentine_news = pd.read_excel('data/Noticias_argentinas.xlsx')
# we want to filter news without category
argentine_news = argentine_news.loc[argentine_news['categoria'].notnull()]

# when commenting this, success rate will be much lower
selected_categories = ['Deportes', 'Salud', 'Economia', 'Entretenimiento']
argentine_news = argentine_news.loc[argentine_news['categoria'].isin(selected_categories)]

print("Finished pre processing.")

# Split into train and test
msk = np.random.rand(len(argentine_news)) < 0.8
train = argentine_news[msk]
test = argentine_news[~msk]

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

print()
print(f'HITS: {result_hits}/{len(test)}')
print(f'Success rate: {result_hits/len(test)}')

print()
# print(result_verbose)
