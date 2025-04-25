import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a pandas dataframe
credit_card_data = pd.read_csv('/content/creditcard.csv')

credit_card_data.head()

credit_card_data.tail()

credit_card_data.info()

credit_card_data.isnull().sum()

credit_card_data['Class'].value_counts()

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
