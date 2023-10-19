import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

# Load the dataset
iris = pd.read_csv("C:\\Users\\Public\\Downloads\\foml\\IRIS.csv")

# Split the data into features (x) and target (y)
x = iris.drop("species", axis=1)
y = iris["species"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Initialize the classifiers
decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

# Fit the models on the training data
knearestclassifier.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
logisticregression.fit(x_train, y_train)
passiveAggressive.fit(x_train, y_train)

# Evaluate the models on the test data
data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier", 
                                       "Logistic Regression", "Passive Aggressive Classifier"],
      "Score": [knearestclassifier.score(x_test, y_test), decisiontree.score(x_test, y_test), 
                logisticregression.score(x_test, y_test), passiveAggressive.score(x_test, y_test)]}
score = pd.DataFrame(data1)
print(score)
