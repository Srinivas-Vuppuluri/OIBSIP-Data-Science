# -*- coding: utf-8 -*-
"""SMSClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ar3X3PwkSVUJ4fYZ8CMzWaY7L5PYj7fB

# Importing the libraries
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""## Convert Text into Feature vector or Numeric values"""

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""## Data Collection & Pre-processing"""

raw_mail_data = pd.read_csv('spam.csv', encoding='latin')

print(raw_mail_data)

raw_mail_data.isnull().sum()

mail_data =raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.head()

mail_data.shape

"""## Rename the columns"""

mail_data=mail_data.rename(columns={'v1':'Category',
                                    'v2':'Message'})

mail_data.head()

"""# Label Encoding:
## Label spam mail as 0.
## Label ham mail as 1.
"""

mail_data.loc[mail_data['Category'] == 'spam','Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category',] = 1

"""## Separating the data as texts and label"""

X = mail_data['Message']
Y = mail_data['Category']

print(X)

print(Y)

"""## Splitting the data into Training data & Test data"""

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=3)

print(X.shape)

print(X_train.shape)

print(X_test.shape)

"""## Feature extraction"""

feature_extraction = TfidfVectorizer(min_df = 1,stop_words='english',lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

Y_train

Y_test

X_train

print(X_train_features)

"""# Training the Model
## Logistics Regression
"""

model = LogisticRegression()

model.fit(X_train_features, Y_train)

"""# Evaluating the Training model
## Prediction on Training data
"""

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data)

"""## Prediction on Test data"""

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

print('Accuracy on test data : ', accuracy_on_test_data)

"""# Building Predictive system
## Import necessary libraries
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

"""## Sample data for illustration"""

input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have b"]

"""## Sample training data"""

training_data = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have b",
                 "Get a free gift now! Click here.",
                 "Congratulations, you've won a lottery!",
                 "Invest in this amazing opportunity today.",
                 "Meeting postponed to tomorrow."]

labels = [0, 1, 1, 1, 0]  # 0: Ham mail, 1: Spam mail

"""## Feature extraction"""

feature_extraction = CountVectorizer()
input_data_features = feature_extraction.fit_transform(training_data)

"""## Train a Naive Bayes classifier"""

model = MultinomialNB()
model.fit(input_data_features, labels)

"""## Convert the input mail to feature vectors"""

input_data_features = feature_extraction.transform(input_mail)

"""## Making prediction"""

prediction = model.predict(input_data_features)

"""# Display the prediction"""

if prediction[0] == 0:
    print('Ham mail')
else:
    print('Spam mail')