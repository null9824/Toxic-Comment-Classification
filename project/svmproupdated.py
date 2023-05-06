# Importing numpy library
import numpy as np
import pandas as pd  # especially for creating dataframes
import scipy.sparse as sps
import joblib
from sklearn.preprocessing import StandardScaler  # to standardize the data values into a standard format.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords  # natural langugae toolkidjt
from nltk.stem.porter import PorterStemmer
import re  # regular expression

# loading the data from csv file to pandas dataframe
# comment_data = pd.read_csv('2200newdataip.csv')
comment_data = pd.read_csv('./dataset112.csv')
#comment_data = pd.read_csv('./verynew20data.csv')


# print first 5 rows.
print(comment_data.head())

# no of rows and column in dataset

print(comment_data.shape)

print(comment_data.isnull().values.any())



# pip install contractions
import nltk
nltk.download('stopwords')


#PREPROCESS
#STEMMING

port_stem = PorterStemmer()
def stemming(text):
    stemmed_content = re.sub('[^a-zA-Z]',' ',str(text))
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content
                     if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

comment_data['comment_text'] = comment_data['comment_text'].apply(stemming)
print(comment_data['comment_text'])

#seperataing data and label
features = comment_data['comment_text'].values
target = comment_data['toxic'].values



# Vectorization



vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                             smooth_idf=1, sublinear_tf=1)
# vectorizer = TfidfVectorizer()

# vectorizer.fit(features)
features = vectorizer.fit_transform(features)
print(features)

print(len(vectorizer.vocabulary_))

print(vectorizer.vocabulary_)



X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)



# features.shape


# ALGORITHM

class SVM_classifier():

    # initiating the parameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter ): # lambda_parameter ->regularization parameter used to prevent overfitting
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    # fit dataset into model(SVM Classifier)
    def fit(self, X, Y):  # X-> all input features and Y-> Label/Outcome/Target
        # X = X_train, Y = Y_train. DIscussed in training model below.
        self.m, self.n = X.shape
        # m = total no of data points/no of rows.
        # n = no of input features/no of columns = w

        # NOW initiating the weight(w) value and bias value(b)

        self.w = np.zeros(self.n)  # numpy array

        self.b = 0  # just a single value. no need to use np array

        self.X = X
        self.Y = Y

        # inplementing the gradient descent algorithm for optimization

        for i in range(self.no_of_iterations):
            self.update_weights()  # Each time the model goes through data, it will try to change the weight and bias value so that we get accurate predictions from our model.

    # for updating the weights and bias value
    def update_weights(self):
        # label encoding
        # np.where tries to find condition. numpy.where() function returns the indices of elements in an input array where the given condition is satisfied
        # if label value is 0 then convert it into -1 else 1

        y_label = np.where(self.Y <= 0, -1, 1)  # for case of svm. it takes +1 or -1. see y cap.
        # gradient descent( finding-> dw, db)

        for index, x_i in enumerate(
                self.X):  # Enumerate() method adds a counter to an iterable and returns it in a form of enumerating object.

            # np.transpose (self.w)
            # condition = (y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1)
            condition = (y_label[index] * (
                    np.dot(x_i.toarray(), np.transpose(self.w)) - self.b) >= 1)  # 2 conditions
            # y_label[index] = outcome's label for individual.     # np.dot = dot product of two arrays. # x_i = all feates of single individual.

            print(condition)
            if (condition == True):
                # if (y_label[index] * (np.dot(x_i,self.w) - self.b) >= 1):

                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:

                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            # now for formula of gradientdescent
            # 1. w2 = w1 -L*dw
            # w2 = updated weight w1 = previous weight L = learning rate

            # 2. b = b-a*db

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    # predict the label for a given input value

    def predict(self, X):  # X -> new value/input

        # output = np.dot(X, self.w) - self.b # eqn of hyperplane. This value can be any positive or negative numbers.
        output = np.dot(X.toarray(), np.transpose(self.w)) - self.b

        # but we need only +1 or -1 so generalize "output"

        predicted_labels = np.sign(output)  # np.sign is used to indicate the sign of a number element-wise.
        # For integer inputs, if array value is greater than 0 it returns 1, if array value is less than 0 it returns -1, and if array value 0 it returns 0
        # This is exactly what we need/SVM need.
        # REverse the label encoding
        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat


classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)

classifier.fit(X_train, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy_svm = accuracy_score(Y_test, X_test_prediction)

print("Acccuracy score on test dataa =", test_data_accuracy_svm * 100, '%')
# X_test_prediction
accucaryscore = test_data_accuracy_svm * 100

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, X_test_prediction)
print(cm)


# JOBLIB
# save model

from joblib import dump

filename = 'preprocess.joblib'
joblib.dump(stemming, filename)

filename = 'model.joblib'
joblib.dump(classifier, filename)

filename = 'vectorizer.joblib'
joblib.dump(vectorizer, filename)



# import pickle
#
# filename = 'final_preprocessing.pkl'
# pickle.dump(preprocess, open(filename, 'wb'))
#
# filename = 'final_svmmodel.pkl'
# pickle.dump(classifier, open(filename, 'wb'))
#
# # save vectorizer
# filename = 'final_vectorizer.pkl'
# pickle.dump(vectorizer, open(filename, 'wb'))
#
# # input from user
# preprocess_save = pickle.load(open("final_preprocessing.pkl", "rb"))
# vectorizer_save = pickle.load(open("final_vectorizer.pkl", "rb"))
#
# classifier_save = pickle.load(open("final_svmmodel.pkl", "rb"))

# just for testing purpose (not required)
# newdata = pd.read_csv("test.csv")
# a = newdata['comment_text']

for i in range (2):
    inp = input("Enter comment")
    a = [inp]
    print(a)
    c = stemming(a)
    print("stem")
    print(c)
    c = [c]
    print(c)
    print("ccc")
    dx = vectorizer.transform(c)
    print("ddd")

    print(dx)
    print("eee")

    pred_value = classifier.predict(dx)
    print(pred_value)

    if pred_value == 1:
        print("You are using toxic word in your comment")

    else:
        print("Submitted")

# accuracy score on the test data
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Acccuracy score on test data =", test_data_accuracy * 100, '%')

# import os
#
# os.getcwd()
