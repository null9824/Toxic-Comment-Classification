import joblib



# Importing numpy library
import numpy as np
import pandas as pd # especially for creating dataframes
from sklearn.preprocessing import StandardScaler # to standardize the data values into a standard format.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords # natural langugae toolkit
from nltk.stem.porter import PorterStemmer
import re # regular expression

comment_data1 = pd.read_csv('./dataset112.csv')
print("Naive bayessss")
import nltk
nltk.download('stopwords')


#seperataing data and label
features = comment_data1['comment_text'].values
target = comment_data1['toxic'].values

port_stem = PorterStemmer()
def stemming(content):
  stemmed_content = re.sub(r'[^a-zA-Z_]',' ',content)
  # stemmed_content = re.sub(r"<.*>"," ",content, flags=re.MULTILINE) #to remove html tags (< >) and its content from the input
  # stemmed_content = re.sub(r"http\S+"," ",content, flags=re.MULTILINE) # to remove any kind of links with no html tags
  # stemmed_content = re.sub(r"www\S+"," ",content, flags=re.MULTILINE)
  # stemmed_content = re.sub(r"[\n\t\\\/]"," ",content, flags=re.MULTILINE) # to remove newline (\n),tab(\t) and slashes (/ , \) from the input text
  # stemmed_content = re.sub(r"(\w)(\1{2,})","\\1",content,flags=re.MULTILINE)  # """function to remove repeated characters if any from the input text"""
#     """for example CAAAAASSSSSSEEEEE SSSSTTTTTUUUUUUDDDDYYYYYY gives CASE STUDY"""
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content
                     if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

comment_data1['comment_text'] = comment_data1['comment_text'].apply(stemming)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(features)
features = vectorizer.transform(features)

# spliting data into train and test data # changed-> stratify=target, random_state=42
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# check
X_train, X_test, Y_train, Y_test = features[:1000],features[1000:],target[:1000],target[1000:]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#
# #NEWWWW ALGO
class MultinomialNaiveBayesNew:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        # Compute class probabilities/ prior probability
        self.class_probabilities = np.zeros(n_classes) #creates a NumPy array of size n_classes initialized with zeros and paxi filled with calculated class probabilities.
        for i, c in enumerate(self.classes):
            self.class_probabilities[i] = np.sum(y == c) / float(n_samples)
            #np.sum function is used to count the number of samples in the training set that belong to each class'c'.
            #y == c returns a boolean array that is True where the class label is equal to c, and False otherwise.
            #The np.sum function then counts the number of True values in this array, which is equivalent to counting the number of samples that belong to class c
            #divide ko case
            #Dividing this count by the total number of samples n_samples gives the probability of class c in the training set. The result is stored in the self.class_probabilities array, with one entry for each class.

        # Compute feature probabilities laplace smoothing ko kura/ posterior prob
        self.feature_probabilities = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c] # first subsets the training data X to obtain only the samples that belong to a particular class c
            total_count = np.sum(X_c) # total number of occurrences of all features in X_c.
            self.feature_probabilities[i, :] = (np.sum(X_c, axis=0) + self.alpha) / (
                        total_count + self.alpha * n_features)
#sums up the number of occurrences of each feature in the subset of training data that belongs to class i,
    # which is stored in X_c, using np.sum(X_c, axis=0). Then, it adds a smoothing parameter self.alpha
# The resulting array self.feature_probabilities has shape (n_classes, n_features),


# UPDATED PREDICT()
    def predict(self, X):
        n_features_expected = self.feature_probabilities.shape[1]
        # dimesion milauna check
        # Checking if the number of features in X matches the number of features in feature_probabilities
        if X.shape[1] != n_features_expected:
            # Truncate/remove or pad X to match the number of features in feature_probabilities
            if X.shape[1] > n_features_expected:
                X = X[:, :n_features_expected] # X is truncated by keeping only the first n_features_expected columns of X.
                # : -> select all rows and only the first n_features_expected columns of X.
            else:
                padded = np.zeros((X.shape[0], n_features_expected))
                # new array padded filled with zeros of shape (X.shape[0], n_features_expected)
                # and copies the existing data from X into the first X.shape[1] columns of padded
                padded[:, :X.shape[1]] = X # X is padded with zeros to match the expected number of features.
                X = padded # X is updated
                # This ensures that X has the same number of features as the trained model.

        # Check if the number of classes matches the length of self.classes
        if len(self.classes) != self.feature_probabilities.shape[0]:
            raise ValueError("The number of classes does not match the length of self.classes")

        # Calculate log probabilities for each class
        log_probs = np.zeros((X.shape[0], len(self.classes))) #array of zeros of shape (n_samples, n_classes)
        for i, c in enumerate(self.classes):
            #P(Toxic)
            prior = np.log(self.class_probabilities[i]) 
            #natural logarithm  is used instead of the regular logarithm
            # because it helps to avoid numerical underflow and overflow errors that can occur when working with very small or large probabilities.
            #P(comment/Toxic)
            conditional = np.sum(np.log(self.feature_probabilities[i, :].reshape(1, -1)) * X, axis=1) #axis =1 specifies that the sum should be taken across the features dimension (i.e., for each sample in X).
            log_probs[:, i] = prior + conditional # i-th column of the log_probs array,

        # Return class with maximum log probability
        return self.classes[np.argmax(log_probs, axis=1)]
#

clssa = MultinomialNaiveBayesNew()
clssa.fit(X_train, Y_train)

# accuracy score on the test data
#X_test_prediction = clssa.predict(X_test.toarray())
X_test_prediction = clssa.predict(X_test.toarray()) #milako
# X_test_prediction = clssa.predict(X_test)
test_data_accuracy_nb = accuracy_score(Y_test,X_test_prediction )
print(test_data_accuracy_nb)

print(X_train.shape)
print(Y_train.shape)




filename = 'nb_model.joblib'
joblib.dump(clssa, filename)


for i in range(5):
    inp = input("Enter comment: ")


    # a = [inp]
    print(inp)
    c = stemming(inp)  # fix: pass inp instead of a to stemming()
    print("stem")
    print(c)
    # c = [c]
    # print(c)
    print("ccc")
    dx = (vectorizer.transform([c])).toarray()
    #dx = (vectorizer.transform(c))

    print("ddd")

    print(dx)
    print("eee")

   # ne = np.array(vectorizer.transform(c))
    pred_value = clssa.predict(dx)
    #print("Okkkk" + pred_value)


    if pred_value == 1:
        print("You are using toxic words in your comment okk")

    else:
        print("Submitted okk")