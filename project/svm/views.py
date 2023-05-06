from django.shortcuts import render
import numpy as np
from joblib import load
from scipy.sparse import csr_matrix

from svmproupdated import test_data_accuracy, accucaryscore, test_data_accuracy_svm

print(test_data_accuracy)

# from svmproupdated import predictor, preprocess_model, svm_model, vectorizer_model, stemming
from naivebayes import MultinomialNaiveBayesNew, test_data_accuracy_nb

preprocess_model = load('./preprocess.joblib')

svm_model = load('./model.joblib')
nbmodel = load('./nb_model.joblib')

vectorizer_model = load('./vectorizer.joblib')

#SVM
# def predictorsvm(request):
#     if request.method == 'POST':
#         cmnt = request.POST['comment_box']
#         print("Commentsvm:", cmnt)
#         cmnt_preprocess = preprocess_model(cmnt)
#         print("Preprocessed commentsvm:", cmnt_preprocess)
#         # cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string
#         cmnt_vectorize = vectorizer_model.transform([cmnt_preprocess])
#         # cmnt_vectorize = csr_matrix(cmnt_vectorize)
#         print("Vectorized commentsvm:", cmnt_vectorize)
#         cmnt_predict = svm_model.predict(cmnt_vectorize)
#         print("Predictionsvm:", cmnt_predict)
#
#         if cmnt_predict == 0:
#             return render(request, '1.html', {'result1': 'Your comment is acceptable. Keep it up! svm'})
#         else:
#             return render(request, '1.html',
#                           {'result1': 'This comment seems to be toxic. Please refrain from posting harmful content svm'})
#         # else:
#         #     return render(request, 'home.html', {'result': 'Accuracy: ' + test_data_accuracy * 100})
#     return render(request, '1.html')
# #

# Naive bayes
def predictornb(request):
    if request.method == 'POST':
        cmnt2 = request.POST['comment_box']
        print("Commentnb:", cmnt2)
        cmnt_preprocess2 = preprocess_model(cmnt2)
        print("Preprocessed commentnb:", cmnt_preprocess2)
        # cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string
        cmnt_vectorize2 = (vectorizer_model.transform([cmnt_preprocess2]).toarray())
        #cmnt_vectorize2 = np.array(vectorizer_model.transform([cmnt_preprocess2]))
        # cmnt_vectorize = csr_matrix(cmnt_vectorize)
        #print("Vectorized commentnb:", cmnt_vectorize2)
        cmnt_predict2 = nbmodel.predict(cmnt_vectorize2)
        print("Predictionnb:", cmnt_predict2)
        accnb = str(test_data_accuracy_nb * 100)
        print("Accuracy svm " + str(test_data_accuracy_nb))




        #svm
        cmnt = request.POST['comment_box']
        print("Commentsvm:", cmnt)
        cmnt_preprocess = preprocess_model(cmnt)
        print("Preprocessed commentsvm:", cmnt_preprocess)
        # cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string
        cmnt_vectorize = vectorizer_model.transform([cmnt_preprocess])
        # cmnt_vectorize = csr_matrix(cmnt_vectorize)
        #print("Vectorized commentsvm:", cmnt_vectorize)
        cmnt_predict = svm_model.predict(cmnt_vectorize)
        print("Predictionsvm:", cmnt_predict)
        accsvm = str(test_data_accuracy_svm * 100)
        print("Accuracy svm " + str(test_data_accuracy_svm))


        # if cmnt_predict == 0 or cmnt_predict == 1:
        if cmnt_predict == 0 and cmnt_predict2 == 0:
            return render(request, '1.html', {'result1': 'Your comment is acceptable. svm. Accuracy of SVM:' + accsvm,
                                              'result2': 'Your comment is acceptable. Keep it up! nb. Accuracy of NB:' + accnb,

                                              })

        elif cmnt_predict == 1 and cmnt_predict2 == 0:
            return render(request, '1.html', {'result1': 'This comment seems to be toxic. svm.  Accuracy of SVM:' + accsvm,
                                              'result2': 'Your comment is acceptable nb. Accuracy of NB:' + accnb,
                                              })
        elif cmnt_predict == 0 and cmnt_predict2 == 1:
            return render(request, '1.html', {'result1': 'Your comment is acceptable svm. Accuracy of SVM:' + accsvm,
                                              'result2': 'This comment seems to be toxic.nb. Accuracy of NB:' + accnb,
                                              })
        else:
            return render(request, '1.html', {'result1': 'This comment seems to be toxic.svm. Accuracy of SVM:' + accsvm,
                                              'result2': 'This comment seems to be toxic.nb. Accuracy of SVM:' + accnb,
                                              })
        # if cmnt_predict2 == 0 or cmnt_predict2 == 1:
        #     if cmnt_predict2 == 0:
        #         return render(request, '1.html', {'result2': 'Your comment is acceptable. Keep it up! nb '})
        #     elif cmnt_predict2 == 1:
        #         return render(request, '1.html',
        #                   {'result2': 'This comment seems to be toxic. Please refrain from posting harmful content nb'})

        # else:
        #     return render(request, 'home.html', {'result': 'Accuracy: ' + test_data_accuracy * 100})
    return render(request, '1.html')



# def predictor(request):
#     result = None
#     if request.method == 'POST':
#         cmnt = request.POST['comment_box']
#         print(cmnt)
#         cmnt_preprocess = preprocess_model(cmnt)
#         print(cmnt_preprocess)
#         cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string

#         cmnt_vectorize = vectorizer_model.transform([cmnt_preprocess])

#         cmnt_vectorize = csr_matrix(cmnt_vectorize)

#         cmnt_predict = svm_model.predict(cmnt_vectorize)
#         print(cmnt_predict)
#         if cmnt_predict == 0:
#             result = 'Your comment is acceptable. Keep it up!'
#         else:
#             result = 'This comment seems to be toxic. Please refrain from posting harmful content'
#     return render(request, 'home.html', {'result': result})
