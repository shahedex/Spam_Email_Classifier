from __future__ import print_function, division
import nltk
import os
from sklearn.svm import SVC
import pickle
from classifier_svm import init_lists, extract_feature, dictionary

loaded_model = pickle.load(open('trained_data.sav','rb'))
testham = init_lists('testmail/')
my_test_x = [(extract_feature(dictionary,am)) for am in testham]
final_result = loaded_model.predict(my_test_x)
res = final_result[1]
if res == 1:
	print('Your entered e-mail is a spam.')
else:
	print('Your entered e-mail is a HAM (Not Spam)')
