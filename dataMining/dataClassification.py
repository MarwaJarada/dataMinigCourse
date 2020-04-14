from dataMining.dataPreparation import heartDiseaseDatasetNoEdu
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import NearestNeighbors as knn, NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score

'''
@Marwa N. Jarada
220170326
'''
# Preparing the training and testing sets
print("DATA CLASSIFICATION")
featuers = heartDiseaseDatasetNoEdu.drop(['TenYearCHD'], axis=1) # Dropping the target Attribute
print("FEATURES:")
print(featuers.head())
labels = heartDiseaseDatasetNoEdu.TenYearCHD # Saving the Target Attribute in variable
print("LABEL:")
print(labels.head())
feature_train,feature_test,label_train,label_test=train_test_split(featuers,labels,test_size=0.3,train_size=0.7)


# 1.Using KNN Classification
model = KNeighborsClassifier(5) # The knearest k=5
model.fit(feature_train, label_train)
predicts=model.predict(feature_test)
print("PREDICT RESULT Using KNN:",predicts)
accuracy = accuracy_score(label_test, predicts)
print('Accuracy of KNN classifier :',accuracy) # =82%


# 2.Using Naive Bayes Classification
model = gnb()
model.fit(feature_train, label_train)
predicts=model.predict(feature_test)
print("PREDICT RESULT Using Naive Bayes:",predicts)
accuracy = accuracy_score(label_test, predicts)
print('Accuracy of Naive Bayes classifier :',accuracy) # =82%

# 3.Using Decision Tree Induction Classification

# First we need to convert continuous values into categorial as much as we can
print(max(featuers.age),min(featuers.age)) # to know pins range #70 #32
ageCategory=pd.cut(featuers.age,bins=[0,17,32,65,100],labels=['child','teenager','adult','elderly'])
cigsPerDayCategory=pd.cut(featuers.cigsPerDay,bins=[-1,2.0,5.0,7.0,20.0],labels=['low','medium','high','veryHigh'])
featuers.insert(2,"ageCategory",ageCategory)
featuers.insert(4,"cigsPerDayCategory",cigsPerDayCategory)
# Now after adding a categorial columns we need to drop continuous values columns
del featuers['age']
del featuers['cigsPerDay']
''' All totChol,sysBP,diaBP,BMI,heartRate,glucose has continuous values between : -0.8,+0.8'''
continuousValuesWithSameRange=["totChol","sysBP","diaBP","BMI","heartRate","glucose"]
for column in continuousValuesWithSameRange:
    columnCategory = pd.cut(featuers[column], bins=[-0.9, -0.3,4,9],
                            labels=['low', 'medium', 'high'])
    featuers.insert(13, column+"Category",columnCategory)
    # Now after adding a categorial columns we need to drop continuous values columns
    del featuers[column]
print("Categorial data set:",featuers) # NOW Data is ready to use Decision Tree Induction Classification
from sklearn.tree import DecisionTreeClassifier as dt
print("here")
model = dt(random_state=1)
model.fit(feature_train, label_train)
predicts=model.predict(feature_test)
print("PREDICT RESULT Using Decision Tree Induction:",predicts)
accuracy = accuracy_score(label_test, predicts)
print('Accuracy of Decision Tree Induction classifier :',accuracy) # =77%