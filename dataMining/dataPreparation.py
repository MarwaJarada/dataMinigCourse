from math import floor, ceil, nan
import pandas as pd
import numpy as np
from numpy.ma import count
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


'''
@Marwa N.Jarada
@220170326
'''
'''
Discovering The dataSet And It's Attributes:
'''
# Reading dataset and save in variable"heartDiseaseDataset"
heartDiseaseDataset = pd.read_csv("D:\\6th Semester\Data Mining\\framingham.csv")
pd.set_option("display.max.columns", None) # To show all attributes
print("Total rows:",heartDiseaseDataset.axes[0]) # rows=4238
print("Total columns:",heartDiseaseDataset.axes[1],"\\\t Number of Columns:") #rows=[...]


'''
Dealing With Missing Data:
'''
# Show the count of every attribute to make sure data is not missing
#  If count column==dataset length == 4238 then there is no missing data
print("The count of fields in every single column:\\\n",heartDiseaseDataset.describe())
# Now we know that there is a missing data in fields in[education,cigsPerDay,BPMeds,totChol,BMI,heartRate,glucose]
attributesList=["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp",
                 "diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose","TenYearCHD"] #All Columns
print("Number of null values of every column:\\\n",heartDiseaseDataset.isnull().sum())
print("Sum of all null values:",heartDiseaseDataset.isnull().sum().sum()) # Number of null fields
# We have 105 empty field in [education] column , But [education] doesn't  effect on heart Disease,645
#  so we need to drop this column
heartDiseaseDatasetNoEdu=heartDiseaseDataset.drop("education",axis=1)
print("After deleting education column,Sum of all null values:",
      heartDiseaseDatasetNoEdu.isnull().sum().sum()) # Now number of null fields reduced into 540
# We have 1 missing field in heart rate, so we can drop this row:
heartDiseaseDatasetNoEdu=heartDiseaseDatasetNoEdu[pd.notnull(heartDiseaseDatasetNoEdu["heartRate"])]
# Now we have 4237 row (one deleted)
columnsMissingData=["cigsPerDay","BPMeds","totChol","heartRate","BMI"]# Columns has ((low)) number of missing data
for column in columnsMissingData:
    heartDiseaseDatasetNoEdu = heartDiseaseDatasetNoEdu[pd.notnull(heartDiseaseDatasetNoEdu[column])] # Rows Deleted
print("ee",heartDiseaseDatasetNoEdu.isnull().sum())

# The number of missing data in glucose column = 388, so we want to fill null values with the mean
print(heartDiseaseDatasetNoEdu["glucose"].min())
print(heartDiseaseDatasetNoEdu["glucose"].max())
glucoseMean=heartDiseaseDatasetNoEdu["BMI"].mean()

lowWieghtMean=heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.BMI<30]["BMI"].mean()#24.7
hieghtWieghtMean=heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.BMI>=30]["BMI"].mean();#33.2
# heartDiseaseDatasetNoEdu["glucose"]=heartDiseaseDatasetNoEdu.where((heartDiseaseDatasetNoEdu["glucose"].isnull) and
#                                                                    (heartDiseaseDatasetNoEdu["BMI"]>=30))
#
# print("Before:",heartDiseaseDatasetNoEdu["glucose"])
# heartDiseaseDatasetNoEdu["glucose"]=np.where(heartDiseaseDatasetNoEdu["glucose"].equals("NaN") and heartDiseaseDatasetNoEdu["BMI"]>=30
#                                              ,hieghtWieghtMean,lowWieghtMean)
# print("After:",heartDiseaseDatasetNoEdu["glucose"])
heartDiseaseDatasetNoEdu["glucose"]=heartDiseaseDatasetNoEdu["glucose"].fillna(glucoseMean)
print("Number of Null values=",heartDiseaseDatasetNoEdu.isnull().sum().sum(),"\\\nNumber of records=",len(heartDiseaseDatasetNoEdu))
#################### Now we have 0 null values in all rows ####################


'''
Dealing With Noisy Data
'''
## This for loop help us to discover if ther is noisy data specially in nominal attributes like: gender=[0,1]
for attr in attributesList:
    print(heartDiseaseDataset[attr].value_counts())
## No noisy in gender,currentSmoker,BPMeds,prevalentStroke,prevalentHyp,diabetes Attributes
print(heartDiseaseDatasetNoEdu["age"].max()) #70
print(heartDiseaseDatasetNoEdu["age"].min()) #32

## No noise data in age
print(heartDiseaseDatasetNoEdu["cigsPerDay"].max()) #70 Not Normal value (So high)
print(heartDiseaseDatasetNoEdu["cigsPerDay"].min()) #0
cigsPerDayMean=ceil(heartDiseaseDatasetNoEdu["cigsPerDay"].mean()) #9
print("Number of records has cigsPerDay more than 9=",heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.cigsPerDay>9].count())
# 1535 record has more than 9 cigsPerDay
print("qq",heartDiseaseDatasetNoEdu)
heartDiseaseDatasetNoEdu.loc[heartDiseaseDatasetNoEdu["cigsPerDay"]>9,"cigsPerDay"]=9
print("Number of records has cigsPerDay more than 9=",heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.cigsPerDay>9].count()) # 1535 record has more than 9 cigsPerDay
# 0 record has more than 9 cigsPerDay
print("qq",heartDiseaseDatasetNoEdu)
## No noise data in cigsPerDay
print(heartDiseaseDatasetNoEdu["totChol"].min())
print(heartDiseaseDatasetNoEdu["totChol"].max())

#Number of currentSmoker=0 must be equal to cigsPerDay=0
print("Number of no smoker people=",heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.currentSmoker==0].count().groupby)
print("Number of cigsPerDay\\0 =",heartDiseaseDatasetNoEdu[heartDiseaseDatasetNoEdu.currentSmoker==0].count().groupby)

# In BMI we have 1109 unique values (high range, we can use BMI as Integer value)
print("Number of unique values in double type=",count(heartDiseaseDatasetNoEdu["BMI"].unique())) #1109
heartDiseaseDatasetNoEdu["BMI"]=heartDiseaseDatasetNoEdu["BMI"].astype(int)
print("Number of unique values in integer type=",count(heartDiseaseDatasetNoEdu["BMI"].unique())) #34


'''
Duplicated Data
'''

print("Duplicated data=",heartDiseaseDatasetNoEdu.duplicated())# No duplicated data here
# heartDiseaseDatasetNoEdu.drop_duplicates()


'''
Removing Irrelevant Attributes
'''

# Education attribute removed before in (Missing Data Level)
# No need foe current smoker because we know if person comke or not if cigsPerDay==0
# We are sure that cigsPerDay==0 for every person has field currentSmoker=0, we checked that in (noisy data level)
heartDiseaseDatasetNoEdu=heartDiseaseDatasetNoEdu.drop("currentSmoker",axis=1)


'''
Transformation Using Min-Max Normalization
'''

# Create scaler
minmaxScale = MinMaxScaler(feature_range=(-0.8, +0.8))
# Scale feature
attTranformation=["totChol","sysBP","diaBP","BMI","heartRate","glucose"]
for attrTra in attTranformation:
    heartDiseaseDatasetNoEdu[[attrTra]] = minmaxScale.fit_transform(heartDiseaseDatasetNoEdu[[attrTra]])
print(heartDiseaseDatasetNoEdu)

'''
Correlation Attributes
'''
iris = load_iris()
x = iris.data
y = iris.target
corr_matrix = heartDiseaseDatasetNoEdu.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(np.bool))


'''
Splitting
'''
featuers = heartDiseaseDatasetNoEdu.drop(['TenYearCHD'], axis=1) # Dropping the target Attribute
print("FEATURES:")
print(featuers.head())
labels = heartDiseaseDatasetNoEdu.TenYearCHD # Saving the Target Attribute in variable
print("LABEL:")
print(labels.head())
feature_train,feature_test,label_train,label_test=train_test_split(featuers,labels,test_size=0.3,train_size=0.7)





