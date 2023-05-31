import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#The training data csv is being converted to a pandas dataframe with the header parameter set to None as the given data csv doesn't include headers
trainingData = pandas.read_csv("TrainingDataMulti.csv",header=None)

#The training data is being split into X (All features except the labels) and y (the labels) variables
X = trainingData.iloc[:, :128]
y = trainingData.iloc[:, 128]

#Splitting the data into training and testing data. The test_size and random_state have been selected in accordance to common machine learning standards. The data has been stratified on the y variable to ensure the training and testing data contain the same proportion of each label trace
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

#Creating RandomForestClassifier training model with certain hyperparameters tuned and fitting the training data to get accuracy readings
trainingModel = RandomForestClassifier(n_estimators=1000,random_state=42)
trainingModel.fit(X_train, y_train)

#Printing model error and accuracy readings
print("Training Model Accuracy: " + str(trainingModel.score(X_test,y_test)))
print("Training Model Error Rate: " + str(mean_absolute_error(y_test,trainingModel.predict(X_test))))

#The testing data csv is being converted to a pandas dataframe with the header parameter set to None as the given data csv doesn't include headers
testingData = pandas.read_csv("TestingDataMulti.csv",header=None)

#Creating RandomForestClassifier testing model with certain hyperparameters tuned and fitting the training data to predict the unknown labels for the testing data
testingModel = RandomForestClassifier(n_estimators=1000,random_state=42)
testingModel.fit(X,y)

#Predicting the unknown labels with the testing model using the testing data
predictedLabels = testingModel.predict(testingData)

#Printing the set of predicted labels
print("Predicted Labels: " + str(predictedLabels))

#Adding another column to the testingData dataframe which contains the predicted labels for each trace and then converting the dataframe to a CSV file
testingData[128] = predictedLabels
testingData.to_csv("TestingResultsMulti.csv",index=False,header=None)