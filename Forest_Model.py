#Import scikit-learn dataset library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Great resource - https://www.datacamp.com/community/tutorials/random-forests-classifier-python

#Load dataset
iris = datasets.load_iris()

#### Classifier setup ####
'''RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False) '''

############################## ---- Building Classifiers ----- #############################
######## Explore Data ############

# print the label species(setosa, versicolor,virginica)
print(iris.target_names)

# print the names of the four features
print(iris.feature_names)

# print the iris data (top 5 records)
print(iris.data[0:5])

# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
print(iris.target)

######## Create a data frame ###########

data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()

################ Split data into labels (X) and data (Y) and test/training sets ################

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

######### Split dataset into training set and test set ##########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # 66% training and 33% test



######### Create a Gaussian Classifier ###########
clf=RandomForestClassifier(n_estimators=100)

############# Train the model #############

clf.fit(X_train,y_train)

############# Predict using model ###############

y_pred=clf.predict(X_test)

############### Check Accuracy ##################

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Predict

######### Make single item prediction ###########
sepal_length = 3
sepal_width = 5
petal_length = 4
petal_width = 2

clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])  # Number answer will indicate the flow type array([2]) being Virginica

###################### --------Finding Important Features-------- ################

clf=RandomForestClassifier(n_estimators=100) # Create Gaussian classifier

############## Train model #############
clf.fit(X_train,y_train) 

feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
feature_imp

########## Creating a bar plot #############
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


###################### -------Generating a model for a single feature---------- ########################

########## Split dataset into features and labels #############
X=data[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"
y=data['species']                                       
######### Split dataset into training set and test set ##################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66, random_state=5) # 66% training and 33% test



########### Perform prefiction of select classifications ###########
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

########## Train the model using the training sets ############
clf.fit(X_train,y_train)

########## prediction on test set ############
y_pred=clf.predict(X_test)

############ Model Accuracy ############# 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # how often is the classifier correct?

