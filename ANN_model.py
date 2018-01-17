import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from matplotlib.colors import ListedColormap 

#importing data
dataset = pd.read_csv('Churn_Modelling.csv')
X = np.array(dataset.iloc[:,3:13].values)
Y = np.array(dataset.iloc[:, 13].values)

#categoricals of the dataset
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])

labelencoder1 = LabelEncoder()
X[:,2] = labelencoder1.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


#folding data into training and testing
from sklearn.model_selection import train_test_split
X_train ,X_test, Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state = 0)

#normalizing data
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)



from keras.models import Sequential
from keras.layers import Dense

#object for classifier
classifier = Sequential()
#adding first layer of the NN
classifier.add(Dense(units=6 , activation='relu' ,input_dim=11  ,kernel_initializer='uniform'))
#adding second layer of the NN
classifier.add(Dense(units=6 , activation='relu' ,kernel_initializer='uniform'))
#adding last layer of the NN
classifier.add(Dense(units=1 , activation='sigmoid' ,kernel_initializer='uniform'))
#compile NN to optimizer and loss function type
classifier.compile(optimizer='rmsprop' ,loss='binary_crossentropy' , metrics=['accuracy'])
#fitting the classifier with input data
classifier.fit(X_train,Y_train,batch_size=32,epochs=500)

#confusion matrix creation of test or training accuracy
from sklearn.metrics import confusion_matrix
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
cm = confusion_matrix(Y_test,Y_pred)












