'''to use this part : copy all code here below the code in Ann_model.py and execute this part in command shell.'''
'''####important note : Do not change this code because the classifier must be with the same name of classifier in Ann_model.py####'''
'''#this part only for getting the best parameters for fitting '''
'''after getting best accuracy check variable "best_accuracy" with its attributes that is generated , and change values for parameters in : optimizer , batch size , epochs 
in this lines : "classifier.fit(X_train, y_train, batch_size = , epochs = )" and "classifier.compile(optimizer='' ,loss='binary_crossentropy' , metrics=['accuracy'])"'''


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6 , activation='relu' ,input_dim=11  ,kernel_initializer='uniform'))
    classifier.add(Dense(units=6 , activation='relu' ,kernel_initializer='uniform'))
    classifier.add(Dense(units=1 , activation='sigmoid' ,kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer ,loss='binary_crossentropy' , metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#tuning of model
parameter = {'batch_size':[25 ,32],
             'epochs':[100 ,500],
             'optimizer':['adam' ,'rmsprop']}
grid_search = GridSearchCV(
              estimator = classifier ,
              param_grid = parameter ,
              scoring = 'accuracy' ,
              cv = 10)
grid_search = grid_search.fit(X_train,Y_train)
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_


#confusion matrix creation of test and training accuracy
from sklearn.metrics import confusion_matrix
Y_pred = grid_search.predict(X_test)
cm = confusion_matrix(Y_test,Y_pred)