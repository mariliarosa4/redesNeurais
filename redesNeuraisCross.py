import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict,KFold
from matplotlib import pyplot as plt
from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix
dataset = pd.read_csv("mammographic_masses_classficacao.data.csv",sep=",")
dataset.iloc[1] = dataset.iloc[1].apply(pd.to_numeric, errors='coerce')
dataset.iloc[0] = dataset.iloc[0].apply(pd.to_numeric, errors='coerce')
dataset.iloc[2] = dataset.iloc[2].apply(pd.to_numeric, errors='coerce')
dataset.iloc[3] = dataset.iloc[3].apply(pd.to_numeric, errors='coerce')
dataset.iloc[4] = dataset.iloc[4].apply(pd.to_numeric, errors='coerce')

X = dataset.iloc[:,:4]
y = dataset.iloc[:,5]

seed = 7
numpy.random.seed(seed)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
      beta_2=0.999, early_stopping=False, epsilon=1e-08,
      hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
      learning_rate_init=0.001, max_iter=200, momentum=0.9,
      nesterovs_momentum=True, power_t=0.5, random_state=None,
      shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
      verbose=False, warm_start=False)

kf = KFold(n_splits=10) # Define the split - into 2 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)
KFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):

   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   mlp.fit(X_train, y_train)
   print confusion_matrix(y_test, mlp.predict(numpy.array(X_test)))

#
#
# results = cross_val_score(mlp, X, y, cv=kfold)
# predictions = cross_val_predict(mlp, X, y, cv=kfold)
# accuracy = metrics.r2_score(y, predictions)
# print "Accuracy:", results.mean()
#
# predict = (cross_val_predict(mlp,X,y,cv=kfold))
# conf_mat = confusion_matrix(y,predict)
# print(conf_mat)

# print(classification_report(y_test,predictions))