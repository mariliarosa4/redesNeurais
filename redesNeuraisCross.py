import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict,KFold
from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix
dataset = pd.read_csv("mammographic_masses_classficacao.data.csv",sep=",")
dataset = dataset.replace("?", -1)
X = numpy.array(dataset.iloc[:,:4])
y = numpy.array(dataset.iloc[:,5])
def somarMatrizes(matriz1, matriz2):
    if(len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0])):
        return None
    result = []
    for i in range(len(matriz1)):
        result.append([])
        for j in range(len(matriz1[0])):
            result[i].append(matriz1[i][j] + matriz2[i][j])
    return result
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
cont=1
for train_index, test_index in kf.split(X):

     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

     mlp.fit(X_train, y_train)
     if (cont==1):
        matriz = confusion_matrix(y_test, mlp.predict(numpy.array(X_test)))
     if (cont==2):
         matriz = somarMatrizes(confusion_matrix(y_test, mlp.predict(numpy.array(X_test))), matriz)
         print(matriz)
     cont = 2

print(matriz)


results = cross_val_score(mlp, X, y, cv=kf.get_n_splits(X))
print "Accuracy:", results.mean()


