import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
dataset = pd.read_csv("mammographic_masses_classficacao.data.csv")
dataset = dataset.replace("?", -1)
X = dataset.iloc[:,:4]
y = dataset.iloc[:,5]
X_train, X_test, y_train, y_test = train_test_split(X, y)
seed = 7
numpy.random.seed(seed)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
      beta_2=0.999, early_stopping=False, epsilon=1e-08,
      hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
      learning_rate_init=0.001, max_iter=200, momentum=0.9,
      nesterovs_momentum=True, power_t=0.5, random_state=None,
      shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
      verbose=False, warm_start=False)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(mlp, X, y, cv=kfold)
print(results.mean())


predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))