from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
cancer = load_breast_cancer()
cancer.keys()
import pandas as pd
dataset = pd.read_csv("mammographic_masses_classficacao.data.csv")
# separate the data from the target attributes
dataset = dataset.replace("?", -1)
X = dataset.iloc[:,:4]
y = dataset.iloc[:,5]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.66)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
mlp.fit(X_train,y_train)


predictions = mlp.predict(X_test)
print "Acuracia:", mlp.score(X_test, y_test)
matriz = confusion_matrix(y_test,predictions)
print"Matriz de confusao: ",(matriz)
sensitivity = matriz[1][1] / float(matriz[1][0] + matriz[1][1])

print"Sensitivity: ",(sensitivity)

specificity = matriz[0][0] / float(matriz[0][0]+ matriz[0][1])

print"Specificity: ",(specificity)
## The line / model
# plt.scatter(y_test, predictions)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()


