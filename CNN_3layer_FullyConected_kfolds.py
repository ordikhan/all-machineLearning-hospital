import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_excel("heart.xlsx")

print(df.head())

data = df.to_numpy()

X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = data[:, -1]

kfold_n_split = 10
kf = KFold(n_splits=kfold_n_split, shuffle=True, random_state=2)
kfold_get = kf.split(X)
list = []
for j in range(kfold_n_split):
    print("fold: ", j)
    result = next(kfold_get)
    X_train = X[result[0]]
    X_test = X[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200, batch_size=100, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    auc_score = auc(fpr, tpr)

    print("for fold :", j, "  auc = ", auc_score)
    list.append(auc_score)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
print(sum(list) / 10)
