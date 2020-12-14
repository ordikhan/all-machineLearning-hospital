import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("heart.xlsx")

print(df.head())

data = df.to_numpy()

X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = data[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape

model = RandomForestClassifier(max_depth=3, random_state=0)
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)
print("auc = ",auc_score)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()