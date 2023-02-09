import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('glass.csv')

x_train = df.iloc[:,:-1].values
y_train = df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=4)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

print(classification_report(y_test, y_pred, zero_division = 0))
print("Naive Bayes accuracy is: ",  (accuracy_score(y_test, y_pred))*100)
