from dual_perceptron import DualPerceptron
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/name/Downloads/moonDataset.csv')
y = df['label'].to_numpy()
y[y==0] = -1
X = df.drop('label', axis=1)
X = X.to_numpy()

model = DualPerceptron(kernel='rbf', patience=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train, X_test, y_test)

print(model.get_accuracy())
