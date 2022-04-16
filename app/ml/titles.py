from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from app.ml.WordVectorTranformer import WordVectorTransformer
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.metrics import accuracy_score

# Load dataset
names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

array = dataset.values
X = array[:, 0:1]
Y = array[:, 1]
validation_size = 0.20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size,
                                                    random_state=1, shuffle=True)

text_clf = Pipeline([
    ('vect', WordVectorTransformer()),
    ('clf', SGDClassifier()),
])

text_clf.fit(X_train, Y_train)

predictions = text_clf.predict(X_test)

accuracy = accuracy_score(Y_test, predictions) * 100

print("Accuracy Score", accuracy)
