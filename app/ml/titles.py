from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from app.ml.WordVectorTranformer import WordVectorTransformer
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.metrics import accuracy_score

# Load dataset
names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

validation_size = 0.20
X_train, X_test, Y_train, Y_test = train_test_split(dataset.title, dataset.category, test_size=validation_size,
                                                    random_state=1, shuffle=True)
#################################
# Accuracy Score 45.687298036291324 with WordVectorTransformer
# Accuracy Score 54.2629878200348 with TfidfVectorizer (and much faster)
text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SGDClassifier()),
])

text_clf.fit(X_train, Y_train)

predictions = text_clf.predict(X_test)

accuracy = accuracy_score(Y_test, predictions) * 100

print("Accuracy Score", accuracy)
#################################
