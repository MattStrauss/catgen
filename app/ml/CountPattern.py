from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv

# Load dataset
names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c
vectorizer = CountVectorizer(stop_words='english', max_df=1)
category_words_dict = {}

# combine the titles from each category then tokenize them
previous_category = 0
category_string = ""
for count, value in enumerate(dataset.category):

    # if new category, save current category string to dict
    if previous_category != value:

        previous_category = value

        # tokenize and build vocab for previous category
        if value > 1:
            vectorizer.fit([category_string])
            category_words_dict[value - 1] = vectorizer.transform([category_string])
            category_string = ""

    else:
        # concatenate the title into the category
        print(dataset.title[count])
        category_string = category_string + " " + dataset.title[count]

for i, count in enumerate(category_words_dict):
    print(category_words_dict[count].toarray())
