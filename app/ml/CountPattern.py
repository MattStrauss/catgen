from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c
vectorizer = CountVectorizer(stop_words='english', max_df=1)
category_words_dict = {}

# combine the titles from each category into a single string
previous_category = 0
category_string = ""
for count, value in enumerate(dataset.category):

    # if new category, save current category string to dict
    # with category (int) as key
    if previous_category != value:

        previous_category = value

        # avoid doing this on the first iter otherwise we get
        # key error (0)
        if value > 1:
            # save concatenated string to dict
            category_words_dict[value - 1] = category_string
            # reset string with first string of next category
            category_string = dataset.title[count]

    else:
        # concatenate the title into the category
        category_string = category_string + " " + dataset.title[count]


# https://www.machinelearningplus.com/nlp/cosine-similarity/
def cosine_similarity_func(title):
    """
    Input:
        A: a String which corresponds to a Title
    Output:
        cos: numerical number representing the max cosine similarity Title and Categories.
    cos value interpretations
        - If they are the total opposite, meaning, A = -B , then you would get -1.
        - If you get 0, that means that they are orthogonal (or perpendicular).
        - Numbers between 0 and 1 indicate a similarity score.
        - Numbers between -1 to 0 indicate a dissimilarity score.
    Concept:
        Because the category strings are much larger than the Title, euclidian distance is not
        appropriate in this case. But, cosine similarity is a good way to normalize the vectors
        into a normalized unit where we can compare their projection angles into n-dimensional space
    """

    # turn the title into a numeric vector
    vectorized_title = vectorizer.fit_transform([title])

    # default, worst case
    cos = -1

    for _, key in enumerate(category_words_dict):

        # turn the category string into a numeric vector
        vectorized_category = vectorizer.transform([category_words_dict[key]])

        cos = cosine_similarity(vectorized_title, vectorized_category)

        print("Category: " + str(key) + " Cosine Similarity = " + str(cos))

    return cos


# test
# returns
cosine_similarity_func("Counting for Kids")

