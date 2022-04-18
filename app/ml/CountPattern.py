from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv
from sklearn.metrics.pairwise import euclidean_distances
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

        # avoid doing this on the first iter otherwise we get
        # key error (0)
        if value > 1:
            # save concatenated string to dict
            category_words_dict[previous_category] = category_string
            # reset string with first string of next category
            category_string = dataset.title[count]

        previous_category = value

    else:
        # concatenate the title into the category
        category_string = category_string + " " + dataset.title[count]


# https://www.machinelearningplus.com/nlp/cosine-similarity/
def cosine_similarity_func(title, top_three=False):
    """
    Input:
        A: a String which corresponds to a Title
    Output:
        if 'top_three = false` (default) -> cosine_values_dict: dictionary representing the
        cosine similarity of the Title in each Category.
        if 'top_three  = true` -> return the maximum three int values of cosine_values_dict.
        (This is largely used for Testing)
        - If they are the total opposite, meaning, A = -B, then you would get -1.
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

    cosine_values_dict = {}

    for _, key in enumerate(category_words_dict):
        # turn the category string into a numeric vector
        vectorized_category = vectorizer.transform([category_words_dict[key]])

        # the returned results are a nested array, the [0][0] on the end gets just the value
        cosine_val = cosine_similarity(vectorized_title, vectorized_category)[0][0]

        cosine_values_dict[key] = cosine_val

    if top_three:
        # return top three
        return sorted(cosine_values_dict, key=cosine_values_dict.get, reverse=True)[:3]

    return cosine_values_dict


# test
# returns

# Load dataset
names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

total_records = 20114
correct_count = 0
skip_count = 0

for index, row in dataset.iterrows():
    try:
        vectorizer.fit_transform([row.title])
    except ValueError:
        skip_count += 1
        continue
    guess_categories = cosine_similarity_func(row.title, True)
    print("Top three guessed categories: ")
    for i in guess_categories:
        print(i)

    print(" Actual category: " + str(row.category))
    if row.category in guess_categories:
        correct_count += 1

accuracy_score = correct_count / (total_records - skip_count)

print("Accuracy = " + str(accuracy_score))
