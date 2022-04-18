from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv
from sklearn.metrics.pairwise import euclidean_distances

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
def euclidian_distance_func(title, top_three=False):
    """
    Input:
        A: a String which corresponds to a Title
    Output:
        if 'maximum = false` (default) -> euclid_values_dict: dictionary representing the euclidian distance
        of the Title in each Category. This is largely used for Testing with testing data
        if 'maximum = true` -> return the int value of the maximum Category
    euclid value interpretations
        - the larger the value, the more likely the Title is to be in that category
    Concept:
        I can't actually explain it at the moment, but after testing with cosine_similarity did
        not yield good results. So, I used euclidian distance just to get an idea of what was going
        wrong with the cosine_similarity, and it tuned out the maximum euclidian distance yields the
        most accurate results. This doesn't quite make sense as it's counterintuitive, but we can think
        on it for a while and figure out why this is happening?!?
    """

    # turn the title into a numeric vector
    vectorized_title = vectorizer.fit_transform([title])

    euclid_values_dict = {}

    for _, key in enumerate(category_words_dict):
        # turn the category string into a numeric vector
        vectorized_category = vectorizer.transform([category_words_dict[key]])

        # the returned results are a nested array, the [0][0] on the end gets just the value
        euclid = euclidean_distances(vectorized_title, vectorized_category)[0][0]

        euclid_values_dict[key] = euclid

    # print("Category: " + str(key) + " Euclidian Distance = " + str(euclid))

    if top_three:
        # return top three
        return sorted(euclid_values_dict, key=euclid_values_dict.get, reverse=True)[:3]

    return euclid_values_dict


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
    guess_categories = euclidian_distance_func(row.title, True)
    print("Top three guessed categories: ")
    for i in guess_categories:
        print(i)

    print(" Actual category: " + str(row.category))
    if row.category in guess_categories:
        correct_count += 1

accuracy_score = correct_count / (total_records - skip_count)

print("Accuracy = " + str(accuracy_score))
