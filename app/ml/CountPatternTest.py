from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from CountPattern import cosine_similarity_func

# To test with top three categories set `top_three_run` to True and `maximum_run` = False
# To test with only one category (the top category returned) set `maximum_run` = True and `top_three_run` = False

names = ['title', 'category']
dataset = read_csv('data/titles_categories.csv', names=names)

# total_records = 20114
correct_count = 0
top_three_run = False
maximum_run = True
category_accuracy_dict = {}

# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c
vectorizer = CountVectorizer(stop_words='english', max_df=1)

# combine the titles from each category into a single string
previous_category = 0

# to speed up testing only take the first
# "x" number of titles per category
# ToDo this **should** be a randomized sample
limit_count_per_category = 15

# for determining the overall accuracy
total_records = limit_count_per_category * 17

for index, row in dataset.iterrows():

    # some titles will include only stop words and that
    # will throw an error, so let's catch it and continue
    try:
        vectorizer.fit_transform([row.title])
    except ValueError:
        continue

    # category accuracy dictionary builder
    # if new category, set up new key in category_accuracy_dict
    # for it with the category (int) as key
    category_count_key = str(row.category) + "_count"
    category_correct_key = str(row.category) + "_correct"
    if previous_category != row.category:

        category_accuracy_dict[category_count_key] = 1
        category_accuracy_dict[category_correct_key] = 0
        previous_category = row.category

    else:
        # speed up testing by limiting the number of titles checked in each category
        if category_accuracy_dict[category_count_key] >= limit_count_per_category:
            continue

        # increment the number of Titles in this category
        category_accuracy_dict[category_count_key] += 1

    # test with top three
    if top_three_run:
        guess_categories = cosine_similarity_func(row.title, top_three=True)
        # print("Top three guessed categories: ")
        # for i in guess_categories:
            # print(i)

        # print(" Actual category: " + str(row.category))
        if row.category in guess_categories:
            correct_count += 1
            category_accuracy_dict[category_correct_key] += 1

    # test with maximum
    if maximum_run:
        guess_category = cosine_similarity_func(row.title, maximum=True)
        # print("Guessed category: " + str(guess_category) + " Actual category: " + str(row.category))

        if row.category == guess_category:
            correct_count += 1
            category_accuracy_dict[category_correct_key] += 1

accuracy_score = correct_count / total_records

print("Accuracy = " + str(accuracy_score))

for category in category_accuracy_dict:
    category_number = category.replace("_count", "")
    if "count" in category:
        print("Category: " + category_number + " Number of Titles: " + str(category_accuracy_dict[category]))
    if "count" not in category:
        print("Number Correctly Guessed: " + str(category_accuracy_dict[category]))

