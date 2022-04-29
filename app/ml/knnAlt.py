# -*- coding: utf-8 -*-
"""
Tutorial: https://colab.research.google.com/github/vinay10949/AnalyticsAndML/blob/master/KNN.ipynb#scrollTo=fcCvmbWGSDhk
Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

SET UP:
    1. Download images from https://www.kaggle.com/lukaanicin/book-covers-dataset
    2. Put in same folder as code, rename  master folder to "dataset"
        For example: "dataset/Humour/0000805.jpg")
    3. Make sure modified_category.csv is in same folder as project
    4. Can't remember if you need to install anything else, but I don't believe so

NOTE:
Can change img_size on line #123 to try different resolutions (could improve accuracy but increase training time?)

Can swap out ML models to any others in sklearn very easily on line #160 model = ???
Currently set to Random Forrest Classifier because that was getting best results
I think more n_estimators is better, but can slow down processing, lots of other parameters to tweak as well
For KNN, could use num neighbors or radius (see commented out lines #158 and #159)

Gets image address from csv (local directories), finds that image, pairs to label in CSV
-Takes a few minutes
-Does not use folder names as categories (if you use folder names, will need to map to the different category set we are using for titles)
-This is different than what the tutorial is doing with google drive images
-MASTER DATA has urls for each image if you want to try that way

Tutorial has a section on how to speed this up, not implemented here

Tutorial also has section on finding the best number for k neighbors

I believe there is a way to add several variations of each image that could improve accuracy
-like stretched, offset, flipped, etc...

Interpreting metrics
-Random Forest Classifier has high(ish) precision but low recall, abysmal f-scores
-"returning very few results, but most of its predicted labels are correct when compared to the training labels"
-https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=A%20system%20with%20high%20precision,with%20all%20results%20labeled%20correctly.
-Sample categories are uneven sizes (e.g. #17 is a mash up of several 900 item original categories)


TO DO:
    1. Does not save anywhere (maybe use Pickle?)
    2. Does not have any code to import a single image and return a label
    3. Does not have any code to show images of the neighbors for KNN (may need to "unflatten" image in data set to show it)
    4. Not sure what Random Forest Classifier visualization would be
    5. Not using cross validation
"""
import os

import cv2

from app import app
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from SimplePreprocessor import SimplePreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
import pandas as pd
from matplotlib import pyplot as plt
import csv
import pickle


def get_neighbors(image):
    # load data from pickle file
    with open('datalist.pickle', 'rb') as data:
        loaded_data = pickle.load(data)

    # ML Model
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(loaded_data)

    full_image_address = os.path.join(app.config['UPLOAD_FOLDER'], image)

    img_height = 50
    img_width = 30

    sp = SimplePreprocessor(img_height, img_width)

    image_vector = cv2.imread("../uploads/" + image)
    resized_image = sp.preprocess(image_vector)

    image_matrix = resized_image.reshape((1, img_height * img_width * 3))

    found_neighbors = neigh.kneighbors(image_matrix)
    row_numbers = found_neighbors[1][0]  # [7453 2753 1353]
    with open('url_and_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row_number in row_numbers:
            print(rows[row_number])


get_neighbors("clockwork_orange.jpeg")
