import os

import cv2
from sklearn.neighbors import NearestNeighbors

from app import app
from app.ml.SimplePreprocessor import SimplePreprocessor
import csv
import pickle


def get_neighbors(image, num_neighbors):
    return_list = []
    root_path = os.path.join(app.root_path, "ml/")
    uploads_path = os.path.join(app.root_path, "static/uploads/")

    # load image data from pickle file
    with open(root_path + "datalist.pickle", 'rb') as data:
        loaded_data = pickle.load(data)

    # ML Model (get three closest neighbors
    neigh = NearestNeighbors(n_neighbors=num_neighbors)
    neigh.fit(loaded_data)

    # size to reduce image to (must match size in preprocessImages file to be accurate)
    img_height = 50
    img_width = 30

    # init the SimplePreprocessor
    sp = SimplePreprocessor(img_height, img_width)

    # turn the user provided image into a vector and resize it
    image_vector = cv2.imread(uploads_path + image)
    resized_image = sp.preprocess(image_vector)

    # reshape to match the other images shame from the preprocessImages file)
    image_matrix = resized_image.reshape((1, img_height * img_width * 3))

    # run the user provided, now preprocessed image through the NN algorithm
    found_neighbors = neigh.kneighbors(image_matrix)
    row_numbers = found_neighbors[1][0]  # [7453 2753 1353]

    # use the indices returned to get the associated url and label from the csv
    with open(root_path + 'url_and_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row_number in row_numbers:
            return_list.append(rows[row_number])

    return return_list
