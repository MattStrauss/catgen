import cv2
from sklearn.neighbors import NearestNeighbors
from SimplePreprocessor import SimplePreprocessor
import csv
import pickle


def get_neighbors(image):
    # load image data from pickle file
    with open('datalist.pickle', 'rb') as data:
        loaded_data = pickle.load(data)

    # ML Model (get three closest neighbors
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(loaded_data)

    # size to reduce image to (must match size in preprocessImages file to be accurate)
    img_height = 50
    img_width = 30

    # init the SimplePreprocessor
    sp = SimplePreprocessor(img_height, img_width)

    # turn the user provided image into a vector and resize it
    image_vector = cv2.imread("../uploads/" + image)
    resized_image = sp.preprocess(image_vector)

    # reshape to match the other images shame from the preprocessImages file)
    image_matrix = resized_image.reshape((1, img_height * img_width * 3))

    # run the user provided, now preprocessed image through the NN algorithm
    found_neighbors = neigh.kneighbors(image_matrix)
    row_numbers = found_neighbors[1][0]  # [7453 2753 1353]

    # use the indices returned to get the associated url and label from the csv
    with open('url_and_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row_number in row_numbers:
            print(rows[row_number])


get_neighbors("clockwork_orange.jpeg")
