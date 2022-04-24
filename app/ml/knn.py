import matplotlib.pyplot as plt
from pandas import read_csv
import pickle
import os
from app import app
import numpy as np

# list that will hold image data, categories, etc.
data_list = []
pca_number = 15000

def load_data_set():
    """
    This should only happen once. We run through the entire dataset and store the created vectors, size and categories
    into a list, which is saved to a `pickle` file. Then we load that data from the pickle file when needed to run the
    KNN algorithm on uploaded images to compare them to the vector dataset to find the nearest neighbors
    """

    dataset_path = os.path.join(app.root_path, "ml/data/small_image_category_data.csv")
    names = ['image', 'category']
    dataset = read_csv(dataset_path, names=names)

    for i, row in dataset.iterrows():
        img = plt.imread(str(row.image), 'jpg')
        rows, cols, colors = img.shape  # get dimensions for RGB array
        size = rows * cols * colors
        vector = img.reshape(size)
        data_list.insert(0, [vector, str(row.image), row.category])


load_data_set()

# save generated data_list to pickle file
with open('datalist.pickle', 'wb') as output:
    pickle.dump(data_list, output)

# load data from pickle file
with open('datalist.pickle', 'rb') as data:
    loaded_data = pickle.load(data)


# turn the uploaded image into a vector for comparison
def vectorize_image(image):
    full_image_address = os.path.join(app.config['UPLOAD_FOLDER'], image)
    img = plt.imread(str(full_image_address), 'jpg')
    rows, cols, colors = img.shape  # get dimensions for RGB array
    size = rows * cols * colors
    return img.reshape(size)


# calculate and return the Euclidean distance of
# two numPy arrays
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum(np.square(row1 - row2)))


# Locate the most similar neighbors
def get_neighbors(uploaded_image, num_neighbors):
    distances = list()
    for row, count in enumerate(loaded_data):
        vectorized_uploaded_image = vectorize_image(uploaded_image)
        pca_uploaded_image = vectorized_uploaded_image[:pca_number]
        current_row = loaded_data[row][0][:pca_number]
        distance = euclidean_distance(current_row, pca_uploaded_image)
        distances.append((loaded_data[row], distance))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
