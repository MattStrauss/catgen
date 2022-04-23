import matplotlib.pyplot as plt
from pandas import read_csv
import pickle
import os
from app import app

# list that will hold image data, categories, etc.
data_list = []


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
        data_list.insert(0, [vector, img.shape, row.category])


load_data_set()

# save generated data_list to pickle file
with open('datalist.pickle', 'wb') as output:
    pickle.dump(data_list, output)

# load data from pickle file
with open('datalist.pickle', 'rb') as data:
    loaded_data = pickle.load(data)

for index in loaded_data:
    print(index)

# you can recover the original image with:
# rows, cols, colors = img.shape
# recovered_image = vector.reshape(rows,cols,colors)
# plt.imshow(recovered_image) # followed by
# plt.show() # to show you the second image.
