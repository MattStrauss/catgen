from SimplePreprocessor import SimplePreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
import pandas as pd
import pickle


def load_data():
    """
    This should only happen once. We run through the entire dataset and store the created vectors, size and categories
    into a list, which is saved to a `pickle` file. Then we load that data from the pickle file when needed to run the
    KNN algorithm on uploaded images to compare them to the vector dataset to find the nearest neighbors

    This only runs once, so we did not commit the full image folder to source control. In order to run it, be sure
    to follow the below three steps beforehand

    1. Download images from https://www.kaggle.com/lukaanicin/book-covers-dataset
    2. Put in same folder as code, rename  master folder to "dataset"
        For example: "dataset/Humour/0000805.jpg")
    3. Make sure modified_category.csv is in same folder as project
    """
    df = pd.read_csv("modified_category.csv")
    image_paths = list(df["image"].tolist())

    img_height = 50
    img_width = 30
    num_samples = 15397  # Could grab this from CSV instead of hard coding?

    # Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
    print('[INFO]: Images loading....')
    sp = SimplePreprocessor(img_height, img_width)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(image_paths, verbose=500)

    # Reshape, sklearn expects 2d array (num samples, image height, image width, 3 rgb matrices)
    # Currently have a list of ~20,000 data points. Each has 3 height x width arrays for rgb
    # I believe you can reshape to "unflatten" later if needed
    return data.reshape((num_samples, img_height * img_width * 3))


returned_data = load_data()

# save generated data to pickle file
with open('datalist.pickle', 'wb') as output:
    pickle.dump(returned_data, output)
