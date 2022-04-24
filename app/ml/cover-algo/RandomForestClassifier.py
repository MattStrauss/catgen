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

########################## IMAGE PREPROCESSING ##########################
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

class SimpleDatasetLoader:
    # Method: Constructor
    def __init__(self, preprocessors=None):
        """
        :param preprocessors: List of image preprocessors
        """
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []


    # Method: Used to load a list of images for pre-processing
    def load(self, image_paths, verbose=-1):
        """
        :param image_paths: List of image paths
        :param verbose: Parameter for printing information to console
        :return: Tuple of data and labels
        """
        data, labels = [], []

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)

            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(df['category'].iloc[i])

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print('[INFO]: Processed {}/{}'.format(i+1, len(image_paths)))

        return (np.array(data), np.array(labels))
    
#Class Preprocessror 
class SimplePreprocessor:
    # Method: Constructor
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """
        :param width: Image width
        :param height: Image height
        :param interpolation: Interpolation algorithm
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation

    # Method: Used to resize the image to a fixed size (ignoring the aspect ratio)
    def preprocess(self, image):
        """
        :param image: Image
        :return: Re-sized image
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
    
####################### LOAD DATA SET#######################
from imutils import paths
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from __main__ import SimplePreprocessor
from __main__ import SimpleDatasetLoader
import pandas as pd

df = pd.read_csv("modified_category_modified.csv")
image_paths = list(df["image"].tolist())

img_size = 50
num_samples = 20113 # Could grab this from CSV instead of hard coding?

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
sp = SimplePreprocessor(img_size, img_size)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)

# Reshape, sklearn expects 2d array (num samples, image height, image width, 3 rgb matrices)
# Currently have a list of ~20,000 data points. Each has 3 height x width arrays for rgb
# I believe you can reshape to "unflatten" later if needed
data = data.reshape((num_samples, img_size * img_size* 3))
print(data.shape)


# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024*1000.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#######################BUILD ML MODEL#######################
# Train and evaluate the k-NN classifier on the raw pixel intensities
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

test_proportion = 0.25

# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=test_proportion, random_state=42)

# for dimension in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
#     for tree in [50, 125, 200, 250, 500, 750, 1000, 1500]:

# pca = PCA(n_components=100)
# print('fitting')
# pca.fit(train_x)
# print('transforming')
# train_x = pca.transform(train_x)
# test_x = pca.transform(test_x)

# ML Models  
print('[INFO]: Classification starting....')
# model = KNeighborsClassifier(radius = 0.1,n_jobs=-1, weights ='distance')
# model = KNeighborsClassifier(n_neighbors =100,n_jobs=-1)
# model = RandomForestClassifier(n_estimators = 125,  n_jobs = -1,random_state =50)
model = LogisticRegression(random_state=12)

#######################TRAIN, PREDICT, ACCURACY#######################
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print(classification_report(test_y, prediction,target_names=le.classes_))