import cv2
import numpy as np
import pandas as pd


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
        df = pd.read_csv("modified_category.csv")

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(df['category'].iloc[i])

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO]: Processed {}/{}'.format(i + 1, len(image_paths)))

        return np.array(data), np.array(labels)
