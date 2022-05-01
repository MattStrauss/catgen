# @author: hare2; Bryce S.;
# Book Cover ML Algorithm by implementing pretrained model - FastAI

"""
Tutorials:
https://medium.com/swlh/judging-a-book-by-its-cover-the-deep-learning-way-94847c7c1274
https://www.analyticsvidhya.com/blog/2020/10/develop-and-deploy-an-image-classifier-app-using-fastai/

Install OLDER version of FastAI on Spyder (https://fastai1.fast.ai/install.html)
[make sure you have PyTorch]
 pip install fastai==1.0.61
[restart Spyder]

"""
import os

from fastai.vision import *
from pathlib import Path
from fastai.vision import ClassificationInterpretation
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def init_resnet():
    if __name__ == '__main__':
        print("Loading images...")
        #p_path = Path(r"dataset")
        df = pd.read_csv("modified_category.csv")
        df.head()
        
        img_size = 100
        
        np.random.seed(90)
        data = (ImageList.from_df(df, r"C:\Users\hare2\OneDrive\Desktop\ML & Data Sci\Final Project Demos\archive")
                            .split_by_rand_pct(0.2)
                            .label_from_df()
                            .transform(get_transforms(), size= img_size)
                            .databunch(bs=6)).normalize(imagenet_stats)
        data.show_batch()
        
        ############ Create Model ############
        from fastai.metrics import accuracy, Precision, Recall
        learn = cnn_learner(data, models.resnet34, metrics=[accuracy, Precision(average='macro'), Recall(average='macro')], opt_func=optim.SGD)
        
        #categories to predict
        data.classes
        
        #no of categories present
        data.c
        
        #length of training  & validation setset
        len(data.train_ds)
        len(data.valid_ds)

        #batch size is how many data points before model updates
        learn.data.batch_size = 50
        
        print("Training model...")
        
        #params are number of epochs and learning rate
        learn.fit(8, 0.005)
        
 
        print("Evaluating...")
        
        #root_path = os.path.join(app.root_path, "ml/")

        interp = ClassificationInterpretation.from_learner(learn)
    
        # confusion matrix: actual on left, predicted on right
        interp.plot_confusion_matrix(figsize=(17, 17))
        
        # print out the most mis-classified classes
        interp.most_confused()
    
        # plotting the top losses
        interp.plot_top_losses(6, figsize=(25,25))
        learn.export(root = Path("."))
    
def get_prediction(image_path):
    #root_path = os.path.join(app.root_path, "ml/")
    #learn_inf = load_learner(root_path/'export.pkl')
    learn_inf = load_learner("./")
    img = open_image(image_path)
    prediction = learn_inf.predict(img)
    print(prediction)
    return prediction

init_resnet()
#get_prediction(r"C:\Users\hare2\OneDrive\Desktop\CountMonteCristo.jpg")