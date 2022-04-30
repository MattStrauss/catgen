# @author: hare2; Bryce S.;
# Book Cover ML Algorithm by implementing pretrained model - FastAI

"""
Tutorial here (may use to enable GPU)
https://medium.com/swlh/judging-a-book-by-its-cover-the-deep-learning-way-94847c7c1274

Install OLDER version of FastAI on Spyder (https://fastai1.fast.ai/install.html)
[make sure you have PyTorch]
 pip install fastai==1.0.61
[restart Spyder]


Keep getting this error but ignoring it: serWarning: torch.solve is deprecated in favor of torch.linalg.solveand will be removed in a future PyTorch release.
torch.linalg.solve has its arguments reversed and does not return the LU factorization.
To get the LU factorization see torch.lu, which can be used with torch.lu_solve or torch.lu_unpack.
"""

from fastai.vision import *
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#p_path = Path(r"dataset")
df = pd.read_csv("modified_category.csv")
df.head()

img_size = 256

np.random.seed(90)
data = (ImageList.from_df(df, "./")
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

#length of training set
len(data.train_ds)

#length of validation set
len(data.valid_ds)

############ Train Model ############
# Build the CNN model with the pretrained resnet34
# Error rate = 1 - accuracy

#EDIT NUMBER OF EPOCHS, CAN ALSO ADD A LEARNING RATE (e.g. lr = 0.003)
# Train the model on # epochs of data at the default learning rate
learn.data.batch_size = 50
learn.unfreeze()

learn.fit(10, 0.005)

learn.save('res34-stage1')

############ Evaluate Model ############
interp = ClassificationInterpretation.from_learner(learn)

# confusion matrix: actual on left, predicted on right
interp.plot_confusion_matrix(figsize=(17, 17))


# print out the most mis-classified classes
interp.most_confused()

# plotting the top losses
interp.plot_top_losses(6, figsize=(25,25))
