
# Folders

## training_scripts
There are 2 Jupyter notebook files in the **training scripts** folder. If you just run the cells from top to bottom, the models will be trained by the code
1) train.ipynb contains code for training the individual pre-trained models
2) train_ensemble.ipynb contains code for training the ensemble models

## models
When you run the training scripts, it trains models and saves them in this folder in the H5 format. These are used by the python flask webapp to detect dog breeds in images uploaded by users.
I couldn't upload all the saved models here because they're very large ( >5 GB in size). But you can run the scripts to produce them though
I put one small model in this folder though just in case anyone wants to quickly test out the web application.

## webapp
Contains the webapp code. There are 2 files of importance
1) **app.py** is the backend script that recieves XHR requests from the frontend, processes the image uploaded by the user and returns a response containing the detected dog breed
2) **index.html** file in **templates** folder contains the front end code'

To run the web app, simply run the following command
```
python <path_to_app,py>
```

For example
```
python "C:\Thesis\webapp\app.py"
```

## images
This folder contains the dataset. There are two subfolders
1) **train** - contains training images
2) **val** - contains images used for validation after each training epoch is finished

Dataset was split 80-20 between training and validation using the **split_data.py** script in the training_scripts folder




   
