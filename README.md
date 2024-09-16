
# Folders

## training_scripts
There are 2 Jupyter notebook files in the **training scripts** folder. If you just run the cells from top to bottom, the models will be trained by the code
1) **train.ipynb** contains code for training the individual pre-trained models
2) **train_ensemble.ipynb** contains code for training the ensemble models

## models
When you run the training scripts **train.ipynb** or **train_ensemble.ipynb**, they'll train models and save them inside this folder. These saved models are used by the python flask webapp to detect dog breeds in images uploaded by users.

I couldn't upload all the saved models here because they're very large ( >5 GB in size). But you can run the training scripts to produce them.

If you want the saved models without running train.ipynb and train_ensemble.ipynb, I've uploaded a zip file on my google drive with all the models ready to use.
You can download it from here: https://drive.google.com/file/d/1MhQgN23XxUVjL36I93Z9B3UAl-SiV5nS/view?usp=sharing
Just unzip the models.zip file and copy all the .h5 files from the extracted folder into the **models** folder inside the repository.
Then run the web app and all models from the folder should be available to use for breed detection

## webapp
Contains the webapp code. There are 2 files of importance
1) **app.py** is the backend script that recieves XHR requests from the frontend, processes the image uploaded by the user and returns a response containing the detected dog breed
2) **index.html** file in **templates** folder contains the front end code'

To run the web app, simply run the following command
```
python <path_to_app.py>
```

For example
```
python "C:\Thesis\webapp\app.py"
```

Then visit the following URL in your web browser 
```
http://localhost
```

## images
This folder contains the dataset. There are two subfolders
1) **train** - contains training images
2) **val** - contains images used for validation after each training epoch is finished

Dataset was split 80-20 between training and validation using the **split_data.py** script in the training_scripts folder

# Online Deployment
I've deployed the web app online on a Microsoft Azure Compute VM instance.
The app can be accessed on the URL: http://jobinjosethesis.xyz/




   
