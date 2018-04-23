# Unet-Semantic-Segmentation
This repository contains an implementation of applying Unet to image semantic segmentation. 

U-net architecture can be illustrated as below: 

<img src="u-net-architecture.png" width="500">

(Image is cited from: Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation",Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015
)

Run 'main.py --use_gpu --optimizer sgd --unet_channels 32 --unet_batch_norm --learning_rate 0.01 --save_model_every 500' to train the model on training set. The model is saved every 500 iterations. Make sure to load the latest model when make prediction on test images.  

The training/validation images need to be download under data/train. 

After training on training image sets, the model can make predictions on the testing data. 
The test images need to be download under data/test

Run 'main.py --predict --load_model --initial_checkpoint your_model_name --out_dir prediction_loc' to predict testing data with trained model. 

Examples of applying a well-trained model on testing images: 

<img src="predictions.png" width="300">

