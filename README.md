## Predicting Pokémon: A use case for rapid ideation on character design

Phase 1: Gotta classify them all

### Data Sourcing Content

Data was sourced from Kaggle and can be found here: https://www.kaggle.com/vishalsubbiah/Pokémon-images-and-types

Dataset contain all Pokémon from Generation 1 to 7, and consists of 809 unique images

<p align='center'>
<img src='images/some_pokemon.png'>
</p>

### Business Case

Design and innovation firms are asked to come up with mock ups and pitches in a fast moving environment.  They also require in-house or freelance resources that are taken away from current paying work. A GAN (Generative Adversarial Network) could be used to generate images quickly that can be used for rapid character prototyping so that the design team can iterate on concepts in a more efficient manner.

Phase 1 of this project consists of classifying whether a given Pokémon has evolved using a Convolutional Neural Network.


This process will in turn enable Phase 2 of the project, which is the creation of a GAN using the same dataset with a goal of producing novel Pokémon.


### Data Cleaning/Visualization

Datasets were built by hand - sorting evolved and not evolved into two separate folders.


Images were reproduced 7 times in order to create the training data resulting in a final total of 3120 EVOLVED Pokémon and 3369 NOT_EVOLVED Pokémon


As a result of this approach, the low class imbalance was retained:

<p align='center'>
<img src='images/class_imbalance.png'>
</p>


If using a Dummy Classifier, the dominant class will be predicted 52% of the time.


Images were converted from PNG to JPG to remove the transparency layer and make the images viable for the models:

<p align='center'>
<img src='images/jpg_version.png'>
</p>

All images are uniform in size: 120x120

### Models:

## Base CNN Model

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 222, 222, 64)      1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 111, 111, 64)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 109, 109, 32)      18464     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 54, 54, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 93312)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                2986016   
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056      
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 3,007,361
Trainable params: 3,007,361
Non-trainable params: 0
_________________________________________________________________

Summary:

<p align='center'>
<img src='images/base_loss.png'>
</p>

<p align='center'>
<img src='images/base_accuracy.png'>
</p>


<p align='center'>
<img src='images/base_results.png'>
</p>


#### Accuracy: 74%
#### Loss: 0.5407

* Optimizer: Adam
* Learning Rate: 0.001
* Steps per epoch: 300

#### Better than the baseline metric (52%)


## Transfer Learning with VGG16

Context: VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”.

Model: "model_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
global_average_pooling2d_13  (None, 512)               0         
_________________________________________________________________
dense_21 (Dense)             (None, 1024)              525312    
_________________________________________________________________
dense_22 (Dense)             (None, 512)               524800    
_________________________________________________________________
dense_23 (Dense)             (None, 1)                 513       
=================================================================
Total params: 15,765,313
Trainable params: 1,050,625
Non-trainable params: 14,714,688
_________________________________________________________________

Summary:

<p align='center'>
<img src='images/model_loss.png'>
</p>

<p align='center'>
<img src='images/model_accuracy.png'>
</p>


<p align='center'>
<img src='images/model_results.png'>
</p>


#### Accuracy: 93%
#### Loss: 0.1650

* Optimizer: Adam
* Learning Rate: 0.0001
* Steps per epoch: 300

#### Significantly better than the baseline metric (52%)


### Conclusion

The VGG16 model worked significantly better than the baseline model.  This means that the data used to train the model is viable for using in a GAN to produce novel Pokémon in Phase 2 of the project.

Next steps:

* Take this dataset and build a GAN.
* Set up cloud computing for GPU powered training and generative output.
* Create custom tuning parameters according to Pokémon class.
* Build GUI that allows user to catch their very own Pokémon.


### Sources

* https://pokemondb.net/evolution
* https://medium.com/@shikharsrivastava_14544/face-recognition-using-transfer-learning-with-vgg16-3caeca4a916e
* https://www.freecodecamp.org/news/how-to-build-an-image-type-convertor-in-six-lines-of-python-d63c3c33d1db/




