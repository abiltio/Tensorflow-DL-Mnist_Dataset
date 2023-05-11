```python
#!pip install tensorflow_datasets
```

# Deep NN for MNIST Classification
The dataset is called MNIST and refers to handwritten digit recognition.   

The dataset provides 70.000 images (28x28pixels) of handwritten digit (1 digit per image)]  

The goal is to write an algorithm that detecs which digit is written. Since there are only 10 digit(0-9), this is a   classification problem with 10 classes. 

Our goal would be to build a neural network with 2 hidden layers.  

## The action Plan
1. Prepare our data and preprocess it. Create training, validation, and test dataset
2. Outline the model and choose activation function
3. Set appropriate advance optimizer and loss function
4. Make it Learn
5. Test the accuracy

## Import the relevant libraries


```python
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Data


```python
# tfds.load actually loads a dataset (or downloads and then loads if that's the first time you use it) 
# mnist_dataset = tfds.load(name='mnist', as_supervised=True)
mnist_dataset, mnist_info = tfds.load(name='mnist',with_info=True, as_supervised = True)
# with_info=True will also provide us with a tuple containing information about the version, features, number of samples
# as_supervised=True will load the dataset in a 2-tuple structure (input, target) 
# alternatively, as_supervised=False, would return a dictionary

mnist_train, mnist_test = mnist_dataset['train'],mnist_dataset['test']

# normally, we would like to scale our data in some way to make the result more numerically stable
# in this case we will simply prefer to have inputs between 0 and 1
# let's define a function called: scale, that will take an MNIST image and its label
num_validation_samples = 0.1*mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)


def scale(image,label):
    # we make sure the value is a float
    image = tf.cast(image,tf.float32)
    # since the possible values for the inputs are 0 to 255 (256 different shades of grey)
    # if we divide each element by 255, we would get the desired result -> all elements will be between 0 and 1 
    image /= 255.
    return image, label

# the method .map() allows us to apply a custom transformation to a given dataset
# we have already decided that we will get the validation data from mnist_train, so 
scaled_train_and_validation_data = mnist_train.map(scale)

# finally, we scale and batch the test data
# we scale it so it has the same magnitude as the train and validation
# there is no need to shuffle it, because we won't be training on the test data
# there would be a single batch, equal to the size of the test data
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000
shuffle_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
# When we're dealing with enormous datasets, we can't shuffle all data at once
#BUFFER_SIZE : TF take n-amount of sample data and shuffle them at once and then take another n-amount
#IF BUFFER_SIZE >= num_samples, shuffling will happen at once (uniformly)
#IF 1<BUFFER_SIZE<num_samples, we will be optimizing computational power

validation_data = shuffle_train_and_validation_data.take(num_validation_samples)
train_data = shuffle_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 150
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))
#The next() loads the next element of an iterable object
#iter() create an object which can be iterated one element at a time (e.g. in for loop or while loop) 
```

# Model

## Outline the model


```python
input_size = 784 #from 28x28 pixel (size of each input data)
output_size = 10 #from 10 digits input (number range from 0-9)
hidden_layer_size = 300 #hyperparameter (pre-set by us)

#tf.keras.Sequential() function that is laying down the model (used to 'stack layers')
#our data (from tfds) is such that each output is 28x28x1 
#tf.keras.layers.Flatten(original shape) transform (flattens) a tensor into a vector
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape = (28,28,1)),
                           #tf.keras.layers.Dense(output_size) calculates the dot product of the inputs and the w and b 
                           
    # The rectified linear activation function or ReLU for short is a piecewise linear function 
    # that will output the input directly if it is positive, otherwise, it will output zero.
    # As a consequence, the usage of ReLU helps to prevent 
        # - the exponential growth in the computation required to operate the neural network.
        # - If the CNN scales in size, the computational cost of adding extra ReLUs increases linearly.
    
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            
                         
                            #we want to use 5 hidden layers
                            tf.keras.layers.Dense(output_size, activation = 'softmax'),
                            # when we're creating a classifier :
                            # activation function of the output must transform the value of probability, so we use softmax
    
                            ])
```

## Choose the optimizer and the loss function


```python
# model.compile(optimizer,loss,metrics) configures the model for training
# adam : adaptive moment estimation optimizer
# loss function for classifier normally use cross entropy

# there are 3 types build-in variation 
# binnary_crossentropy() : Binary encoding
# categorical_crossentropy() : Expects that you've one-hot encoded the targets
# sparse_categorical_crossentropy() : applies one-hot encoding

# The output and target layers should have matching forms 


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```

## Training


```python
NUM_EPOCHS = 5
model.fit(train_data, epochs = NUM_EPOCHS, validation_data = (validation_inputs, validation_targets),verbose =2 )
```

    Epoch 1/5
    360/360 - 3s - loss: 0.2688 - accuracy: 0.9179 - val_loss: 0.1162 - val_accuracy: 0.9647 - 3s/epoch - 9ms/step
    Epoch 2/5
    360/360 - 2s - loss: 0.1044 - accuracy: 0.9683 - val_loss: 0.1136 - val_accuracy: 0.9640 - 2s/epoch - 7ms/step
    Epoch 3/5
    360/360 - 2s - loss: 0.0734 - accuracy: 0.9776 - val_loss: 0.0767 - val_accuracy: 0.9765 - 2s/epoch - 7ms/step
    Epoch 4/5
    360/360 - 2s - loss: 0.0568 - accuracy: 0.9823 - val_loss: 0.0766 - val_accuracy: 0.9773 - 2s/epoch - 7ms/step
    Epoch 5/5
    360/360 - 2s - loss: 0.0453 - accuracy: 0.9860 - val_loss: 0.0736 - val_accuracy: 0.9798 - 2s/epoch - 7ms/step
    




    <keras.callbacks.History at 0x1ab7620e190>



What Happens inside an Epochs 
1. At the beginning of the epochs, the training loss will be set to 0
2. The algorithm will iterate over a preset number of batches, all from train_data
3. The weights and biases will be updated as many times as there are batches 
4. We will get a value for the loss fucntion, indicating how the training is going 
5. We will see the accuracy 
6. At the end of the epoch, the algorithm will forward propagate the whole validation set 
* When we reach the maximum number of epochs the training will be over

`val_accuracy` = The true accuracy of the model

## Test the model

It is very important to realize that fiddling with the hyperparameters overfits the validation dataset. 

The test is the absolute final instance. You should not test before you are completely done with adjusting your model.

If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.


```python
test_loss, test_accuracy = model.evaluate(test_data)
```

    1/1 [==============================] - 0s 175ms/step - loss: 0.0942 - accuracy: 0.9755
    


```python
# We can apply some nice formatting if we want to
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
```

    Test loss: 0.09. Test accuracy: 97.55%
    

## Result 
The final result give us 97.55% test accuracy, by using 5 hidden layers with 'relu' activication function, with :  
`hidden_layer_size` = 300  
`BUFFER_SIZE` = 10000  
`BATCH_SIZE` = 150  
`NUM_EPOCHS` = 5  
  

However we still can improve our model by using different methods or hyperparameters size.


```python

```
