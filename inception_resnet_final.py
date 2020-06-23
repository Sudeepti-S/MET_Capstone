#!/usr/bin/env python
# coding: utf-8

# In[2]:


try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[3]:


import tensorflow as tf
#from tensorflow.python.keras import layers
#from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import datetime


# In[4]:


# Clear any logs from previous runs

get_ipython().system('rm -rf ./logs/ ')


# In[5]:


DATASET_PATH  = '/project/sds-capstone-aaai/met/DATA/Pytorch'
IMAGE_SIZE    = (299, 299)
NUM_CLASSES   = 23
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 25
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'


# In[10]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


# In[11]:


train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)


# In[12]:


valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[13]:


valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


# In[14]:


# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')


# In[15]:


# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
net_final.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
#print(net_final.summary())


# In[16]:


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[ ]:


#%%capture cap --no-stderr
# train the model
net_final.fit(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[tensorboard_callback])

# save trained weights
net_final.save(WEIGHTS_FINAL)




#with open('output.txt', 'w') as f:
    #f.write(cap.stdout)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[ ]:


acc = net_final.history['accuracy']
val_acc = net_final.history['val_acc']
loss = net_final.history['loss']
val_loss = net_final.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Loss.png')



# In[ ]:


plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('Accuracy.png')


# In[ ]:


plt.show()


# acc = net_final.history['accuracy']
# val_acc = net_final.history['val_acc']
# loss = net_final.history['loss']
# val_loss = net_final.history['val_loss']
# 
# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# plt.plot(epochs, acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()

# In[ ]:




