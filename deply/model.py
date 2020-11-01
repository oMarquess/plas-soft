import os
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from tensorflow.python.framework import ops
from keras.preprocessing.image import img_to_array, load_img


#Load_model
Plas_spec = load_model('model_VGG_16_1.h5')
graph = ops.reset_default_graph()
Plas_spec.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Successfully loaded Plas_spec model...')

#function that does the preprocessing of the image and returns the prediction.
def model_predict(img_path):
  
    '''
        helper method to process an uploaded image
    '''
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
 
    global graph
    preds = Plas_spec.predict(image)
    a = (preds[0])
    total = sum(sum(preds))
    result = 'Falciparum: {} - Malariae:  {} - Ovale: {} -Vivax: {}'.format(round(a[0]/total), round(a[1]/total), round(a[2]/total), round(a[3]/total))
    msg = 'Thank you, But do confirm from a pathologist..'
    return result + msg
    # return 'Falciparum: ',round(a[0]/total),'\n Malariae: ' , round(a[1]/total), '\n Ovale: ',round(a[2]/total), '\n Vivax: ',round(a[3]/total)



