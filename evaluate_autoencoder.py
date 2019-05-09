'''
Evaluate autoencoder
'''


import logging
import numpy as np
import tflearn
from keras.preprocessing import image

import train_autoencoder

# setup
logging.basicConfig(level=logging.DEBUG)


# functions
def get_img(img_path):
    logging.debug('loading image at '.format(img_path))
    img = image.load_img(img_path,
                         target_size=train_autoencoder.IMAGE_INPUT_SIZE)

    return np.expand_dims(image.img_to_array(img), axis=0)


def main():
    model = train_autoencoder.build_model()
    model = tflearn.DNN(model)

    logging.info('loading checkpoint')
    #checkpoint_path = sys.argv[1]
    #checkpoint_path = 'checkpoints'
    #model.load(checkpoint_path)
    model.load('my_model.tflearn')
    #img_path = sys.argv[2]
    img_path = 'images/0/1.jpg'
    img_arr = get_img(img_path)

    logging.info('getting output')
    pred = model.predict(img_arr)

    logging.debug('saving output to output.jpg')
    pred = pred[0]
    pred_img = image.array_to_img(pred)
    pred_img.save('output.jpg')


if __name__ == '__main__':
    main()

