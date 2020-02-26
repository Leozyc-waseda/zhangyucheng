# zhangyucheng

# Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#　


import tensorflow as tf

from keras import backend as K

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

K.set_session(sess)

# ubuntu系统中，批量对图片进行翻转

convert -rotate 180 *.jpg new.jpg 




