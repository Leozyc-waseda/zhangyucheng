# zhangyucheng

# Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#　


import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
