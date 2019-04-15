from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imgPreprocess import *
from vgg_16 import *

# 变量声明
BATCH_SIZE = 10
IMAGE_SIZE = (32,32) 
NUM_BATCH = 10
learning_rate = 0.0001
keep_prob = 1.0
logs_test_dir = './model'
tfrecords_file = './data/train/train_small.tfrecords'
test_dir = 'image/test'

if __name__ == '__main__':

    val, val_label = get_file(test_dir)
    # train, train_label, val, val_label = get_files(test_dir, 0.3)
 
    i = np.random.randint(0, len(val))
    img_dir = val[i]
    img = Image.open(img_dir)
    plt.title(val_label[i])
    plt.imshow(img)
    plt.show()

test_image_batch,test_label_batch = read_and_decode(tfrecords_file,BATCH_SIZE,IMAGE_SIZE)
test_image_batch = tf.cast(test_image_batch,tf.float32)
print('********************1************************')
logit = inference_op(test_image_batch,keep_prob)
acc = evaluation(logit,test_label_batch)
print('********************2************************')
x = tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMAGE_SIZE[0],IMAGE_SIZE[1],3])
print('********************3************************')
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
print('********************4************************')
sess = tf.Session()
sess.run(init)
print('********************5************************')
ckpt = tf.train.get_checkpoint_state(logs_test_dir)
print('********************6************************')
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print('***********************7*********************')
	predictions = sess.run([acc])
	print('The possibility is %.6f' % predictions)

