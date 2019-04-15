import tensorflow as tf
from vgg_16 import *
from imgPreprocess import read_and_decode
import numpy as np
from datetime import datetime
import time

IMAGE_SIZE = (224,224)
batch_size = 5 # 每个batch放多少张img   batch过大内存会不够用
num_batch = 100 # 产生的批次数

learning_rate = 0.0001 # 一般不小于
keep_prob = 0.8

total_duration = 0

tfrecords_file = './data/train/train_small.tfrecords' # tfrecords数据文件名（在目标文件目录下）
saver_path = './model/model.ckpt' # 模型保存路径
logs_trian_dir = './logs'
image_batch,label_batch = read_and_decode(tfrecords_file,
											batch_size,
											image_size=IMAGE_SIZE)


# 训练操作定义
image_batch = tf.cast(image_batch,tf.float32) # 需要将image_batch dtype转换成tf.float32 不然会报错
train_logits = inference_op(image_batch,keep_prob)
train_loss = losses(train_logits,label_batch)
train_op = training(train_loss,learning_rate)
trian_acc = evaluation(train_logits,label_batch)



# log汇总记录
summary_op = tf.summary.merge_all()

# 产生一个会话
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_trian_dir,sess.graph)
saver = tf.train.Saver()
# 节点初始化
sess.run(tf.global_variables_initializer())
#队列监控
coord = tf.train.Coordinator() #设置多线程协调器
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# 进行batch训练
try:
	for step in np.arange(num_batch):
		if coord.should_stop():
			break
		start_time = time.time()
		_,tra_loss,tra_acc = sess.run([train_op,train_loss,trian_acc])
		duration = time.time() - start_time
		# 每隔100步打印一次当前的loss , acc ,同时记录log,写入writer
		if step % 10 == 0:

			print('%s Step %d, trian loss = %.2f, train accuracy = %.2f%%, duration = %s' %
				(datetime.now(),step,tra_loss,tra_acc*100.0,duration))
			summary_str = sess.run(summary_op)
			train_writer.add_summary(summary_str,step)
			total_duration += duration
	mn = total_duration / num_batch 
	print('Average consumed time per batch:',mn)
	# 保存最后一次网络参数
	saver.save(sess,saver_path)

	'''
	# 每隔100步，保存一次训练好的模型
	if(step+1) == num_batch:
		saver.save(sess,saver_path)
	'''
except tf.errors.OutOfRangeError:
	print('Done Training -- epoch limit reached')

finally:
	coord.request_stop()
coord.join(threads)
sess.close()
