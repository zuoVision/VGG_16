import tensorflow as tf

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel = tf.get_variable(scope+'w',
								shape=[kh,kw,n_in,n_out],
								dtype=tf.float32,
								initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op,kernel,[1,dh,dw,1],padding='SAME')
		bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
		biases = tf.Variable(bias_init_val,trainable=True,name='b')
		z = tf.nn.bias_add(conv,biases)
		activation = tf.nn.relu(z,name=scope)
		p += [kernel,biases]
	return activation

def fc_op(input_op,name,n_out,p):
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel = tf.get_variable(scope+'w',
								shape=[n_in,n_out],
								dtype=tf.float32,
								initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.Variable(tf.constant(0.0,shape=[n_out],dtype=tf.float32),name='b')
		activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
		p += [kernel,biases]
		return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
	return tf.nn.max_pool(input_op,
							ksize=[1,kh,kw,1],
							strides=[1,dh,dw,1],
							padding='SAME',
							name=name)

def inference_op(input_op,keep_prob):
	p=[]
	conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	conv1_2 = conv_op(conv1_1,  name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	pool1 = mpool_op(conv1_2,   name='pool1',   kh=2, kw=2, dh=2, dw=2)

	conv2_1 = conv_op(pool1, 	name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv2_2 = conv_op(conv2_1,  name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	pool2 = mpool_op(conv2_2,   name='pool2',   kh=2, kw=2, dh=2, dw=2)

	conv3_1 = conv_op(pool2, 	name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_2 = conv_op(conv3_1,  name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_3 = conv_op(conv3_2,  name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	pool3 = mpool_op(conv3_3,   name='pool3',   kh=2, kw=2, dh=2, dw=2)

	conv4_1 = conv_op(pool3, 	name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_2 = conv_op(conv4_1,  name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_3 = conv_op(conv4_2,  name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool4 = mpool_op(conv4_3,   name='pool4',   kh=2, kw=2, dh=2, dw=2)

	conv5_1 = conv_op(pool4, 	name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_2 = conv_op(conv5_1,  name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_3 = conv_op(conv5_2,  name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool5 = mpool_op(conv5_3,   name='pool5',   kh=2, kw=2, dh=2, dw=2)

	shp = pool5.get_shape()
	flattened_shape = shp[1].value * shp[2].value * shp[3].value
	reshl = tf.reshape(pool5,[-1,flattened_shape],name='reshl')

	fc6 = fc_op(reshl,name='fc6',n_out=4096,p=p)
	fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')

	fc7 = fc_op(fc6_drop,name='fc7',n_out=4096,p=p)
	fc7_drop = tf.nn.dropout(fc7,keep_prob,name='fc7_drop')

	fc8 = fc_op(fc7_drop,name='fc8',n_out=2,p=p)

	softmax = tf.nn.softmax(fc8)

	return softmax

def losses(logits,labels): # logits:网络计算输出值，labels:真实值，0,1
	with tf.variable_scope('loss') as scope:
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																		labels=labels,
																		name='x_entropy_per_example')
		loss = tf.reduce_mean(cross_entropy,name='loss')
		tf.summary.scalar(scope.name + '/loss',loss)
	return loss

def training(loss,learning_rate):
	with tf.name_scope('optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		global_step = tf.Variable(0,name='global_step',trainable=False)
		train_op = optimizer.minimize(loss,global_step=global_step)
	return train_op # 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。

def evaluation(logits,labels):
	with tf.variable_scope('accuracy') as scope:
		correct = tf.nn.in_top_k(logits,labels,1)
		accuracy = tf.reduce_mean(tf.cast(correct,tf.float16))
		tf.summary.scalar(scope.name + '/accuracy',accuracy)
	return accuracy

# tensorboard
def variable_summary(var,name):
	with tf.name_scope(name):
		mean = tf.reduce_mean(var)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('mean',mean)
		tf.summary.scalar('stddev',stddev)
		tf.summary.scalar('min',tf.reduce_min(var))
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.histogram('histogram',var)
		# histogram:打印直方图，反应变量分布


