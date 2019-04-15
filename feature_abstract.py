from imgPreprocess import read_and_decode
import tensorflow as tf


batch_size = 4
dropout = 1.0
tfrecords_file = 'train.tfrecords'
image_batch,label_batch = read_and_decode(tfrecords_file,batch_size)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	image,label = sess.run([image_batch,label_batch])
	saver = tf.train.import_meta_graph('model/model.ckpt.meta')
	saver.restore(sess,'./model/medel.ckpt')
	graph = tf.get_default_graph()
	x_placeholder = graph.get_tensor_by_name('x_placeholder:0')
	fc7_features = graph.get_tensor_by_name('fc7:0') # 或取要提取的特征
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	print(sess.run(fc7_features,feed_dict={x_placeholder:image,keep_prob:dropout}))
	# sess.close()





