import tensorflow as tf
import numpy as np
import os
from PIL import Image

def get_file(file_dir):
	print('searching images...')
	images = []
	temp = [] # 临时list
	labels = []

	# for root,sub_folders,files in os.walk(file_dir):
	# 	for  name in files:
	# 		if name.endswith('.jpg'):
	# 			images.append(os.path.join(root,name))
	# 		if name in sub_folders:
	# 			temp.append(os.path.join(root,name)) #存放文件路径

	# 	for one_folder in temp:
	# 		n_img = len(os.listdir(one_folder))
	# 		print('number of files:',n_img)
	# 		class_name = one_folder.split('\\')[-1] # 文件名即为分类名称
	# 		if class_name == 'cats':
	# 			labels = np.append(labels,n_img*[1])
	# 		elif class_name == 'dogs':
	# 			labels = np.append(labels,n_img*[2])
	
	cats_path = file_dir + '/cats'
	dogs_path = file_dir + '/dogs'
	m=n=0
	for filename in os.listdir(cats_path):
		m += 1
		images.append(os.path.join(cats_path,filename))
		labels.append(0)
		if m >= 100:break
	# print(len(images))
	for filename in os.listdir(dogs_path):
		n += 1
		images.append(os.path.join(dogs_path,filename))
		labels.append(1)
		if n >= 100:break
	# print(len(images))
	temp = np.array([images,
					labels])  # [2,total_n_img]
	temp = temp.transpose()  # 转置
	np.random.shuffle(temp)  # shuffle
	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	label_list = [int(float(i)) for i in label_list]
	
	'''
	# 将所有的list分为两部分，一部分用来训练tra,一部分用来验证val
	n_sample = len(image_list)
	n_val = int(math.ceil(n_sample * ratio))
	n_train = n_sample - n_val

	tra_images = image_list[0:n_train]
	tra_labels = label_list[0:n_train]
	tra_labels = [int(flaot(i)) for i in tra_labels]

	val_images = image_list[n_train:-1]
	val_labels = label_list[n_train:-1]
	val_labels = [int(flaot(i)) for i in val_labels]
	
	return tra_images,tra_labels,val_images,val_labels
	'''
	print('search done !')
	return image_list,label_list

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convet_to_tfrecord(images,labels,save_dir,name,image_size):
	filename = os.path.join(save_dir,name+'.tfrecords')

	if tf.gfile.Exists(filename):
		print('\n%s already exist!\n' % filename)
		return 
	n_samples = len(labels)
	if np.shape(images)[0] != n_samples:
		raise ValueError('Image size %d dose not match labels size %d.' % 
						(images.size(),labels.size()))
 	
	writer = tf.python_io.TFRecordWriter(filename)
	print('\nTransform start...')
	m=n=0
	for i in np.arange(0,n_samples):
		try:
			m += 1
			image = Image.open(images[i])
			image = image.resize(image_size)
			image_raw = image.tobytes()
			label = int(labels[i])
			example = tf.train.Example(features=tf.train.Features(feature={
				'image_raw':_bytes_feature(image_raw),
				'label':_int64_feature(label)	
				}))
			writer.write(example.SerializeToString())
			print('Num of successful:',m)
		except IOError as e:
			n += 1
			print('Could not read:',images[i])
			print('Error type:',e)
			print('Skip it !\n')
	writer.close()	
	print('Transform done !')
	print('Transformed : %d\t failed : %d\n' % (m,n))
	return filename

def read_and_decode(tfrecords_file,batch_size,image_size):
	filename_queue = tf.train.string_input_producer([tfrecords_file])
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	img_feature = tf.parse_single_example(serialized_example,
											features={
											'image_raw':tf.FixedLenFeature([],tf.string),
											'label':tf.FixedLenFeature([],tf.int64)
											})
	image = tf.decode_raw(img_feature['image_raw'],tf.uint8)
	image = tf.reshape(image,[image_size[0],image_size[1],3])
	label = tf.cast(img_feature['label'],tf.int32)
	image_batch, label_batch = tf.train.batch([image,label],
												batch_size=batch_size,
												num_threads=64,
												capacity=2000)
	return image_batch,tf.reshape(label_batch,[batch_size])
























