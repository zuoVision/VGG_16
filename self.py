from imgPreprocess import *
from PIL import Image
import matplotlib.pyplot as plt

img_path = './image'
save_dir = './data/train'
tfrecords_name = 'train_small'
image_size = (224,224)
batch_size = 5

image_list,label_list = get_file(img_path)

print('number of image:',len(image_list))
print('number of label:',len(label_list))
# 来10张图片测试一下image和label是否一一对应
# for i in range(10):
# 	image = Image.open(image_list[i])
# 	plt.subplot()
# 	plt.title(label_list[i])
# 	plt.imshow(image)
# 	plt.show()

tfrecords_file_path = convet_to_tfrecord(image_list,
										label_list,
										save_dir=save_dir,
										name=tfrecords_name,
										image_size=image_size)
print(tfrecords_file_path)



image_batch,label_batch = read_and_decode(tfrecords_file_path,
										batch_size=batch_size,
										image_size=image_size)
print('image shape:',image_batch.shape)
print('label shape:',label_batch.shape)

