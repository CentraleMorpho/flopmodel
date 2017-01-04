import os
import random
import math
import tensorflow as tf
from pylab import *
import numpy as np

IMAGE_SIZE=39


def read_image(filename_queue): 
	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	
	image = tf.image.decode_jpeg(value, channels=1)

	offset_height =  int(133+np.random.uniform(-10,10,1))
	offset_width =  int(112+np.random.uniform(-10,10,1))
	im_size = int(128+np.random.uniform(-20,20,1))
	image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, im_size, im_size)
	image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)
	image = tf.slice(image, [0,0,0], [IMAGE_SIZE, IMAGE_SIZE, 1])
	

	return image  
	
def inputs(data_dir,batch_size):
	#with open(data_dir, 'r') as f:
    #		filenames = [line.rstrip('\n') for line in f]
	filenames = ["/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg","/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106/106000000/106000000_002_000472.jpg"]
	filename_queue = tf.train.string_input_producer(filenames)
	image = read_image(filename_queue)
	min_queue_examples=2
	return _generate_image_flop_batch(image,min_queue_examples, batch_size)
	
def _generate_image_flop_batch(image, min_queue_examples,
                                    batch_size):

  num_preprocess_threads = 16
  images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  

  images = tf.reshape(images,[batch_size,39,39,1])

  return images
		


if __name__=='__main__':
	batch_size=2
	with tf.Graph().as_default():
		
		images = inputs("validationImagesFlopPaths.txt", batch_size)
		
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summary = tf.image_summary('images',images)
		summaries.append(summary)
		summary_op = tf.merge_summary(summaries)
		
		init_op = tf.initialize_all_variables()
		with tf.Session() as sess:
		  sess.run(init_op)
		  coord = tf.train.Coordinator()
		  threads = tf.train.start_queue_runners(coord=coord)
		  
		  writer = tf.train.SummaryWriter('../summaries' + '/testflop', sess.graph)

		  for i in range(1): 
			result=sess.run([images,summary_op])
			writer.add_summary(result[1])
		  
		  writer.close()

		  coord.request_stop()
		  coord.join(threads)

	
