import vgg as model
import tensorflow as tf
import numpy as np
import time
import pickle


def test(nb_iterations=1013,
          batch_size=64):


    with tf.Graph().as_default():

		imagesVal, labelsVal = model.distorted_inputs('test', batch_size)
		logitsVal = model.inference_vgg(imagesVal, False, training=False)
		accuracyVal = model.evaluate(logitsVal, labelsVal, 0.0277778)
		accuracy10Val = model.evaluate(logitsVal, labelsVal, 0.05555555)
		accuracy20Val = model.evaluate(logitsVal, labelsVal, 0.1111111)
		
		accuracys5 = np.zeros([1,3], dtype=float) # Calculate mean of accuracys
		accuracys10 = np.zeros([1,3], dtype=float) 
		accuracys20 = np.zeros([1,3], dtype=float) 
		
		# Start running operations on the Graph.
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess,"modelFlop.cpkt")
			print("Model restored")

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				result = sess.run(
					[imagesVal, labelsVal, logitsVal, accuracyVal, accuracy10Val,accuracy20Val], 
					) 

				trn_acc = result[3]
				trn_acc10 = result[4]
				trn_acc20 = result[5]
				#print("iter:%5d, VALIDATION BATCH, precisions YPR : %s, %s, %s" % (iteration, trn_acc[0], trn_acc[1], trn_acc[2]))
				accuracys5[0,0]+=trn_acc[0]
				accuracys5[0,1]+=trn_acc[1]
				accuracys5[0,2]+=trn_acc[2]
				
				accuracys10[0,0]+=trn_acc10[0]
				accuracys10[0,1]+=trn_acc10[1]
				accuracys10[0,2]+=trn_acc10[2]
				
				accuracys20[0,0]+=trn_acc20[0]
				accuracys20[0,1]+=trn_acc20[1]
				accuracys20[0,2]+=trn_acc20[2]
        
		accuracys5/=nb_iterations
		accuracys10/=nb_iterations
		accuracys20/=nb_iterations
		print("accuracys 5 degrees validation set : %s, %s, %s" %(accuracys5[0,0],accuracys5[0,1],accuracys5[0,2]))
		print("accuracys 10 degrees validation set : %s, %s, %s" %(accuracys10[0,0],accuracys10[0,1],accuracys10[0,2]))
		print("accuracys 20 degrees validation set : %s, %s, %s" %(accuracys20[0,0],accuracys20[0,1],accuracys20[0,2]))

if __name__ == '__main__':
    test()
