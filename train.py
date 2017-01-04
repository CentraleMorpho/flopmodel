import vgg as model
import tensorflow as tf
import numpy as np
import time
import pickle


def train(lr=0.0001,
          nb_iterations=100000,
          batch_size=64):


    with tf.Graph().as_default():
		

		images, labels = model.distorted_inputs('training', batch_size)
		imagesVal, labelsVal = model.distorted_inputs('test', batch_size)

		logits = model.inference_vgg(images, False, training=True)
		objectiveGT = model.loss_op(logits, labels, batch_size)
		
		accuracy = model.evaluate(logits, labels, 0.05555555)
		
		logitsVal = model.inference_vgg(imagesVal, True, training=False)
		objectiveGTVal = model.loss_op(logitsVal, labelsVal, batch_size)
		accuracyVal = model.evaluate(logitsVal, labelsVal, 0.0277778)

		###########################
		#FLOP
		###########################

		imagesFlop = model.flop_inputs('training', batch_size)
		imagesFlopVal = model.flop_inputs('test', batch_size)
		
		imagesFlop2 = tf.reverse(imagesFlop, [False,False,True,False])
		imagesFlopVal2 = tf.reverse(imagesFlopVal, [False,False,True,False])

		logitsFlop = model.inference_vgg(imagesFlop, True, training=True)
		logitsFlop2 = model.inference_vgg(imagesFlop2, True, training=True)
		objectiveFlop = model.loss_flop(logitsFlop, logitsFlop2,batch_size)
		
		
		logitsFlopVal = model.inference_vgg(imagesFlopVal, True, training=False)
		logitsFlopVal2 = model.inference_vgg(imagesFlopVal2, True, training=False)
		objectiveFlopVal = model.loss_flop(logitsFlopVal, logitsFlopVal2, batch_size)

		###########################
		
		optimizer = tf.train.AdamOptimizer(lr)
		global_step = tf.Variable(0, name="global_step", trainable=False)
		train_step_GT = optimizer.minimize(objectiveGT, global_step=global_step)
		train_step_Flop = optimizer.minimize(objectiveFlop, global_step=global_step)
		
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		#summaries.append(tf.scalar_summary('Loss Training GT', objectiveGT))
		#summaries.append(tf.scalar_summary('Loss Training Flop', objectiveFlop))
		summaries.append(tf.scalar_summary('Loss Validation GT', objectiveGTVal))
		summaries.append(tf.scalar_summary('Loss Validation Flop', objectiveFlopVal))
		summary_op = tf.merge_summary(summaries)
		


		# Start running operations on the Graph.
		with tf.Session() as sess:
		
			train_writer = tf.train.SummaryWriter('../summaries' + '/flop', sess.graph)
			
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())
			#saver.restore(sess,"modelFlop.cpkt")
			#print("Model restored")

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				
				result = sess.run([logits, train_step_GT, objectiveGT, accuracy, objectiveFlop, train_step_Flop])	
				
				if iteration%100==0:
					trn_lossGT = result[2]

					print("iter:%5d, trn_lossGT: %s, acc : %s, trn_lossFlop : %s" % (iteration, trn_lossGT, result[3], result[4]))
					

				if iteration%1000==0:
					result = sess.run([labelsVal, logitsVal,objectiveGTVal, objectiveFlopVal, summary_op])
					trn_lossGT = result[2]
					trn_lossFlop = result[3]
					# print("VALIDATION BATCH, iter:%5d, trn_lossGT: %s, trn_lossFlop : %s" % (iteration, trn_lossGT, trn_lossFlop))
					# print(result[0])
					# print(result[1])
					
					#Save to summaries
					summary = tf.Summary()
					summary.ParseFromString(result[4])
					train_writer.add_summary(summary, iteration)
				
			print("Saving model...")
			save_path=saver.save(sess, "modelFlop100000.cpkt")
			print("Model saved in file : %s" % save_path)
				
                    


if __name__ == '__main__':
    batch_size = 64
    train(batch_size = batch_size)
