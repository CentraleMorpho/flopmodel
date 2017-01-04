from datetime import datetime
import math
import time
import numpy as np
import tensorflow.python.platform
import tensorflow as tf

import pickle


  
def loss_flop(logitsFlop, logitsFlop2,batch_size):
	pitchs1 = tf.slice(logitsFlop, [0,1],[batch_size,1])
	pitchs2 = tf.slice(logitsFlop2, [0,1],[batch_size,1])
	yaws1 = tf.slice(logitsFlop, [0,0],[batch_size,1])
	yaws2 = tf.slice(logitsFlop2, [0,0],[batch_size,1])
	rolls1 = tf.slice(logitsFlop, [0,2],[batch_size,1])
	rolls2 = tf.slice(logitsFlop2, [0,2],[batch_size,1])
	loss = tf.nn.l2_loss(tf.add(pitchs1,tf.scalar_mul(-1, pitchs2)))+tf.nn.l2_loss(tf.add(yaws1,yaws2))+tf.nn.l2_loss(tf.add(rolls1,rolls2))
	return loss
	
	
if __name__ == '__main__':
	A = np.zeros((2,3))
	A[0,1] = 3
	
	
	
	B = np.zeros((2,3))
	B[0,1] = 3
	print(A)
	print(B)
	tA = tf.convert_to_tensor(A)
	tB = tf.convert_to_tensor(B)
	loss = loss_flop(tA,tB,2)
	
	with tf.Session() as sess:
		result = sess.run([tA,tB,loss])
		print(result[2])
	
	

	
	
