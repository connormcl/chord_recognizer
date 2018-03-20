import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sys import argv

def one_hot_labels(y):
	one_hot = np.zeros((y.size, y.max()+1))
	one_hot[np.arange(y.size), y] = 1
	return one_hot

def init_weights(shape):
	weights = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
	h = tf.nn.sigmoid(tf.matmul(X, w_1))
	yhat = tf.matmul(h, w_2)
	return yhat

def main(restore=False, train=True):
	# load pcp vectors
	data = np.loadtxt('pcp.data', delimiter=',')
	X = data[:, 0:12]
	y = data[:, 12]
	y = y.astype(int)
	y = one_hot_labels(y)

	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# feed-forward network layer sizes
	x_size = X_train.shape[1]
	h_size = 35
	y_size = y_train.shape[1]

	# placeholders for data and labels
	X = tf.placeholder("float", shape=[None, x_size])
	y = tf.placeholder("float", shape=[None, y_size])

	# initialize TensorFlow variables to hold connection weights
	w_1 = init_weights((x_size, h_size))
	w_2 = init_weights((h_size, y_size))

	# TensorFlow op for forward propogation
	yhat = forwardprop(X, w_1, w_2)
	# TensorFlow op for selecting most probable class (chord type)
	predict = tf.argmax(yhat, axis=1)

	# cost function... idk just copied this
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
	# update operation, using Gradient Descent with lr=0.01
	lr = 0.005
	updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

	sess = tf.Session()
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	if restore:
		saver.restore(sess, "/tmp/predict_chords_model.ckpt")
	else:
		sess.run(init)

	if train:
		for epoch in range(1000):
			for i in range(len(X_train)):
				sess.run(updates, feed_dict={X: X_train[i: i+1], y: y_train[i: i+1]})

			train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(predict, feed_dict={X: X_train, y: y_train}))
			test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(predict, feed_dict={X: X_test, y: y_test}))

			print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
	              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

		save_path = saver.save(sess, "/tmp/predict_chords_model.ckpt")
		print "Model saved in path: %s" % save_path
	else:
		import matplotlib.pyplot as plt
		chords = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
		semi_tones = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
		for i in range(len(X_test)):
			# print 'Prediction: ', sess.run(predict, feed_dict={X: X_test[0:1], y: y_test[0:1]})
			print 'Prediction:', chords[sess.run(predict, feed_dict={X: X_test[i: i+1], y: y_test[i: i+1]})[0]], 'True chord:', chords[np.argmax(y_test[i: i+1], axis=1)[0]]
			# plt.title('%s, chroma #%s'%(file_name,i))
			plt.bar(range(12), X_test[i])
			plt.xticks(range(12), semi_tones)
			plt.show()

	sess.close()
	
	# else:
	# 	saver.restore(sess, "/tmp/predict_chords_model.ckpt")
	# 	# print "w_1 : %s" % w_1.eval(session=sess)
	# 	train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(predict, feed_dict={X: X_train, y: y_train}))
	# 	test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(predict, feed_dict={X: X_test, y: y_test}))

	# 	print("train accuracy = %.2f%%, test accuracy = %.2f%%"
 #              % (100. * train_accuracy, 100. * test_accuracy))

if __name__ == '__main__':
	if len(argv) > 1 and argv[1] == '--restore':
		# main(restore=True, train=False)
		main(restore=True)
	else:
		main()
