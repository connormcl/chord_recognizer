import numpy as np
import tensorflow as tf
import time, sys, config

class Network(object):
	"""Neural Network for Chord Recognition"""
	def __init__(self, model_dest=None, lr=0.001, l1_reg=0.00005, momentum=0.95, dropout_rate=0.3):
		self.lr = lr
		self.chords = config.chords
		self.model_dest = model_dest
		self.live_sess = None
		# architecture
		self.input = tf.placeholder(shape=[None, 12], dtype=tf.float32)
		self.labels = tf.placeholder(shape=[None,], dtype=tf.int32)
		self.dense = tf.layers.dense(self.input, units=1000, activation=tf.nn.relu)
		self.dropout = tf.layers.dropout(self.dense, rate=dropout_rate)
		self.dense2 = tf.layers.dense(self.dropout, units=36, activation=tf.nn.relu)
		self.logits = tf.layers.dense(self.dense2, units=len(self.chords))
		# prediction ops
		self.pred_class = tf.argmax(self.logits, axis=1)
		self.pred_probs = tf.nn.softmax(self.logits)
		# loss op
		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_reg, scope=None)
		penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
		self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits) + penalty
		# optimizer
		optimizer = tf.train.MomentumOptimizer(self.lr, momentum)
		# training op
		self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

	def get_batch(self, X, y, batch_size):
		random_indices = np.arange(X.shape[0])
		np.random.shuffle(random_indices)
		return (X[random_indices[0:batch_size]], y[random_indices[0:batch_size]])

	def train(self, X_train, y_train, X_test, y_test, num_epochs=500, batch_size=100, restore=False):
		sess = tf.Session()
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		sess.run(init)

		if self.model_dest and restore:
			print('Restoring model from %s...' % self.model_dest)
			saver.restore(sess, self.model_dest)
			print('Done.')

		losses = []

		for epoch in range(num_epochs):
			t0 = time.time()
			# train batch
			batch = self.get_batch(X_train, y_train, batch_size)
			# test batch
			test_batch = self.get_batch(X_test, y_test, batch_size)
			
			# gradient update step
			sess.run(self.train_op, feed_dict={self.input: batch[0], self.labels: batch[1]})

			# accuracies on current batch
			train_accuracy = np.mean(batch[1] == sess.run(self.pred_class, feed_dict={self.input: batch[0], self.labels: batch[1]}))
			test_accuracy = np.mean(test_batch[1] == sess.run(self.pred_class, feed_dict={self.input: test_batch[0], self.labels: test_batch[1]}))

			losses.append(sess.run(self.loss, feed_dict={self.input: batch[0], self.labels: batch[1]}))

			print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% (%.3f seconds)"
                      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, time.time() - t0))
			if self.model_dest and (epoch % 500 == 0):
				print('Saving model to %s...' % self.model_dest)
				saver.save(sess, self.model_dest)
				print('Done.')

		t0 = time.time()
		print('Computing final accuracies...')

		train_accuracy = np.mean(y_train == sess.run(self.pred_class, feed_dict={self.input: X_train, self.labels: y_train}))
		test_accuracy = np.mean(y_test == sess.run(self.pred_class, feed_dict={self.input: X_test, self.labels: y_test}))

		print("Full, final stats: train accuracy = %.2f%%, test accuracy = %.2f%% (%.3f seconds)"
                      % (100. * train_accuracy, 100. * test_accuracy, time.time() - t0))

		if self.model_dest:
			print('Saving model to %s...' % self.model_dest)
			saver.save(sess, self.model_dest)
			print('Done.')
		return train_accuracy, test_accuracy, losses

	def classify(self, chroma):
		if not self.live_sess:
			sys.exit('Error: Must start live session before classifying')
		chroma = chroma.reshape((1,12))
		return self.chords[self.live_sess.run(self.pred_class, feed_dict={self.input: chroma})[0]]

	# to be used before classify
	def start_live_session(self):
		if not self.model_dest:
			sys.exit('Error: No path to model provided')

		self.live_sess = tf.Session()
		init = tf.global_variables_initializer()
		self.live_sess.run(init)
		saver = tf.train.Saver()

		print('Loading model from %s...' % self.model_dest)
		saver.restore(self.live_sess, self.model_dest)
		print('Done.')



