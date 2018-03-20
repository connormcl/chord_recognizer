"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
import HPCP
import numpy as np
import tensorflow as tf

class ChordClassifier(object):
	"""docstring for ChordClassifier"""
	def __init__(self, model_loc='/tmp/predict_chords_model.ckpt'):
		self.chords = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
		self.semi_tones = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
		self.x_size = len(self.semi_tones)
		self.h_size = 35
		self.y_size = len(self.chords)
		self.X = tf.placeholder("float", shape=[None, self.x_size])
		self.w_1 = self.init_weights((self.x_size, self.h_size))
		self.w_2 = self.init_weights((self.h_size, self.y_size))
		self.yhat = self.forwardprop(self.X, self.w_1, self.w_2)
		self.predict = tf.argmax(self.yhat, axis=1)
		self.sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(self.sess, model_loc)

	def init_weights(self, shape):
		weights = tf.random_normal(shape, stddev=0.1)
		return tf.Variable(weights)

	def forwardprop(self, X, w_1, w_2):
		h = tf.nn.sigmoid(tf.matmul(X, w_1))
		yhat = tf.matmul(h, w_2)
		return yhat

	def classify(self, chroma):
		chroma = chroma.reshape((1,12))
		print self.sess.run(self.yhat, feed_dict={self.X: chroma})
		return self.chords[self.sess.run(self.predict, feed_dict={self.X: chroma})[0]]

def live_demo():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 2
	WAVE_OUTPUT_FILENAME = "raw_chord.wav"

	classifier = ChordClassifier()

	while True:
		p = pyaudio.PyAudio()
		stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)
		print("* recording")

		frames = []

		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		    data = stream.read(CHUNK)
		    frames.append(data)

		print("* done recording")

		stream.stop_stream()
		stream.close()
		p.terminate()

		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()

		# quit()

		chroma = HPCP.hpcp(WAVE_OUTPUT_FILENAME, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')
		avg_chroma = np.mean(chroma, axis=0)
		avg_chroma /= sum(avg_chroma)

		# print avg_chroma

		print 'Prediction:', classifier.classify(avg_chroma)

if __name__ == '__main__':
	live_demo()
