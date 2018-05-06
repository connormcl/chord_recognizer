"""A live demo of the chord recognition neural net"""

import numpy as np
import tensorflow as tf
import pyaudio, wave, HPCP, config
from network import Network

chords = config.chords

def print_probs(probs):
	indices = np.argsort(probs)[::-1]
	for i in range(5):
		print("P(%s) = %.2f%%, " % (chords[indices[i]], probs[indices[i]]*100), end='')
	print()

def live_demo():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 4
	WAVE_OUTPUT_FILENAME = "raw_chord.wav"

	net = Network('checkpoints/model0.ckpt')
	net.start_live_session()

	while True:
		p = pyaudio.PyAudio()
		input("Press enter to start recording\n")

		stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                input_device_index=0,
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

		chroma = HPCP.hpcp(WAVE_OUTPUT_FILENAME, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')
		avg_chroma = np.mean(chroma, axis=0)
		avg_chroma /= sum(avg_chroma)

		probs = net.live_sess.run(net.pred_probs, feed_dict={net.input: avg_chroma.reshape((1,12))})[0]

		print('-----------------------')
		print('Prediction:', net.classify(avg_chroma))
		print('-----------------------')
		print_probs(probs)
		print()

if __name__ == '__main__':
	live_demo()
