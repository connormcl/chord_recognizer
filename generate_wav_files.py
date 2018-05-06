import numpy as np
import pyaudio, wave
import os.path

# create wav files: chords/chord_name/chord_name_#.wav (with # in [start_n, end_n] inclusive)
def generate_wavs(chord_name, start_n, end_n, overwrite=False):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 4
	BASE_DIR = 'chords/' + chord_name + '/'

	for i in range(start_n, end_n + 1):
		WAVE_OUTPUT_FILENAME = BASE_DIR + chord_name + '_' + str(i) + ".wav"
		if (not os.path.isfile(WAVE_OUTPUT_FILENAME)) or overwrite:
			p = pyaudio.PyAudio()

			input("Press enter to start recording\n")

			stream = p.open(format=FORMAT,
		                channels=CHANNELS,
		                rate=RATE,
		                input=True,
		                frames_per_buffer=CHUNK)
			print("* recording " + WAVE_OUTPUT_FILENAME)

			frames = []

			for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			    data = stream.read(CHUNK)
			    frames.append(data)

			print("* done recording\n\n\n\n")

			stream.stop_stream()
			stream.close()
			p.terminate()

			wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
			wf.setnchannels(CHANNELS)
			wf.setsampwidth(p.get_sample_size(FORMAT))
			wf.setframerate(RATE)
			wf.writeframes(b''.join(frames))
			wf.close()
		else:
			print('Aborting: file already exists. Use overwrite=True if you wish to overwrite.')
			quit()

if __name__ == '__main__':
	# example: 
	# generate_wavs('e7', 1,100)
