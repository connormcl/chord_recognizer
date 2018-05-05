import json
import matplotlib.pyplot as plt
import HPCP, config
import numpy as np

chords = config.chords[:10]
connor_chords = config.chords[10:]
connor_nums = config.connor_nums
semi_tones = config.semi_tones
show_avg_chroma = True
root_dir = '/Users/connor/Documents/yale/computer_science/490/jim2012Chords/Guitar_Only/'
root_dir2 = '/Users/connor/Documents/yale/computer_science/490/jim2012Chords/Other_Instruments/Guitar/'
root_dir3 = '/Users/connor/Documents/yale/computer_science/490/chord_recognizer/connor_chords/'

def plot_chroma(chroma):
	plt.bar(range(12), chroma)
	plt.xticks(range(12), semi_tones)
	plt.show(block=True)

def noise(mu=0.0, sigma=0.01):
	return np.random.normal(mu, sigma, 12)

# add noise --> generate noiseless vector, and vector with noise
def generate_vectors(add_noise=False):
	i = 0
	if add_noise:
		all_chromas = np.zeros((2*(len(chords)*200 + len(chords)*10 + sum(connor_nums)),13))
	else:
		all_chromas = np.zeros((len(chords)*200 + len(chords)*10 + sum(connor_nums),13))
	for chord in chords:
		print('Generating PCP vectors for', chord, 'WAV files...')
		for n in range(1,201):
			file_name = chord + str(n) + '.wav'
			path = root_dir + chord + '/' + file_name

			chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')

			avg_chroma = np.mean(chroma, axis=0)
			if add_noise:
				noisy_chroma = avg_chroma + noise()
				noisy_chroma /= sum(noisy_chroma)
				noisy_chroma = np.append(noisy_chroma, int(chords.index(chord)))
			avg_chroma /= sum(avg_chroma)
			avg_chroma = np.append(avg_chroma, int(chords.index(chord)))
			all_chromas[i] = avg_chroma
			i += 1

			if add_noise:
				all_chromas[i] = noisy_chroma
				i += 1
		for n in range(1,11):
			file_name = chord + str(n) + '.wav'
			path = root_dir2 + chord + '/' + file_name

			chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')

			avg_chroma = np.mean(chroma, axis=0)
			if add_noise:
				noisy_chroma = avg_chroma + noise()
				noisy_chroma /= sum(noisy_chroma)
				noisy_chroma = np.append(noisy_chroma, int(chords.index(chord)))
			avg_chroma /= sum(avg_chroma)
			avg_chroma = np.append(avg_chroma, int(chords.index(chord)))
			all_chromas[i] = avg_chroma
			i += 1
			
			if add_noise:
				all_chromas[i] = noisy_chroma
				i += 1

	for j in range(len(connor_chords)):
		chord = connor_chords[j]
		print('Generating PCP vectors for', chord, 'WAV files...')
		for n in range(1,connor_nums[j] + 1):
			file_name = chord + '_' + str(n) + '.wav'
			path = root_dir3 + chord + '/' + file_name

			chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')

			avg_chroma = np.mean(chroma, axis=0)
			if add_noise:
				noisy_chroma = avg_chroma + noise()
				noisy_chroma /= sum(noisy_chroma)
				noisy_chroma = np.append(noisy_chroma, int(len(chords) + connor_chords.index(chord)))
			avg_chroma /= sum(avg_chroma)
			avg_chroma = np.append(avg_chroma, int(len(chords) + connor_chords.index(chord)))
			all_chromas[i] = avg_chroma
			i += 1

			if add_noise:
				all_chromas[i] = noisy_chroma
				i += 1

	np.savetxt('pcp.data', all_chromas, delimiter=',')

if __name__ == '__main__':
	generate_vectors(add_noise=False)
