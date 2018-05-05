import json
import matplotlib.pyplot as plt
import HPCP, config
import numpy as np

root_dir = config.root_dir + '/'
chords = config.chords
chord_nums = config.chord_nums
semi_tones = config.semi_tones

def plot_chroma(chroma):
	plt.bar(range(12), chroma)
	plt.xticks(range(12), semi_tones)
	plt.show(block=True)

def noise(mu=0.0, sigma=0.01):
	return np.random.normal(mu, sigma, 12)

def generate_vectors():
	i = 0 # index for chroma matrix
	all_chromas = np.zeros((sum(chord_nums),13))
	for j in range(len(chords)):
		chord = chords[j]
		n = chord_nums[j]
		print('Generating PCP vectors for', chord, 'WAV files...')
		for k in range(1,n+1):
			file_name = chord + '_' + str(k) + '.wav'
			path = root_dir + chord + '/' + file_name

			chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')

			avg_chroma = np.mean(chroma, axis=0)
			avg_chroma /= sum(avg_chroma)
			avg_chroma = np.append(avg_chroma, j)
			all_chromas[i] = avg_chroma
			i += 1

	np.savetxt('pcp.data', all_chromas, delimiter=',')

if __name__ == '__main__':
	generate_vectors()
