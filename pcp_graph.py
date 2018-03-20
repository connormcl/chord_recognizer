import json
import matplotlib.pyplot as plt
import HPCP
import numpy as np

semi_tones = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
# semi_tones = ('g', 'g#', 'a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#')
show_avg_chroma = True

# ideal_d_major = [0.0,0.0,0.33,0.0,0.0,0.0,0.33,0.0,0.0,0.33,0.0,0.0]
# plt.title('Ideal D Major PCP representation')
# plt.bar(range(12), ideal_d_major)
# plt.xticks(range(12), semi_tones)
# plt.show()

for n in range(1,21):
	chord = 'd'
	file_name = chord + str(n) + '.wav'
	path = '/Users/connor/Documents/yale/computer_science/490/jim2012Chords/Guitar_Only/' + chord + '/' + file_name
	chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')
	# import pdb ; pdb.set_trace()
	# parsed = json.loads(chroma)
	# chroma = parsed['chroma']
	# chroma = np.array(chroma)

	if show_avg_chroma:
		avg_chroma = np.mean(chroma, axis=0)

		plt.title('Real D Major PCP representation')
		plt.bar(range(12), avg_chroma)
		plt.xticks(range(12), semi_tones)
		plt.show()
	else:
		for i in range(len(chroma)):
			plt.figure(i)
			plt.title('%s, chroma #%s'%(file_name,i))
			plt.bar(range(12), chroma[i])
			plt.xticks(range(12), ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'))
			plt.show()
