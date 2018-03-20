import json
import matplotlib.pyplot as plt
import HPCP
import numpy as np

chords = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
semi_tones = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
show_avg_chroma = True
root_dir = '/Users/connor/Documents/yale/computer_science/490/jim2012Chords/Guitar_Only/'

all_chromas = np.zeros((len(chords)*200,13))
i = 0

for chord in chords:
	for n in range(1,201):
		file_name = chord + str(n) + '.wav'
		path = root_dir + chord + '/' + file_name

		chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024, output='numpy')

		# import pdb ; pdb.set_trace()

		# chord_col = []


		avg_chroma = np.mean(chroma, axis=0)
		avg_chroma /= sum(avg_chroma)
		avg_chroma = np.append(avg_chroma, int(chords.index(chord)))
		all_chromas[i] = avg_chroma

		print all_chromas[i]
		i += 1

np.savetxt('pcp.data', all_chromas, delimiter=',')

# for n in range(11,21):
# 	chord = 'am'
# 	file_name = chord + str(n) + '.wav'
# 	path = '/Users/connor/Documents/yale/computer_science/490/jim2012Chords/Guitar_Only/' + chord + '/' + file_name
# 	chroma = HPCP.hpcp(path, norm_frames=False, win_size=4096, hop_size=1024)
# 	parsed = json.loads(chroma)
# 	chroma = parsed['chroma']
# 	chroma = np.array(chroma)

# 	if show_avg_chroma:
# 		avg_chroma = np.mean(chroma, axis=0)

# 		plt.title(file_name)
# 		plt.bar(range(12), avg_chroma)
# 		plt.xticks(range(12), semi_tones)
# 		plt.show()
# 	else:
# 		for i in range(len(chroma)):
# 			plt.figure(i)
# 			plt.title('%s, chroma #%s'%(file_name,i))
# 			plt.bar(range(12), chroma[i])
# 			plt.xticks(range(12), ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'))
# 			plt.show()
