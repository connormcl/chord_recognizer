# config.py – Connor McLaughlin

import os

root_dir = 'chords' # parent directory for all chord directories
if os.path.isdir(root_dir): # if WAV files exist, dynamically load chords
	chords = [x for x in os.listdir(root_dir) if not os.path.isfile(x)] # all chord directories
	chords = [x[0].upper() + x[1:] for x in chords] # formatting
	chord_nums = [len(os.listdir(root_dir+'/'+chord)) for chord in chords] # number of wav files for each chord
else: # WAV files don't exist, so presume default (note: cannot generate PCP vectors in this case)
	chords = ['A', 'Am', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G', 'A7', 'B', 'C7', 'E7', 'G7', 'D7', 'Cm', 'Fm', 'Gm']
chords = sorted(chords)
semi_tones = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
