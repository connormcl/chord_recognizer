import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time, sys, config

semi_tones = config.semi_tones
chords = config.chords

def ideal_pcp(chord_tones):
	vector = np.zeros((12,))
	magnitude = 1.0 / len(chord_tones)
	for tone in chord_tones:
		vector[semi_tones.index(tone)] = magnitude
	return vector

def get_classifier():
	A = ideal_pcp(['a', 'c#', 'e'])
	Am = ideal_pcp(['a', 'c', 'e'])
	Bm = ideal_pcp(['b', 'd', 'f#'])
	C = ideal_pcp(['c', 'e', 'g'])
	D = ideal_pcp(['d', 'f#', 'a'])
	Dm = ideal_pcp(['d', 'f', 'a'])
	E = ideal_pcp(['e', 'g#', 'b'])
	Em = ideal_pcp(['e', 'g', 'b'])
	F = ideal_pcp(['f', 'a', 'c'])
	G = ideal_pcp(['g', 'b', 'd'])
	A7 = ideal_pcp(['a', 'c#', 'e', 'g'])
	B = ideal_pcp(['b', 'd#', 'f#'])
	C7 = ideal_pcp(['c', 'e', 'g', 'a#'])
	E7 = ideal_pcp(['e', 'g#', 'b', 'd'])
	G7 = ideal_pcp(['g', 'b', 'd', 'f'])
	D7 = ideal_pcp(['d', 'f#', 'a', 'c'])
	Cm = ideal_pcp(['c', 'd#', 'g'])
	Fm = ideal_pcp(['f', 'g#', 'c'])
	Gm = ideal_pcp(['g', 'a#', 'd'])

	X = [A, Am, Bm, C, D, Dm, E, Em, F, G, A7, B, C7, E7, G7, D7, Cm, Fm, Gm]

	classifier = KNeighborsClassifier(n_neighbors=1)
	classifier.fit(X, np.arange(len(chords)))

	return classifier

def main():
	# Load training and eval data
	print('loading data...')
	# load pcp vectors
	data = np.loadtxt('pcp.data', delimiter=',')
	X = data[:, 0:12]
	y = data[:, 12]
	y = y.astype(int)
	print('done.')

	classifier = get_classifier()
	accuracy = np.mean(y == classifier.predict(X))
	print('Nearest Neighbors Accuracy =', accuracy)

if __name__ == '__main__':
	main()
