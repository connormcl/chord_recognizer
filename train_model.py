import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
from network import Network

def main():
	# Load training and eval data
	print('loading pcp.data...')
	# load pcp vectors
	data = np.loadtxt('pcp.data', delimiter=',')
	X = data[:, 0:12]
	y = data[:, 12]
	y = y.astype(int)
	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	print('done.')

	net = Network(model_dest='checkpoints/model0.ckpt')
	net.train(X_train, y_train, X_test, y_test, num_epochs=120000, batch_size=200)

if __name__ == '__main__':
	main()
