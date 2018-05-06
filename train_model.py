import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
from network import Network
import matplotlib.pyplot as plt

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

	net = Network(model_dest='checkpoints/model0.ckpt', l1_reg=0.0005, lr=0.0001)
	_, _, losses = net.train(X_train, y_train, X_test, y_test, num_epochs=100000, batch_size=300, restore=True)
	plt.plot(losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Neural Network Learning Curve')
	plt.show()

if __name__ == '__main__':
	main()
