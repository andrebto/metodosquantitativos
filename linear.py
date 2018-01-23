import argparse
import numpy as np
import math
import pandas as pd


def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("file", action='store', type=str, help="Input file (x)", default='list')
	parser.add_argument("file2", action="store", type=str, help="Input file (y)")
	return parser.parse_args()


def cost_function(X, y, w):
	
	err = X.dot(w) - y
	cost = np.sum(err ** 2) / (2 * len(y))
	return cost


def predict(X, w):
	return X.dot(w)


def gradient_descent(X, y, w, learning_rate, iters, debug):

	for iter_ in range(iters):
	
		predictions = predict(X, w)
		loss = predictions - y
		grad = X.T.dot(loss) / len(y)
		w = w - learning_rate * grad
		cost = cost_function(X, y, w)
		
		if(debug):
			msg = "Iter: " + str(iter_) + " Cost: " + str(cost)
			print(msg) 
	return w,cost


def train(X, y, iters, learning_rate = 0.0001, debug = False):

	w = [0 for i in range(X.shape[1])]
	w = np.array(w)
	
	w_final = gradient_descent(X, y, w, learning_rate, iters, debug)
	return w_final


def r2_score(X, y, w):
	
	predictions = predict(X, w)
	meany = sum(y) / len(y)
	sstotal = sum([math.pow(yvalue - meany, 2) for yvalue in y])
	ssres = sum([math.pow(y[i] - predictions[i], 2) for i in range(len(y))])
	return 1 - (ssres / sstotal)



def main(args):

	X = pd.read_csv(args.file, header=None)
	X.insert(0, 'Uns', 1) #interceptor
	X = np.array(X.values)	

	y = pd.read_csv(args.file2, header=None)
	y = np.array(y)
	y = np.ravel(y[:,0]) #Cut extra dimension

	w,cost = train(X, y, 1000, debug = False)

if __name__ == '__main__':
	# parse arguments
	args = get_args()
	main(args)
	


