import math
import os
import argparse
import numpy as np
import collections
#import matplotlib.pyplot as plt
import random
from io_ import read_csv


def get_args():

        parser = argparse.ArgumentParser()
        parser.add_argument("file", action='store', type=str, help="Input file", default='list')
        return parser.parse_args()


def euclidian_distance(point1, point2):
	return sum([math.pow(point1[i] - point2[i], 2) for i in range(len(point1))])


def kcluster(data, k = 3, max_iter = 100):

	maxs = [max([value[i] for value in data]) for i in range(len(data[0]))] 
	mins = [min([value[i] for value in data]) for i in range(len(data[0]))]

	clusters = []
	point_clusters = []
	for i in range(k):
	
		c = [random.random()*(maxs[j] - mins[j]) + mins[j] for j in range(len(data[0]))]
		clusters.append(c)
		point_clusters.append([c])
	
	for row in data:
		
		min_value = euclidian_distance(row, clusters[0])
		index_min = 0
		for i in range(1, k):
			
			dist = euclidian_distance(row, clusters[i])
			if(min_value > dist):
				min_value = dist
				index_min = i
		
		point_clusters[index_min].append(row)
	
	for iter_ in range(max_iter):
	
		clusters = []
		new_point_clusters = []
		for i in range(k):
		
			means = [sum([value[index_] for value in point_clusters[i]]) for index_ in range(len(data[0]))]
			means = [value / float(len(point_clusters[i])) for value in means]

			clusters.append(means)
			new_point_clusters.append([means])
	
		point_clusters = []
		point_clusters = new_point_clusters
	
		for row in data:
			
			min_value = euclidian_distance(row, clusters[0])
			index_min = 0
			for i in range(1, k):
		
				dist = euclidian_distance(row, clusters[i])
				if(min_value > dist):
					min_value = dist
					index_min = i

			point_clusters[index_min].append(row)

	return point_clusters


def betaCV(point_clusters):

	Nin = 0
	for c in point_clusters:
		
		size = len(c)
		num_edges = (size * (size - 1)) / 2.0
		Nin += num_edges
	
	Nout = 0
	for i in range(len(point_clusters) - 1):
		
		size_i = len(point_clusters[i])
		for j in range(i+1, len(point_clusters)):			
		
			size_j = len(point_clusters[j])
			num_edges = size_i * size_j
			Nout += num_edges
	
	
	Win = 0
	for c in point_clusters:
		
		for i in range(len(c) - 1):
			
			for j in range(i + 1, len(c)):
				
				dist_c = euclidian_distance(c[i], c[j])
				Win += dist_c
	
	Wout = 0	
	for i in range(len(point_clusters) - 1):
		
		for j in range(i+1, len(point_clusters)):
			
			for k in range(len(point_clusters[i])):
				
				for l in range(len(point_clusters[j])):
					
					dist_c = euclidian_distance(point_clusters[i][k], point_clusters[j][l])
					Wout += dist_c
	
	result = (Win / Nin) / (Wout / Nout)
	return result


def kmeans(data, k = 3, max_iter = 100, beta_CV = False):

	iter_improve = 10
	res = kcluster(data, k, max_iter)
	
	if(beta_CV is False):
		return res
	
	betacv = betaCV(res)
	
	for i in range(iter_improve):
		
		res_actual = kcluster(data)
		betacv_actual = betaCV(res_actual)
		if(betacv_actual < betacv):

			res = res_actual
			betacv = betacv_actual
	
	return res		


def main(args):

	data = read_csv(args.file)
	result = kmeans(data, beta_CV = True)

if __name__ == '__main__':
	# parse arguments
	args = get_args()
	main(args)

	
