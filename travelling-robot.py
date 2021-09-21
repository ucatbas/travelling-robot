#!/usr/bin/python3
import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import time  

DEBUG = False
"""
## Planning: The Traveling Robot Problem

Visit a collection of points in the shortest path you can find.
The catch? You have to "go home to recharge" every so often.

We want fast approximations rather than a brute force perfect solution.
Your solution will be judged on:
* the length of path it produces
* fast runtime
* code quality and maintainability

### Details

* There are 5000 points distributed uniformly in [0, 1]
* The recharge station is located at (.5, .5)
* You cannot travel more than 3 units of distance before recharging
* You must start and end at the recharge station
* Skeleton code provided in Python. Python and C++ are acceptable
"""

#############################
begin_time = datetime.datetime.now()
home = np.array([0.5, 0.5])  # home is the recharging station
max_charge = 3.0
#############################

# generate the points to visit uniformly in [0,1]
# recharging station is index 0
N = 5000
pts = np.vstack((home, np.random.rand(N, 2)))


def check_order(pts, order):
	"""Check whether a given order of points is valid, and prints the total
	length. You start and stop at the charging station.
	pts: np array of points to visit, prepended by the location of home
	order: array of pt indices to visit, where 0 is home
	i.e. order = [0, 1, 0, 2, 0, 3, 0]"""

	print("Checking order")
	assert(pts.shape == (N + 1, 2))  # nothing weird
	assert(order[0] == 0)  # start path at home
	assert(order[-1] == 0)  # end path at home
	assert(set(order) == set(range(N + 1)))  # all pts visited

	print("Assertions passed")

	# traverse path
	total_d = 0
	charge = max_charge
	last = pts[0, :]

	for idx in order:
		pt = pts[idx, :]
		d = np.linalg.norm(pt - last)

		# update totals
		total_d += d
		charge -= d
		assert(charge > 0)  # out of battery

		# did we recharge?
		if idx == 0:
			charge = max_charge

		# moving to next point
		last = pt

	# We made it to end! path was valid
	print("Valid path!")
	print(f"{total_d}")
	draw_path(pts, order)


def draw_path(pts, order):
	"""Draw the path to the screen"""
	path = pts[order, :]

	plt.plot(path[:, 0], path[:, 1])

	if DEBUG:
		plt.scatter(pts[:, 0], pts[:, 1])

	plt.show()



#############################
## Solution Skeleton: The Traveling Robot Problem

# A replica of the nearest neighbor algorithm was developed.
# While looking at each new point, algorithm also checks the charge status and distance from home position.
# 6 different functions are used below:

# Function  						|	Time Complexity
# -----------------------------------------------------------------
# distance  						|	O(1)
# distance_modified					|	O(1)	
# get_index							|	O(N)	
# remove							|	O(N)			
# nearest_neighbor_modified			|	O(N*O(1)) -> O(N)	
# reorder							|	O(N*(O(N)+O(N))) -> O(N^2)

# Tested Time for 50, 500 and 5000:

# Avg(50)   -> (0.0121 + 0.0119 + 0.0124 + 0.0123)/4 = 0.012175 seconds
# Avg(500)  -> (1.1072 + 1.0975 + 1.0973 + 1.1125)/4 = 1.103625 seconds
# Avg(5000) -> (108.06 + 109.10 + 110.02 + 108.82)/4 = 109 seconds
# It can be seen from a small statistics that when N increases 10 times and time increases around 100 times which satisfies O(N^2) 

# Space Complexity of reorder function is O(N) since we create only arrays with length of N

#############################
# Your code goes here
# Read the "pts" array
# generate a valid order, starting and ending with 0, the recharging station

max_distance = 10  # Any value greater than the highest possible distance value

def distance(p1, p2):
    '''calculate the distance between to locations using their coordinates
    Input:
    - p1 [float float]
    - p2 [float float]
    Output:
    - return the euclidian distance between the 2 points A and B
    '''
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def distance_modified(p1, p2, charge):
	'''calculate the distance between to locations using their coordinates and checks if charge is sufficient
	Input:
	- p1 [float float]
	- p2 [float float]
	- charge float
	Output:
	- return the euclidian distance between the 2 points A and B or max_charge (since the grid is small)
	'''
	if distance(p1, p2) + distance(p1, home) < charge:
		return distance(p1, p2)
	return max_distance


def get_index(points, p):
	'''Find the index in points of a given point
	Input:
	- points [[float float] ... [float float]]
	- p [float float]
	Output:
	- return the index of a point in all points
	'''
	for i in range(len(points)):
		comparison = points[i] == p
		if comparison.all():
			return i
	return -1


def remove(points, p):
	'''Removes the point from points
	Input:
	- points [[float float] ... [float float]]
	- p [float float]
	Output:
	- return the new array of points
	'''
	point_list = list(points)
	for i in range(len(point_list)):
		comparison = point_list[i] == p
		if comparison.all():
			del point_list[i]
			break
	return np.asarray(point_list)


def nearest_neighbor_modified(points, p, charge):
	'''Find the point in points that is nearest to point A with sufficient .
	Input:
	- points [[float float] ... [float float]]
	- p [float float]
	- charge float
	Output:
	- return the next point with minimum distance or home
	'''
	min_distance = max_distance
	point = home
	for i in range(len(points)):
		distance = distance_modified(points[i], p, charge)
		if distance < min_distance:
			point = points[i]
			min_distance = distance
	return min_distance, point


def reorder(points):
	'''Start the tour at the first point; at each step extend the tour by moving from the previous point to its nearest neighbor that has not yet been visited.
	Input:
	- points [[float float] ... [float float]]
	Output:
	- return the array of indexes that robot will follow
	'''
	s = time.time()
	best_order = [0]
	robot_charge = max_charge
	unvisited_points = points[1:] 	# All points except home to visit
	while unvisited_points.any():
		nn_distance, nn_point = nearest_neighbor_modified(unvisited_points, points[best_order[-1]], robot_charge)
		if nn_distance == max_distance: 	# Robot have to move to home
			best_order.append(0)
			robot_charge = max_charge
		else:
			best_order.append(get_index(points, nn_point))
			robot_charge = robot_charge - nn_distance
			unvisited_points = remove(unvisited_points, nn_point)
	print("Elapsed Time for reorder Function : {} seconds".format(time.time() - s))
	return best_order


order = reorder(pts)
order.append(0)

check_order(pts, order)

#############################
