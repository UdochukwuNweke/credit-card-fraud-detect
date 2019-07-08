import matplotlib.pyplot as plt
import numpy as np

def euclideanDist(a, b):
	size = len(a)
	total = 0

	for i in range(size):
		diff = a[i]- b[i]
		diff = diff * diff
		total += diff

	return total**0.5

def getDistMatrix(points):
	row = len(points)
	col = len(points[0])
	matrix = np.zeros((row, row))
	
	for i in range(row):
		for j in range(row):
			dist = 0

			for k in range(col):
				diff = (points[i,k] - points[j,k])
				diff = diff * diff
				dist += diff
			matrix[i, j] = dist**0.5
	
	return matrix

def getClustMinDists(clusters, distMat):
	
	'''
	print('clusters:')
	print(clusters)
	print()
	print('dist matrix:')
	print(distMat)
	'''
	
	mini = -1
	minj = -1
	minVal = 10000

	for i in range(distMat.shape[0]):
		for j in range(distMat.shape[1]):
			
			if( i == j ):
				continue

			if( distMat[i, j] < minVal ):
				minVal = distMat[i, j]
				mini = i
				minj = j
	
	#print('minVal:', minVal, mini, minj)
	return (mini, minj)

def mergeClusters(clusters, mergeIndx):
	
	merged = clusters[mergeIndx[0]] + clusters[mergeIndx[1]]
	newClusters = [merged]
	#print('\tmerged:', merged)
	
	for i in range(len(clusters)):
		if( i in mergeIndx ):
			continue

		newClusters.append( clusters[i] )
	
	#print('\tnewClusters:', newClusters)
	return newClusters

def distBetweenClusters(first, second, points, linkage):
	
	if( linkage == 'min' ):
		clustsDist = 1000
	else:
		clustsDist = -1
	

	if( linkage == 'min' or linkage == 'max' or linkage == 'groupAvg' ):

		#print('first:', first)
		#print('second:', second)
		groupAvg = 0
		for firstIndx in first:
			for secondIndx in second:
				
				dist = euclideanDist(points[firstIndx], points[secondIndx])
				groupAvg += dist
				#print('\t', points[firstIndx], points[secondIndx])
				
				if( linkage == 'min' and dist < clustsDist ):
					clustsDist = dist
				elif( linkage == 'max' and dist > clustsDist ):
					clustsDist = dist

		clustsDist = groupAvg/(len(first)*len(second))


	elif( linkage == 'centroid' ):
		
		firstCentroid = {'x': 0, 'y': 0}
		secondCentroid = {'x': 0, 'y': 0}
		
		for firstIndx in first:
			firstCentroid['x'] += points[firstIndx][0]
			firstCentroid['y'] += points[firstIndx][1]

		firstCentroid['x'] = firstCentroid['x']/len(first)
		firstCentroid['y'] = firstCentroid['y']/len(first)

		for secondIndx in second:
			secondCentroid['x'] += points[secondIndx][0]
			secondCentroid['y'] += points[secondIndx][1]

		secondCentroid['x'] = secondCentroid['x']/len(second)
		secondCentroid['y'] = secondCentroid['y']/len(second)

		
		clustsDist = euclideanDist(
			[firstCentroid['x'], firstCentroid['y']],
			[secondCentroid['x'], secondCentroid['y']]
		)

	return clustsDist

def updateDistMatrix(distMat, linkage, clusters, points):

	row = len(clusters)
	matrix = np.zeros((row, row))

	for i in range(len(clusters)):
		for j in range(len(clusters)):
			dist = distBetweenClusters(clusters[i], clusters[j], points, linkage)
			matrix[i, j] = dist
	
	return matrix
	

def HAC(points, linkage, maxIter):
	
	distMat = getDistMatrix(points)
	
	clusters = []
	for i in range(len(points)):
		clusters.append( [i] )

	curIter = 1
	while len(clusters) != maxIter:
		
		print('iter:', curIter, 'len:', len(clusters))

		minDist = getClustMinDists(clusters, distMat)
		clusters = mergeClusters(clusters, minDist)
		distMat = updateDistMatrix(distMat, linkage, clusters, points)
		curIter += 1


	#print('last:', clusters)
	return clusters


def drawPoints(plt, clusters, title, points):
	
	'''
	
	'''
	colors = ['ro', 'bo', 'go', 'yo', 'co', 'mo', 'ko', 'C0o', 'C1o', 'C2o']
	for i in range(len(clusters)):
		pointsIndices = clusters[i]

		X = []
		Y = []
		for pointIndex in pointsIndices:
			X.append( points[pointIndex][0] )
			Y.append( points[pointIndex][1] )

		plt.suptitle(title, fontsize=10)
		plt.plot(X, Y, colors[i])

def getData(fname):
	iFile = open(fname, 'r')
	lines = iFile.readlines()
	iFile.close()

	points = []
	for l in lines:
		l = l.strip().split(' ')
		points.append( [float(l[0]), float(l[1])] )

	return points

maxClustSize = 2
linkage = 'max'
filename = 'B.txt'
figName = filename + '.' + linkage + '.' + str(maxClustSize) + '.png'
points = getData(filename)
points = np.array(points)

clusters = HAC(points, linkage, maxClustSize)

drawPoints(plt, clusters, 'HAC Alg. result, clust size = ' + str(maxClustSize) + ', linkage:' + linkage, points)
plt.savefig(figName, dpi=300)