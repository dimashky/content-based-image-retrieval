from keras.preprocessing import image
from neuralnetwork.FeatureExtractor import FeatureExtractor
import csv, numpy as np

featureExtractor = FeatureExtractor()

def match(imgPath, features, filePaths, limit=30):
	global featureExtractor
	img = image.load_img(imgPath)
	query = featureExtractor.extract(img)
	dists = np.linalg.norm(features - query, axis=1)
	ids = np.argsort(dists)[:limit]
	scores = [(dists[id], filePaths[id]) for id in ids]
	return scores

def getFeatures(featuresPath):
	with open(featuresPath) as f:
		reader = csv.reader(f)
		features = []
		paths = []
		for row in reader:
			features.append([float(x) for x in row[1:]])
			paths.append(row[0])
	return (features,paths)