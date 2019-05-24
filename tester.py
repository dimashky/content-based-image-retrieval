import glob
import pickle
import cv2
import os
from colorsearch.searcher import Searcher
from colorsearch.colordescriptor import ColorDescriptor
from neuralnetwork.matcher import match as vggMatch, getFeatures as vggGetFeatures
from objectdetection.matcher import match as objectMatching
from os import path
from imageai.Prediction import ImagePrediction

class Tester:
	def __init__(self):
		self.vgg16Features, self.vgg16Paths = vggGetFeatures("./storage/indexvgg.csv")
		self.searcher = Searcher("./storage/index.csv")
		model_path = os.path.join( os.path.expanduser("~"), ".keras","models")
		prediction = ImagePrediction()
		prediction.setModelTypeAsResNet()
		prediction.setModelPath(os.path.join(model_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
		prediction.loadModel()
		self.prediction = prediction

	def testing(self, test_type, query_path="./queries", dataset_path="./dataset"):
		dataset = glob.glob(dataset_path + "/*.jpg")
		queries = glob.glob(query_path + "/*.jpg")
		discretise_recalls = [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
		APs = []
		cd = ColorDescriptor((8, 12, 3))
		for query_path in queries:
			query_name = path.splitext(path.basename(query_path))[0]
			print("Testing: " + query_name)
			relevance_imgs = [f.replace("./dataset\\", "") for f in dataset if query_name in f]
			total_rel = len(relevance_imgs)
			print("\tTotal Relevance = %d", total_rel)
			results = []
			if (test_type == "vgg"):
				results = vggMatch(query_path, self.vgg16Features, self.vgg16Paths, limit=13000)
			elif (test_type == "color"):
				try:
					features = cd.describe(cv2.imread(query_path))
					results = self.searcher.search(features, limit=13000)
				except Exception as e:
					print(e)
					continue
			elif (test_type == "objects"):
				results = objectMatching(os.path.join(os.getcwd(), query_path), self.prediction)
			results = [res[1].replace("./dataset\\", "") for res in results]
			PR = []
			current_relevance = 0
			for idx, res_path in enumerate(results):
				if (res_path in relevance_imgs):
					current_relevance += 1
				precision = current_relevance / (idx + 1)
				recall = current_relevance / (total_rel)
				PR.append((precision, recall))
				if recall == 1:
					break
				discretise_recall = self.discretiseRecall(PR)
				for i in range(len(discretise_recall)):
					discretise_recalls[i] = (discretise_recalls[i][0]+ discretise_recall[i], discretise_recalls[i][1] + 1)
				AP = sum(discretise_recall) / (len(discretise_recall) + 0.000001)
				APs.append(AP)
		MAP = sum(APs) / (len(APs) + 0.000001)
		return {
			"MAP": MAP,
			"discretise_recalls": [x[0] / (x[1]+0.000001) for x in discretise_recalls]
		}

	def discretiseRecall(self, PR):
		disc_recall = []
		current_disc_recall = 0
		for (precision, recall) in PR:
			if ((current_disc_recall / 10) < recall):
				current_disc_recall += 1
				disc_recall.append(precision)
			if (current_disc_recall == 10):
				break
		return disc_recall
