import glob, pickle
from neuralnetwork.matcher import match as vggMatch, getFeatures as vggGetFeatures
from os import path
import statistics

class Tester:
	def __init__(self):
		self.vgg16Features, self.vgg16Paths = vggGetFeatures("./storage/indexvgg.csv")
        
	def testVGG(self, query_path ="./queries", dataset_path = "./dataset"):
		dataset = glob.glob(dataset_path + "/*.jpg")
		queries = glob.glob(query_path + "/*.jpg")
		discretise_recalls = [0,0,0,0,0,0,0,0,0,0]
		APs = []
		for query_path in queries:
			query_name = path.splitext(path.basename(query_path))[0]
			print("Testing: " + query_name)
			relevance_imgs = [f for f in dataset if query_name in f]
			total_rel = len(relevance_imgs)
			results = vggMatch(query_path, self.vgg16Features, self.vgg16Paths, limit=2000)
			results = [path.basename(res) for res in results]
			print(results)
			print(relevance_imgs)
			PR = []
			current_relevance = 0
			for idx,(score,res_path) in enumerate(results):
				if(res_path in relevance_imgs):
					current_relevance += 1
					print("REL => "+res_path)
				precision = current_relevance / (idx + 1)
				recall = current_relevance / (total_rel)
				PR.append((precision, recall))
				if recall == 1:
					break
			discretise_recalls = [x + y for x, y in zip(discretise_recalls, self.discretiseRecall(PR))]
			print(discretise_recalls)
			AP = sum(discretise_recalls)/(len(discretise_recalls) + 0.0001)
			APs.append(AP)
		MAP = sum(APs)/(len(APs) + 0.0001)
		return {
			"MAP": MAP,
			"discretise_recalls": discretise_recalls
		}



	def discretiseRecall(self, PR):
		disc_recall = []
		current_disc_recall = 0
		for (precision, recall) in PR:
			if((current_disc_recall/10) < recall):
				current_disc_recall += 1
				disc_recall.append(precision)
			if(current_disc_recall == 10):
				break
		return disc_recall
				

