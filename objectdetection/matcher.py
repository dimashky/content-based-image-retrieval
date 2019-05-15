from imageai.Prediction import ImagePrediction
import os, json

with open('./storage/index.json') as json_file:
	indexTable = json.load(json_file)
	indexTableKeys = indexTable.keys()

def match(imagePath):
	model_path = os.path.join( os.path.expanduser("~"), ".keras","models")
	prediction = ImagePrediction()
	prediction.setModelTypeAsResNet()
	prediction.setModelPath(os.path.join(model_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
	prediction.loadModel()
	print(imagePath)
	predictions, probabilities = prediction.predictImage(imagePath, result_count=20)
	results = {}
	for eachPrediction, eachProbability in zip(predictions, probabilities):
		print(eachPrediction , " : " , eachProbability)
		if(eachPrediction not in indexTableKeys or eachProbability < 1):
			continue
		docsArray = indexTable[eachPrediction]
		for prob, img in docsArray:
			if(img in results.keys()):
				results[img] += 1
			else:
				results[img] = 1
	results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
	return [(res[1], res[0]) for res in results]
