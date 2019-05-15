from imageai.Prediction import ImagePrediction
import os, pickle, argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

indexTable = {}

def getPredictor():
	model_path = os.path.join( os.path.expanduser("~"), ".keras","models")
	prediction = ImagePrediction()
	prediction.setModelTypeAsResNet()
	prediction.setModelPath(os.path.join(model_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
	prediction.loadModel()
	return prediction

multiple_prediction = getPredictor()
all_images_array_paths = []
all_images_array_names = []

all_files = os.listdir(args["dataset"])
for each_file in all_files:
	if(each_file.endswith(".jpg") or each_file.endswith(".png")):
		all_images_array_paths.append(os.path.join(args["dataset"],each_file))
		all_images_array_names.append(each_file)

results_array = multiple_prediction.predictMultipleImages(all_images_array_paths, result_count_per_image=5)

for idx, each_result in enumerate(results_array):
	predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
	for index in range(len(predictions)):
		print(predictions[index] , " : " , percentage_probabilities[index])
		if(predictions[index] in indexTable.keys()):
			indexTable[predictions[index]].append((percentage_probabilities[index], all_images_array_names[idx]))
		else:
			indexTable[predictions[index]] = [((percentage_probabilities[index], all_images_array_names[idx]))]
	print("-----------------------")

with open(args["index"], 'w') as outfile:  
    json.dump(indexTable, outfile)