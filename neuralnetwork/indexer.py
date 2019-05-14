import glob, pickle, argparse
from keras.preprocessing import image
from FeatureExtractor import FeatureExtractor

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

output = open(args["index"], "w")

fe = FeatureExtractor()

for imagePath in sorted(glob.glob(args["dataset"] + "/*.jpg")):
	print(imagePath)
	img = image.load_img(imagePath)
	imageID = imagePath[imagePath.rfind("\\") + 1:]

	feature = fe.extract(img)
	features = [str(f) for f in feature]

	output.write("%s,%s\n" % (imageID, ",".join(features))) 
output.close()