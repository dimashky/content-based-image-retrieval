from app.colordescriptor import ColorDescriptor
from app.searcher import Searcher
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index that calculated before")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
args = vars(ap.parse_args())
 
# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)
 
# perform the search
searcher = Searcher(args["index"])
results = searcher.search(features, limit=5)

query = cv2.resize(query, (960, 540)) 
# display the query
cv2.imshow("Query", query)

# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(resultID)
	result = cv2.resize(result, (960, 540))
	cv2.imshow("Result", result)
	cv2.waitKey(0)