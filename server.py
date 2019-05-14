from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from colorsearch.colordescriptor import ColorDescriptor
from colorsearch.searcher import Searcher
from neuralnetwork.matcher import match, getFeatures
import os, cv2

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['SESSION_TYPE'] = 'filesystem'

vgg16Features, vgg16Paths = getFeatures("./storage/indexvgg.csv")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
	results = []
	if request.method == 'POST':
		if 'file' not in request.files:
			print('No file part')
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '' or not allowed_file(file.filename):
			print('No selected file or not supported extension')
			return redirect(request.url)

		filename = secure_filename(file.filename)
		filePath = os.path.join("queries", filename)
		file.save(filePath)
		""" **** COLOR SEARCH **** 
		cd = ColorDescriptor((8, 12, 3))
		query = cv2.imread(filePath)
		features = cd.describe(query)
		searcher = Searcher("./storage/index.csv")
		results = searcher.search(features, limit=20)
		"""
		""" **** VGG 16 SEARCH ****  """
		results = match(filePath, vgg16Features, vgg16Paths)
		return render_template('index.html', results = results, imgQuery=filename)
	return render_template('index.html')

@app.route('/similar/<imageID>')
def similar(imageID):
	cd = ColorDescriptor((8, 12, 3))
	filePath = "dataset/"+imageID
	query = cv2.imread(filePath)
	features = cd.describe(query)
	
	# perform the search
	searcher = Searcher("./storage/index.csv")
	results = searcher.search(features, limit=20)
	return render_template('index.html', results = results)

@app.route('/assets/<path:path>')
def sendAsset(path):
    return send_from_directory('static', path)

@app.route('/img/<path:path>')
def sendImg(path):
    return send_from_directory("dataset",path)

@app.route('/query/<path:path>')
def sendImgQuery(path):
    return send_from_directory("queries",path)

if __name__ == '__main__':
    app.run(debug=True)