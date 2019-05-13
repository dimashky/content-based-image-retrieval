from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.colordescriptor import ColorDescriptor
from app.searcher import Searcher
import os, cv2

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['SESSION_TYPE'] = 'filesystem'

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

		cd = ColorDescriptor((8, 12, 3))

		query = cv2.imread(filePath)
		features = cd.describe(query)
		
		# perform the search
		searcher = Searcher("./storage/index.csv")
		results = searcher.search(features, limit=5)
		os.remove(filePath)
		
	return render_template('index.html', results = results)

@app.route('/similar/<imageID>')
def similar(imageID):
	cd = ColorDescriptor((8, 12, 3))
	filePath = "dataset/"+imageID
	query = cv2.imread(filePath)
	features = cd.describe(query)
	
	# perform the search
	searcher = Searcher("./storage/index.csv")
	results = searcher.search(features, limit=5)
	return render_template('index.html', results = results)

@app.route('/assets/<path:path>')
def sendAsset(path):
    return send_from_directory('static', path)

@app.route('/img/<path:path>')
def sendImg(path):
    return send_from_directory("dataset",path)

if __name__ == '__main__':
    app.run(debug=True)