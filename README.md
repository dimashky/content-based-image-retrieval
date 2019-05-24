# Content Based Image Retrieval
## Requirements
* Python 3.5-3.6
* Flask 1.2
* opencv python
* keras
* (imageai)[https://github.com/OlafenwaMoses/ImageAI]
## How to start
* Add images to the dataset folder (only images dont include folders)
* Run this command in your cmd to create index table
```
cd ./colorsearch && python index.py -d dataset -i ./storage/index.csv
cd ./neuralnetwork && python index.py -d dataset -i ./storage/indexvgg.csv
cd ./objectdetection && python index.py -d dataset -i ./storage/index.json
```
* Run this command in your cmd to run the server
```
python server.py
```
## Paper
We have released a [paper](https://github.com/dimashky/content-based-image-retrieval/raw/master/Content-Based%20Image%20Retrival%20A%20Comprehensive%20Study.pdf) for this work, it may be help new students.
## Contributors
* [Mohamed Khair Dimashky](https://github.com/dimashky)
* [Iman AlSamman](https://github.com/iman-sa)
