import os
import random
import math


def getImagesPaths():
	listPaths = []
	listDirectories = [x[0] for x in os.walk('/mnt/SSD/stagiaire/VisagePoseEstimation/Img/B106')]

	for directory in listDirectories:
		for f in os.listdir(directory):
			filename, extension=os.path.splitext(f)
			if(extension==".jpg"):
				path = os.path.join(directory, f)
				listPaths.append(path)	
	return listPaths
	
	


if __name__=='__main__':
	print('Getting all the images paths...')
	listPaths = getImagesPaths()

	random.seed(123)
	print('Shuffling the list')
	random.shuffle(listPaths)
	nbTrainingSet = int(math.floor(len(listPaths)*90/100))
	print('Creating the training set')
	trainingSetPaths = listPaths[:nbTrainingSet]
	print('Creating the validation set')
	validationSetPaths = listPaths[nbTrainingSet+1:]
	
	print('Saving the training paths to a txt file')
	trainingSetPathsFile = open('trainingImagesFlopPaths.txt','w')
	for item in trainingSetPaths:
		trainingSetPathsFile.write("%s\n" % item)
	trainingSetPathsFile.close()

	print('Saving the validation paths to a txt file')
	validationSetPathsFile = open('validationImagesFlopPaths.txt','w')
	for item in validationSetPaths:
		validationSetPathsFile.write("%s\n" % item)
	validationSetPathsFile.close()
	print('Completed')
	

	
