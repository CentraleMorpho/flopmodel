import os
import random
import math
import pickle


def createDictFromLabelsFile():
	posesFile = '../Cropped/024_poseonly_normalised180.txt'
	dico = {}
	
	with open(posesFile,'r') as f:
			i=0
			for line in f:
				path = line.split(' ')[0]
				if(path.split('/')[0]=='020' | path.split('/')[0]=='022'):
					# | path.split('/')[0]=='014b'
					path = os.path.join("../Cropped",path)
					dico[path]=[float(line.split(' ')[1]),float(line.split(' ')[2]),float(line.split(' ')[3])]
				print(i)
				i=i+1
				
	return dico
	
	


if __name__=='__main__':
	print('Making the dictionary...')
	dico = createDictFromLabelsFile()
	
    	with open('dictLabels' '.pkl', 'wb') as f:
        	pickle.dump(dico, f, pickle.HIGHEST_PROTOCOL)

