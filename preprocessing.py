from sklearn import datasets
import pandas as pd


def species_label(raw_data, theta):
	if theta==0:
		return raw_data.target_names[0]
	if theta==1:
		return raw_data.target_names[1]
	if theta==2:
		return raw_data.target_names[2]
	
def loadData():
    raw_data = datasets.load_iris()
    data_desc = raw_data.DESCR
    data = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    data['species'] = [species_label(raw_data, theta) for theta in raw_data.target]
    data['species_id'] = raw_data.target
    print(data)
	
loadData()