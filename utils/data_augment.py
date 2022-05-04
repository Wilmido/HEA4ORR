import numpy as np
import pandas as pd
from PearsonSelection import feature_selection

def atom_augment(inputs):
	'''
	increase the number of atoms' descriptors
	Args:
		inputs.shape[0]: number of atoms
		inputs.shape[1]: 5 descriptors
	'''
	assert inputs.shape[1] == 5
	data = []

	for i in range(inputs.shape[0]):
		# Ru
		if np.all(inputs[i][:3] == [0, 0, 0]):
			inputs[i][:3] = [5, 8, 1.338]
			data.append(np.hstack((inputs[i], [2.20, 8, 101.07, 44])))
		# Rh    
		if np.all(inputs[i][:3] == [0, 1, 0]):
			inputs[i][:3] = [5, 9, 1.345]
			data.append(np.hstack((inputs[i], [2.28, 9, 102.906, 45])))
		# Pd
		if np.all(inputs[i][:3] == [0, 2, 3]):
			inputs[i][:3] = [5, 10, 1.375]
			data.append(np.hstack((inputs[i], [2.20, 10, 106.42, 46])))
		# Ir
		if np.all(inputs[i][:3] == [1,1,1.5]):
			inputs[i][:3] = [6, 9, 1.357]
			data.append(np.hstack((inputs[i], [2.20, 9, 192.2, 77])))  
		# Pt 
		if np.all(inputs[i][:3] == [1,2,4.5]):
			inputs[i][:3] = [6, 10, 1.387]
			data.append(np.hstack((inputs[i], [2.28, 10, 195.08, 78])))       
	return data


def function_augment(data):
	"""using 4 fundamental functions to generate new non-linear descriptors """
	data1 = np.sqrt(data)
	data2 = data ** 2 
	data3 = data ** 3
	data4 = np.log1p(data)

	new_data = np.hstack((data1, data2, data3, data4))
	return new_data

def multiply_augment(data):
	new_data = []
	for i in range(0, data.shape[1]-1):
		for j in range(i+1, data.shape[1]):
			new_data.append(data[:,i] * data[:,j])

	new_data = np.array(new_data, dtype=float).T
	return new_data


if __name__ == '__main__':
	df = pd.read_csv('../data/data.csv',header=None)
	inputs = df.iloc[:, 2:-1]     # (1370, 120)
	outputs = df.iloc[:, -1]
	num_hea = inputs.shape[0]

	inputs = np.asarray(inputs,dtype=float).reshape(-1, 5)
	inputs = atom_augment(inputs)
	
	outputs = np.asarray(outputs, dtype=float).reshape(num_hea,-1)
	
	
	new_data = pd.DataFrame(inputs, columns=['Period','Group','Radius','CN','AtSite','Negativity', 'VEC', 'M','atomic number'])
	print(new_data.shape)     #(32880, 9)
	new_data.to_csv('../data/my_data.csv', index=False)
	np.savetxt('../data/label.txt/', outputs, delimiter=" ")

	inputs = feature_selection(inputs) 
	print(inputs.shape)     # (32880,6)
	new_data = pd.DataFrame(inputs, columns=['Period','Radius','CN','AtSite','Negativity', 'M'])
	new_data.to_csv('../data/new_data.csv', index=False)
	np.savetxt('../data/label.txt/', outputs, delimiter=" ")


	data2 = function_augment(inputs)
	data3 = multiply_augment(inputs)
	FMdata = np.hstack((inputs,data2,data3))
	
	print(FMdata.shape)   # (32880,45)

	augment_data = np.hstack((FMdata,np.divide(1, FMdata,out=np.zeros_like(FMdata), where=FMdata!=0)))
	print(augment_data.shape)   # (32880,90)
 
	augment_data = pd.DataFrame(augment_data,columns=['A','B','C','D','E','F',
													  'A^0.5','B^0.5','C^0.5','D^0.5','E^0.5','F^0.5',
													  'A^2','B^2','C^2','D^2','E^2','F^2',
													  'A^3','B^3','C^3','D^3','E^3','F^3',
													  'log(1+A)','log(1+B)','log(1+C)','log(1+D)','log(1+E)','log(1+F)',
													  'AB','AC','AD','AE','AF',
													  'BC','BD','BE','BF','CD',
													  'CE','CF','DE','DF','EF',
													  '1/A','1/B','1/C','1/D','1/E','1/F',
													  '1/A^0.5','1/B^0.5','1/C^0.5','1/D^0.5','1/E^0.5','1/F^0.5',
													  '1/A^2','1/B^2','1/C^2','1/D^2','1/E^2','1/F^2',
													  '1/A^3','1/B^3','1/C^3','1/D^3','1/E^3','1/F^3',
													  '1/log(1+A)','1/log(1+B)','1/log(1+C)','1/log(1+D)','1/log(1+E)','1/log(1+F)',
													 '1/AB','1/AC','1/AD','1/AE','1/AF',
													 '1/BC','1/BD','1/BE','1/BF','1/CD',
													 '1/CE','1/CF','1/DE','1/DF','1/EF',
													 ])
	augment_data.to_csv('../data/augment_data.csv',index=False)