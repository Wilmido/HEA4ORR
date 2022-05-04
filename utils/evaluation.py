import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss):
	plt.plot(train_loss,label='training loss')
	plt.plot(test_loss,label="validation loss")
	plt.legend() 
	plt.show()


def evaluate(y, pred):
	mae = np.sum(np.abs(y - pred)) / len(y)
	rmse = np.sqrt(np.sum((y - pred)**2) / len(y))
	return mae, rmse


