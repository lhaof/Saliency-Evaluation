import torch
import salmetric_cpp

if __name__ == '__main__':
	print('done.')
	predpath = '../YOUR_SALIENCY_MAPS_PATH/'
	gtpath = '../YOUR_GROUND_TRUTH_PATH/'
	metrics = torch.zeros(4)
	fmeasures = torch.zeros(256)
	salmetric_cpp.evaluate(predpath, gtpath, 256, 10, metrics, fmeasures, 0)
	# metrics contains maxf, precision, recall and mae.
	print(metrics)
