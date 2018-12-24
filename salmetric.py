import torch
import salmetric_cpp

if __name__ == '__main__':
	print('done.')
	predpath = '../deepMDC/deeplabv3plus_MDC/tmp_pred_mobilenetv2_epoch-8_on_HKU-IS/'
	gtpath = '../deepMDC/deeplabv3plus_MDC/tmp_gt_mobilenetv2_epoch-8_on_HKU-IS/'
	metrics = torch.zeros(4)
	fmeasures = torch.zeros(256)
	salmetric_cpp.evaluate(predpath, gtpath, 256, 10, metrics, fmeasures, 0)
	print(metrics)