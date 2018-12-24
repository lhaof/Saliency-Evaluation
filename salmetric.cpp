#include <iostream>
#include <torch/torch.h>
#include "lodepng.h"
#include "tinydir.h"
#include <vector>
#include <string>
#include <cassert>
#include <TH/TH.h>

#define THRESHOLDS 256
#define EPSILON 1e-4
#define BETA 0.3

using namespace std;

struct thread_param 
{
	float precision[256];
	float recall[256];
	float mae;
	int start_line;
	int end_line;
	vector<vector<string> > lines;
};

float eval_mae(vector<unsigned char>& pred, vector<unsigned char>& gt, 
	unsigned height, unsigned width)
{
	float mae = 0;
	for (int i=0;i<height*width;i++)
	{
		mae += abs((double)(pred[i*4]) - (double)(gt[i*4])) / 255.0;
	}
	assert(height*width > 0);
	mae = mae / (height*width);
	return mae;
}

float eval_pr(vector<unsigned char>& pred, vector<unsigned char>& gt,
	unsigned height, unsigned width, float* precision, float* recall)
{
	for (int th=0;th<THRESHOLDS;th++)
	{
		float a_sum = 0;
		float b_sum = 0;
		float ab = 0;
		for (int i=0;i<height*width;i++)
		{
			unsigned int a = (pred[i*4] > th) ? 1 : 0;
			unsigned int b = (gt[i*4] > THRESHOLDS/2) ? 1 : 0;
			ab += (a & b);
			a_sum += a;
			b_sum += b;
		}
		float pre = (ab + EPSILON) / (a_sum + EPSILON);
		float rec = (ab + EPSILON) / (b_sum + EPSILON);
		//precision[th] += a_sum == 0 ? 0 : pre;
		//recall[th]    += b_sum == 0 ? 0 : rec;
		precision[th] += pre;
		recall[th]    += rec;
	}
}

void* evaluate_single_thread(void *thread_args)
{
	thread_param* param = (thread_param *)thread_args;
	for (int i = param->start_line; i < param->end_line; ++i) {

		vector<unsigned char> pred; //the raw pixels
		unsigned pred_w, pred_h;
		unsigned error = lodepng::decode(pred, pred_w, pred_h, param->lines[i][0].c_str());

		vector<unsigned char> gt;
		unsigned gt_w, gt_h;
		error = lodepng::decode(gt, gt_w, gt_h, param->lines[i][1].c_str());

		if (pred_w <= 0 || pred_h <= 0)
		{
			cout << "Saliency map should has non-zero size !" << param->lines[i][0] << endl;
		}
		if (gt_w <= 0 || gt_h <= 0)
		{
			cout << "Ground truth should has non-zero size !" << param->lines[i][0] << endl;
		}
		if (pred_w != gt_w || pred_h != gt_h) {
			cout << "Saliency map should share the same size as ground truth, " << param->lines[i][0] << endl;
		}

		param->mae += eval_mae(pred, gt, pred_h, pred_w);
		eval_pr(pred, gt, pred_h, pred_w, param->precision, param->recall);
	}
	return NULL;
}

void listdir(const char* inputdir, vector<string>& vec)
{
	vec.clear();
	tinydir_dir dir;
	int i;
	tinydir_open_sorted(&dir, inputdir);
	for (i = 0; i < dir.n_files; i++)
	{
		tinydir_file file;
		tinydir_readfile_n(&dir, &file, i);
		if (!file.is_dir)
		{
			vec.push_back(string(file.name));
		}
	}
	tinydir_close(&dir);
}

void evaluate(const char* predpath, const char* gtpath, 
	int num_thresholds, int num_threads, at::Tensor Metrics,
	at::Tensor Fmeasures, int verbose)
{
	vector<string> predlist, gtlist;
	listdir(predpath, predlist);
	listdir(gtpath, gtlist);
	assert(predlist.size() == gtlist.size());
	vector<vector<string> > lines;
	string predpath_str = string(predpath) + "/";
	string gtpath_str = string(gtpath) + "/";
	for (int i=0;i<predlist.size();i++)
	{
		lines.push_back(vector<string>());
		lines[i].push_back(predpath_str + predlist[i]);
		lines[i].push_back(gtpath_str + gtlist[i]);
	}

	int num_lines = lines.size();
	int lines_per_thread = (num_lines + num_threads - 1)/num_threads;
	pthread_t * pthread_id = new pthread_t[num_threads];
	thread_param *param = new thread_param[num_threads];
	for (int i=0;i<num_threads;i++)
	{
		param[i].lines = lines;
		param[i].mae = 0;
		memset(param[i].precision, 0, sizeof(float) * THRESHOLDS);
		memset(param[i].recall, 0, sizeof(float) * THRESHOLDS);
		param[i].start_line = lines_per_thread * i;
		param[i].end_line = lines_per_thread * (i + 1);
		if (param[i].end_line > num_lines)
			param[i].end_line = num_lines;
		pthread_create(&pthread_id[i], NULL, evaluate_single_thread, (void*)&param[i]);
	}
	for (size_t i = 0; i < num_threads; ++i) {
		pthread_join(pthread_id[i], NULL);
	}

	// post-processing
	float precision[THRESHOLDS];
	float recall[THRESHOLDS];
	float mae = 0;
	for (int th = 0; th < THRESHOLDS; ++th) {
	    precision[th] = 0;
	    recall[th] = 0;
	}
	int fmeasure_argmax = 0;
	float fmeasure_max = 0;
	for (size_t i = 0; i < num_threads; ++i) {
	    mae += param[i].mae / num_lines;
	    for (int th = 0; th < THRESHOLDS; ++th) {
	        precision[th] += param[i].precision[th] / num_lines;
	        recall[th] += param[i].recall[th] / num_lines;
	    }
	}
	for (int th = 0; th < THRESHOLDS; ++th) {
		float fmeasure = ((1 + BETA) * precision[th] * recall[th]) / (BETA * precision[th] + recall[th]);
		if (fmeasure > fmeasure_max) {
			fmeasure_max = fmeasure;
			fmeasure_argmax = th;
		}
		if (verbose==1)
		{
			cout << "Threashold " << th << ":\tMAE: " << mae << "\tPrecision: " << precision[th];
			cout << "\tRecall: " << recall[th] << "\tFmeasure: " << fmeasure << endl;
		}
		Fmeasures[th] = fmeasure;
	}
	if (verbose==1)
	{
		cout << "Max F-measre: " << fmeasure_max << endl;
		cout << "Precision:    " << precision[fmeasure_argmax] << endl;
		cout << "Recall:       " << recall[fmeasure_argmax] << endl;
		cout << "MAE:          " << mae << endl;
	}
	Metrics[0] = fmeasure_max;
	Metrics[1] = precision[fmeasure_argmax];
	Metrics[2] = recall[fmeasure_argmax];
	Metrics[3] = mae;
	delete [] pthread_id;
	delete [] param;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("evaluate", &evaluate, "evaluate");
}