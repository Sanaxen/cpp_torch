/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"

#define _LIBRARY_EXPORTS
#include "tiny_dnn2libtorch_dll.h"

#define TEST		//cpp_torch
#define USE_CUDA
#define RNN_LAYERS_OPT


namespace rnn_dll_variables
{
#ifdef USE_CUDA
	int use_gpu = 1;
#else
	int use_gpu = 0;
#endif


	// The batch size for training.
	int64_t kTrainBatchSize = 0;

	// The batch size for Test.
	int64_t kTestBatchSize = 0;

	// The number of epochs to train.
	int64_t kNumberOfEpochs = 0;


	cpp_torch::network_torch<cpp_torch::Net>* nn_ = nullptr;
	cpp_torch::network_torch<cpp_torch::Net>* nn2_ = nullptr;
	cpp_torch::network_torch<cpp_torch::Net>* nn3_ = nullptr;
	torch::Device device(torch::kCPU);
	int test_mode = 0;
	float maxvalue = 0.0;
	float minvalue = 0.0;

	int x_dim = 0;
	int y_dim = 0;
	int sequence_length = 0;
	int out_sequence_length = 0;
	int target_position = 0;
	int rnn_layers = 0;
	int n_layers = 0;
	int n_hidden_size = 0;
	int fc_hidden_size = 0;
	float dropout = 0;
	float learning_rate = 0;
	float clip_gradients = 0;
	int early_stopping = 0;
	char opt_type[32] = { '\0' };
	int n_train_epochs = 0;
	int n_minibatch = 0;
	int n_eval_minibatch = 0;
	int prophecy = 10;
	float tolerance = 1.0e-6;
	char rnn_type[16] = { '\0' };

	char weight_init_type[16] = { '\0' };
	char activation_fnc[16] = { '\0' };
	char regression[16] = { '\0' };
	int input_size = 0;
	int classification = 0;
	int num_class = 0;
	int use_attention = 1;
	int use_cnn = 0;
	int use_cnn_add_bn = 0;
	int use_add_bn = 0;
	int padding_prm = 0;
	int residual = 0;
	float use_observed_value = 1.0;
	int use_self_sequence = 0;

	std::string optimizer_name = "adam";
	bool batch_shuffle = true;
	int shuffle_seed = 1;

	float moment = 0.01;
	float scale = 1.0;

	bool state_reset_mode = false;
	int pycode_dump_only = 0;

	bool L1_loss = false;

	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;
};
using namespace rnn_dll_variables;

extern "C" _LIBRARY_EXPORTS void torch_params(
	int n_train_epochs_ = 100,
	int n_minibatch_ = 32,
	int input_size_ = 64,

	int n_layers_ = 5,
	float dropout_ = 0.1,
	int n_hidden_size_ = 64,
	int fc_hidden_size_ = 64,
	float learning_rate_ = 0.001,

	float clip_gradients_ = 0,
	int use_cnn_ = 0,
	int use_add_bn_ = 0,
	int use_cnn_add_bn_ = 0,
	int residual_ = 0,
	int padding_prm_ = 0,

	int classification_ = 0,
	char* weight_init_type_ = "xavier",
	char* activation_fnc_ = "tanh",
	int early_stopping_ = 100,
	char* opt_type_ = "adam",
	bool batch_shuffle_ = true,
	int shuffle_seed_ = 1,
	bool L1_loss_ = false,
	int test_mode_ = 0
){
	 n_train_epochs = n_train_epochs_;
	 n_minibatch = n_minibatch_;
	 input_size = input_size_;
	 
	 n_layers = n_layers_;
	 dropout = dropout_;
	 n_hidden_size = n_hidden_size_;
	 fc_hidden_size = fc_hidden_size_;
	 learning_rate = learning_rate_;
	 
	 clip_gradients = clip_gradients_;
	 use_cnn = use_cnn_;
	 use_add_bn = use_add_bn_;
	 use_cnn_add_bn = use_cnn_add_bn_;
	 residual = residual_;
	 padding_prm = padding_prm_;
	 
	 classification = classification_;
	 strcpy(weight_init_type, weight_init_type_);
	 strcpy(activation_fnc, activation_fnc_);
	 
	 early_stopping = early_stopping_;
	 strcpy( opt_type, opt_type_);
	 test_mode = test_mode_;
	 batch_shuffle = batch_shuffle_;
	 shuffle_seed = shuffle_seed_;
	 L1_loss = L1_loss_;

	 //printf("dropout:%f\n", dropout);
}

extern "C" _LIBRARY_EXPORTS void read_mnist_dataset(const std::string &data_dir_path)
{
	std::vector<tiny_dnn::label_t> tr_labels;
	std::vector<tiny_dnn::label_t> te_labels;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels-idx1-ubyte",
		&tr_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images-idx3-ubyte",
		&train_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels-idx1-ubyte",
		&te_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images-idx3-ubyte",
		&test_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	cpp_torch::label2vec(tr_labels, train_labels, 10);
	cpp_torch::label2vec(te_labels, test_labels, 10);
}

cpp_torch::network_torch<cpp_torch::Net>* toNet(void* nn)
{
	return (cpp_torch::network_torch<cpp_torch::Net>*)nn;
}

extern "C" _LIBRARY_EXPORTS int cuda_is_available()
{
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		return 0;
	}
	return -1;
}
extern "C" _LIBRARY_EXPORTS void  set_sampling(float rate)
{
	cpp_torch::Net& model = toNet(nn_)->model;

	model.get()->set_sampling_rate(rate);
}
extern "C" _LIBRARY_EXPORTS void  reset_sampling()
{
	cpp_torch::Net& model = toNet(nn_)->model;

	model.get()->reset_sampling_rate();
}

extern "C" _LIBRARY_EXPORTS void state_reset(std::string& rnn_type , void* nn)
{
	//return;
	if (!state_reset_mode) return;

	cpp_torch::Net& model = toNet(nn)->model;
	if (rnn_type == "lstm")
	{
		for (int i = 0; i < model.get()->lstm.size(); i++)
		{
			model.get()->lstm[i].get()->reset_parameters();
		}
	}
	if (rnn_type == "gru")
	{
		for (int i = 0; i < model.get()->gru.size(); i++)
		{
			model.get()->gru[i].get()->reset_parameters();
		}
	}
	if (rnn_type == "rnn")
	{
		for (int i = 0; i < model.get()->rnn.size(); i++)
		{
			model.get()->rnn[i].get()->reset_parameters();
		}
	}
}


extern "C" _LIBRARY_EXPORTS void torch_getData(const char* filename, std::vector<tiny_dnn::vec_t>& data)
{
	CSVReader csv(filename, ',', false);
	data = csv.toTensor();
}
void scaling_data(std::vector<tiny_dnn::vec_t>& data, std::vector<tiny_dnn::vec_t>& dataset)
{
	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < dataset.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < dataset[0].size(); k++)
		{
			y.push_back(dataset[i][k]/ scale);
		}
		data.push_back(y);
	}
}


extern "C" _LIBRARY_EXPORTS void send_train_images(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(train_images, data);
}
extern "C" _LIBRARY_EXPORTS void send_train_labels(std::vector<tiny_dnn::vec_t>& data)
{
	float scale_ = scale;
	if (classification >= 2)
	{
		scale = 1.0;
	}
	scaling_data(train_labels, data);
	scale = scale_;
}
extern "C" _LIBRARY_EXPORTS void send_test_images(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(test_images, data);
}
extern "C" _LIBRARY_EXPORTS void send_test_labels(std::vector<tiny_dnn::vec_t>& data)
{
	float scale_ = scale;
	if (classification >= 2)
	{
		scale = 1.0;
	}
	scaling_data(test_labels, data);
	scale = scale_;
}

extern "C" _LIBRARY_EXPORTS void get_train_images(std::vector<tiny_dnn::vec_t>& data)
{
	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < train_images.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < train_images[0].size(); k++)
		{
			y.push_back(train_images[i][k] * scale);
		}
		data.push_back(y);
	}
}
extern "C" _LIBRARY_EXPORTS void get_train_labels(std::vector<tiny_dnn::vec_t>& data)
{
	float scale_ = scale;
	if (classification >= 2)
	{
		scale = 1.0;
	}

	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < train_labels.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < train_labels[0].size(); k++)
		{
			y.push_back(train_labels[i][k] * scale);
		}
		data.push_back(y);
	}
	scale = scale_;
}

extern "C" _LIBRARY_EXPORTS void get_test_images(std::vector<tiny_dnn::vec_t>& data)
{
	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < test_images.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < test_images[0].size(); k++)
		{
			y.push_back(test_images[i][k] * scale);
		}
		data.push_back(y);
	}
}
extern "C" _LIBRARY_EXPORTS void get_test_labels(std::vector<tiny_dnn::vec_t>& data)
{
	float scale_ = scale;
	if (classification >= 2)
	{
		scale = 1.0;
	}
	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < test_labels.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < test_labels[0].size(); k++)
		{
			y.push_back(test_labels[i][k] * scale);
		}
		data.push_back(y);
	}
	scale = scale_;
}

extern "C" _LIBRARY_EXPORTS void torch_read_train_params()
{
	torch_read_params(true);
}
extern "C" _LIBRARY_EXPORTS void torch_read_test_params()
{
	torch_read_params(true);
	torch_read_params(false);
}

extern "C" _LIBRARY_EXPORTS void torch_read_params(bool train)
{
	char* buf = new char[100000];
	char* param = "train_params.txt";
	if (!train) param = "test_params.txt";

	FILE* fp = fopen( param, "r");

	while (fgets(buf, 100000, fp) != NULL)
	{
		if (buf[0] == '#')continue;
		if (strstr(buf, "test_mode"))
		{
			sscanf(buf, "test_mode:%d", &test_mode);
		}
		if (strstr(buf, "x_dim"))
		{
			sscanf(buf, "x_dim:%d", &x_dim);
		}
		if (strstr(buf, "y_dim"))
		{
			sscanf(buf, "y_dim:%d", &y_dim);
		}
		if (strstr(buf, "sequence_length"))
		{
			sscanf(buf, "sequence_length:%d", &sequence_length);
		}
		if (strstr(buf, "out_sequence_length"))
		{
			sscanf(buf, "out_sequence_length:%d", &out_sequence_length);
		}
		if (strstr(buf, "target_position"))
		{
			sscanf(buf, "target_position:%d", &target_position);
		}
		if (strstr(buf, "clip_gradients"))
		{
			sscanf(buf, "clip_gradients:%f", &clip_gradients);
		}
		if (strstr(buf, "rnn_layers"))
		{
			sscanf(buf, "rnn_layers:%d", &rnn_layers);
		}
		if (strstr(buf, "n_layers"))
		{
			sscanf(buf, "n_layers:%d", &n_layers);
		}
		if (strstr(buf, "n_hidden_size"))
		{
			sscanf(buf, "n_hidden_size:%d", &n_hidden_size);
		}
		if (strstr(buf, "fc_hidden_size"))
		{
			sscanf(buf, "fc_hidden_size:%d", &fc_hidden_size);
		}
		if (strstr(buf, "early_stopping"))
		{
			sscanf(buf, "early_stopping:%d", &early_stopping);
		}
		if (strstr(buf, "dropout"))
		{
			sscanf(buf, "dropout:%f", &dropout);
		}
		if (strstr(buf, "opt_type"))
		{
			sscanf(buf, "opt_type:%s", opt_type);
		}
		if (strstr(buf, "learning_rate"))
		{
			sscanf(buf, "learning_rate:%f", &learning_rate);
		}
		if (strstr(buf, "n_train_epochs"))
		{
			sscanf(buf, "n_train_epochs:%d", &n_train_epochs);
		}
		if (strstr(buf, "n_eval_minibatch"))
		{
			sscanf(buf, "n_eval_minibatch:%d", &n_eval_minibatch);
		}
		if (strstr(buf, "n_minibatch"))
		{
			sscanf(buf, "n_minibatch:%d", &n_minibatch);
		}
		if (strstr(buf, "prophecy"))
		{
			sscanf(buf, "prophecy:%d", &prophecy);
		}
		if (strstr(buf, "tolerance"))
		{
			sscanf(buf, "tolerance:%f", &tolerance);
		}
		if (strstr(buf, "rnn_type"))
		{
			sscanf(buf, "rnn_type:%s", rnn_type);
		}
		if (strstr(buf, "regression"))
		{
			sscanf(buf, "regression:%s", regression);
		}
		if (strstr(buf, "input_size"))
		{
			sscanf(buf, "input_size:%d", &input_size);
		}
		if (strstr(buf, "classification"))
		{
			sscanf(buf, "classification:%d", &classification);
			if (classification >= 2) num_class = classification;
		}
		if (strstr(buf, "padding_prm"))
		{
			sscanf(buf, "padding_prm:%d", &padding_prm);
		}
		if (strstr(buf, "residual"))
		{
			sscanf(buf, "residual:%d", &residual);
		}
		if (strstr(buf, "use_attention"))
		{
			sscanf(buf, "use_attention:%d", &use_attention);
		}
		if (strstr(buf, "use_cnn"))
		{
			sscanf(buf, "use_cnn:%d", &use_cnn);
		}
		if (strstr(buf, "use_cnn_add_bn"))
		{
			sscanf(buf, "use_cnn_add_bn:%d", &use_cnn_add_bn);
		}
		if (strstr(buf, "use_add_bn"))
		{
			sscanf(buf, "use_add_bn:%d", &use_add_bn);
		}
		if (strstr(buf, "use_observed_value"))
		{
			sscanf(buf, "use_observed_value:%f", &use_observed_value);
		}
		//
		if (strstr(buf, "state_reset_mode"))
		{
			int dmy = 0;
			sscanf(buf, "state_reset_mode:%d", &dmy);
			if (dmy) state_reset_mode = true;
			else state_reset_mode = false;
		}
		if (strstr(buf, "batch_shuffle"))
		{
			int dmy = 0;
			sscanf(buf, "batch_shuffle:%d", &dmy);
			if (dmy) batch_shuffle = true;
			else batch_shuffle = false;
		}
		if (strstr(buf, "L1_loss"))
		{
			int dmy = 0;
			sscanf(buf, "L1_loss:%d", &dmy);
			if (dmy) L1_loss = true;
			else L1_loss = false;
		}

		//
		if (strstr(buf, "pycode_dump_only"))
		{
			sscanf(buf, "pycode_dump_only:%d", &pycode_dump_only);
		}
		if (strstr(buf, "weight_init_type"))
		{
			sscanf(buf, "weight_init_type:%s", weight_init_type);
		}
		if (strstr(buf, "activation_fnc"))
		{
			sscanf(buf, "activation_fnc:%s", activation_fnc);
		}
		if (strstr(buf, "use_self_sequence"))
		{
			sscanf(buf, "use_self_sequence:%d", &use_self_sequence);
		}
		//

		if (train)
		{
			if (strstr(buf, "maxvalue"))
			{
				sscanf(buf, "maxvalue:%f", &maxvalue);
			}
			if (strstr(buf, "minvalue"))
			{
				sscanf(buf, "minvalue:%f", &minvalue);
			}
		}
	}
	fclose(fp);
	delete[] buf;


	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;
	if (learning_rate >= 1.0)
	{
		learning_rate *= 0.0001;
	}

	//scale = maxvalue;
	//if (fabs(maxvalue) < fabs(minvalue))
	//{
	//	scale = minvalue;
	//}

	//printf("scale:%f\n", scale);

	printf("x_dim:%d\n", x_dim);
	printf("y_dim:%d\n", y_dim);
	printf("sequence_length:%d\n", sequence_length);
	printf("out_sequence_length:%d\n", out_sequence_length);
	printf("target_position:%d\n", target_position);
	printf("clip_gradients:%f\n", clip_gradients);
	printf("rnn_layers:%d\n", rnn_layers);
	printf("n_layers:%d\n", n_layers);
	printf("n_hidden_size:%d\n", n_hidden_size);
	printf("fc_hidden_size:%d\n", fc_hidden_size);
	printf("dropout:%f\n", dropout);
	printf("learning_rate:%f\n", learning_rate);
	printf("early_stopping:%d\n", early_stopping);
	printf("opt_type:%s\n", opt_type);
	printf("n_train_epochs:%d\n", n_train_epochs);
	printf("n_minibatch:%d\n", n_minibatch);
	printf("n_eval_minibatch:%d\n", n_eval_minibatch);
	printf("rnn_type:%s\n", rnn_type);
	printf("state_reset_mode:%d\n", state_reset_mode ? 1 : 0);
	printf("batch_shuffle:%d\n", batch_shuffle ? 1 : 0);
	printf("weight_init_type:%s\n", weight_init_type);
	printf("activation_fnc:%s\n", activation_fnc);
	printf("use_attention:%d\n", use_attention);
	printf("use_cnn:%d\n", use_cnn);
	printf("padding_prm:%d\n", padding_prm);
	printf("residual:%d\n", residual);
	printf("use_self_sequence:%d\n", use_self_sequence);

	printf("(fc)input_size:%d\n", input_size);
	printf("(fc)regression:%s\n", regression);
	printf("classification:%d\n", classification);
	//
	printf("test_mode:%d\n", test_mode);
	printf("pycode_dump_only:%d\n", pycode_dump_only);
}

extern "C" _LIBRARY_EXPORTS int getBatchSize()
{
	return kTrainBatchSize;
}
extern "C" _LIBRARY_EXPORTS int getNumberOfEpochs()
{
	return kNumberOfEpochs;
}
extern "C" _LIBRARY_EXPORTS int getSequence_length()
{
	return sequence_length;
}
extern "C" _LIBRARY_EXPORTS int getYdim()
{
	return y_dim;
}
extern "C" _LIBRARY_EXPORTS int getXdim()
{
	return x_dim;
}
extern "C" _LIBRARY_EXPORTS float getScale()
{
	return scale;
}
extern "C" _LIBRARY_EXPORTS float getTolerance()
{
	return tolerance;
}

extern "C" _LIBRARY_EXPORTS float torch_get_train_loss()
{
	float loss = nn_->get_train_loss();
	return loss;
}

extern "C" _LIBRARY_EXPORTS float torch_get_loss(std::vector<tiny_dnn::vec_t>& train_images_,  std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	float loss = nn_->get_loss(train_images_, train_labels_, batch);
	return loss;
}
extern "C" _LIBRARY_EXPORTS float torch_get_Loss(int batch)
{
	float loss = nn_->get_loss(train_images, train_labels, batch);
	return loss;
}
extern "C" _LIBRARY_EXPORTS float torch_get_loss_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	//printf("train_images_:%d\n", train_images_.size());

	float loss = toNet(nn)->get_loss(train_images_, train_labels_, batch);
	return loss;
}
extern  _LIBRARY_EXPORTS tiny_dnn::result torch_get_accuracy_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	tiny_dnn::result result;
	if (train_images_.size() == 0)
	{
		result.num_total = 1;
		return result;
	}

	result = toNet(nn)->get_accuracy(train_images_, train_labels_, batch);

	//ConfusionMatrix
	std::cout << "ConfusionMatrix:" << std::endl;
	result.print_detail(std::cout);
	std::cout << result.num_success << "/" << result.num_total << std::endl;
	printf("accuracy:%.3f%%\n", result.accuracy());

	return result;
}


extern "C" _LIBRARY_EXPORTS void* torch_model(std::string& ptfile)
{
	cpp_torch::Net bakup_model;
	cpp_torch::network_torch<cpp_torch::Net>* nn2 = new cpp_torch::network_torch<cpp_torch::Net>(bakup_model, device);
	*nn2 = *nn_;

	nn2->load(ptfile);

	//CUDA information is also included in the data that has been learned and serialized by CUDA.
	//nn2->model.get()->to(device);

	return (void*)nn2;
}

extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_invpost_predict(tiny_dnn::vec_t x)
{
	if (scale != 1.0)
	{
		for (auto& xx : x) xx /= scale;
	}
	tiny_dnn::vec_t y = nn3_->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		if (scale != 1.0)
		{
			for (auto& yy : y) yy *= scale;
		}
	}
	return y;
}
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_post_predict(tiny_dnn::vec_t x)
{
	if (scale != 1.0)
	{
		for (auto& xx : x) xx /= scale;
	}
	tiny_dnn::vec_t y = nn2_->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		if (scale != 1.0)
		{
			for (auto& yy : y) yy *= scale;
		}
	}
	return y;
}

extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x)
{
	if (scale != 1.0)
	{
		for (auto& xx : x) xx /= scale;
	}
	tiny_dnn::vec_t y = nn_->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		if (scale != 1.0)
		{
			for (auto& yy : y) yy *= scale;
		}
	}
	return y;
}
extern _LIBRARY_EXPORTS std::vector<tiny_dnn::vec_t> torch_model_predict_batch(const void* nn, std::vector<tiny_dnn::vec_t>& x, int bacth = 1)
{
	cpp_torch::network_torch<cpp_torch::Net>* nn2
		= (cpp_torch::network_torch<cpp_torch::Net>*)(nn);

	if (scale != 1.0)
	{
		for (auto& xx : x) for (auto& xxx : xx) xxx /= scale;
	}
	std::vector <tiny_dnn::vec_t>& y = nn2->predict(x, bacth);
	if (num_class >= 2)
	{
	}
	else
	{
		if (scale != 1.0)
		{
			for (auto& yy : y)for (auto& yyy : yy) yyy *= scale;
		}
	}
	return y;
}
extern _LIBRARY_EXPORTS std::vector<tiny_dnn::vec_t> torch_predict_batch(std::vector<tiny_dnn::vec_t>& x, int bacth = 1)
{
	return  torch_model_predict_batch((void*)nn_, x, bacth);
}

extern _LIBRARY_EXPORTS tiny_dnn::vec_t torch_model_predict(const void* nn, tiny_dnn::vec_t x)
{
	cpp_torch::network_torch<cpp_torch::Net>* nn2
		= (cpp_torch::network_torch<cpp_torch::Net>*)(nn);

	if (scale != 1.0)
	{
		for (auto& xx : x) xx /= scale;
	}
	tiny_dnn::vec_t& y = nn2->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		if (scale != 1.0)
		{
			for (auto& yy : y) yy *= scale;
		}
	}
	return y;
}

extern "C" _LIBRARY_EXPORTS void torch_delete_model()
{
	if (nn_ == nullptr) return;
	delete nn_;

	nn_ = nullptr;

	if (nn2_ == nullptr) return;
	delete nn2_;

	nn2_ = nullptr;

	if (nn3_ == nullptr) return;
	delete nn3_;

	nn3_ = nullptr;
}
extern "C" _LIBRARY_EXPORTS void torch_delete_load_model(void* n)
{
	if (n == nullptr) return;
	delete n;

	n = nullptr;
}


extern "C" _LIBRARY_EXPORTS void* torch_getNet()
{
	return(void*)(nn_);
}

extern "C" _LIBRARY_EXPORTS void* torch_getDevice()
{
	return(void*)(&device);
}
extern "C" _LIBRARY_EXPORTS void torch_setDeviceIndex(const int id)
{
	device.set_index(id);
}
extern "C" _LIBRARY_EXPORTS void torch_setDevice(const char* device_name)
{
	printf("device_name:%s\n", device_name);
	torch::DeviceType device_type;

	c10::DeviceIndex id = c10::DeviceIndex (-1);
	if (std::string(device_name) == std::string("gpu") || std::string(device_name) == std::string("cuda"))
	{
		if (torch::cuda::is_available()) {
			std::cout << "CUDA available! Training on GPU." << std::endl;
			device_type = torch::kCUDA;
		}
		else
		{
			std::cout << "CUDA no available -> Training on CPU." << std::endl;
			device_type = torch::kCPU;
		}
	}
	else
	if (strstr(device_name,"gpu:") || strstr(device_name,"cuda:"))
	{
		if (torch::cuda::is_available()) {
			std::cout << "CUDA available! Training on GPU." << std::endl;
			device_type = torch::kCUDA;
		}
		else
		{
			std::cout << "CUDA no available -> Training on CPU." << std::endl;
			device_type = torch::kCPU;
		}
		char* p = strchr((char*)device_name, ':');
		id = c10::DeviceIndex(atoi(p+1));
	}
	else
	{
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	printf("DeviceIndex:%d\n", id);
	device = torch::Device(device_type, id);
}

extern "C" _LIBRARY_EXPORTS void* torch_progress_display(size_t length)
{
	cpp_torch::progress_display* disp = new cpp_torch::progress_display(length);
	return (void*)disp;
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_restart(void* disp_, size_t length)
{
	cpp_torch::progress_display* disp = (cpp_torch::progress_display*)disp_;
	disp->restart(length);
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_count(void* disp_, int count)
{
	cpp_torch::progress_display* disp = (cpp_torch::progress_display*)disp_;
	(*disp) += count;
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_delete(void* disp_)
{
	cpp_torch::progress_display* disp = (cpp_torch::progress_display*)disp_;
	delete disp;
}

extern "C" _LIBRARY_EXPORTS void torch_stop_ongoing_training()
{
	nn_->stop_ongoing_training();
}

extern "C" _LIBRARY_EXPORTS void torch_save_nn(void* nn, const char* name)
{
	toNet(nn)->save(std::string(name));
}

extern "C" _LIBRARY_EXPORTS void torch_save(const char* name)
{
	nn_->save(std::string(name));
}
extern "C" _LIBRARY_EXPORTS void torch_load(const char* name)
{
	nn_->load(std::string(name));
	
	//CUDA information is also included in the data that has been learned and serialized by CUDA.
	//nn_->model.get()->to(device);
}
extern "C" _LIBRARY_EXPORTS void* torch_load_new(const char* name)
{
	void* nn2 = torch_model(std::string(name));
	return nn2;
}

extern "C" _LIBRARY_EXPORTS int torch_train_custom(
	std::string define_layers_file_name,
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	char buf[4096];
	FILE* fp = fopen(define_layers_file_name.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}

	char* p = NULL;
	int nparam = 0;
	while (fgets(buf, 4096, fp) != NULL)
	{
		if (buf[0] == '\n') continue;
		if (strstr(buf, "device="))
		{
			char dev[32];
			sscanf(buf, "device=%s\n", dev);
			if (strcmp(dev, "cpu") == 0) torch_setDevice("cpu");
			if (strcmp(dev, "gpu") == 0) torch_setDevice("gpu");
			if (strcmp(dev, "cuda") == 0) torch_setDevice("gpu");
			continue;
		}
		if (strstr(buf, "scale="))
		{
			sscanf(buf, "scale=%f", &scale);
			continue;
		}
		if (strstr(buf, "num_class="))
		{
			sscanf(buf, "num_class=%d", &num_class);
			continue;
		}
	}
	fclose(fp);

	if (train_images_.size() > 0 && train_labels_.size() > 0)
	{
		send_train_images(train_images_);

		auto scale_bak = scale;
		if (num_class >= 2)
		{
			scale = 1.0;
		}
		send_train_labels(train_labels_);
		scale = scale_bak;
	}

	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;

	cpp_torch::Net model;

	model.get()->device = device;
	model.get()->pycode_dump_only = pycode_dump_only;

	fp = fopen(define_layers_file_name.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}

	p = NULL;
	while (fgets(buf, 4096, fp) != NULL)
	{
		if (buf[0] == '\n') continue;
		if (strstr(buf, "epoch="))
		{
			sscanf(buf, "epoch=%d", &kNumberOfEpochs);
			continue;
		}
		if (strstr(buf, "train_batch="))
		{
			sscanf(buf, "train_batch=%d", &kTrainBatchSize);
			continue;
		}
		if (strstr(buf, "test_batch="))
		{
			sscanf(buf, "test_batch=%d", &kTestBatchSize);
			continue;
		}
		if (strstr(buf, "learning_rate="))
		{
			sscanf(buf, "learning_rate=%f", &learning_rate);
			continue;
		}
		if (strstr(buf, "optimizer="))
		{
			char solv[32];
			sscanf(buf, "optimizer=%s\n", solv);
			optimizer_name = std::string(solv);
			continue;
		}
		//


		if (strstr(buf, "input"))
		{
			int input_c, input_w, input_h;
			nparam = sscanf(buf, "input c=%d w=%d h=%d", &input_c, &input_w, &input_h);
			if (nparam != 3)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (input_c <= 0 && input_w <= 0 && input_h <= 0)
			{
				input_c = 1;
				input_w = 1;
				input_h = train_images.size();
			}
			model.get()->setInput(input_c, input_w, input_h);
			continue;
		}
		char* token = NULL;
		if (strstr(buf, "fc"))
		{
			int unit = -1, bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "unit=")) != NULL)
				{
					nparam = sscanf(token, "unit=%d", &unit);
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					nparam = sscanf(token, "bias=%d", &bias);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (unit <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (bias == -1) bias = 1;
			model.get()->add_fc(unit, (bias != 0) ? true : false);
			continue;
		}
		/*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		*/
		if (strstr(buf, "conv2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 1, 1 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			bool bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
						dilation[1] = dilation[0];
					}
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					int b;
					nparam = sscanf(token, "bias=%d", &b);
					if (b) bias = true;
					else bias = false;
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			else
			{
				model.get()->add_conv2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ dilation[0],dilation[1] },
					bias
				);
			}
			continue;
		}

		/**
		* constructing transposed convolution layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param out_padding  [in] out_padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		if (strstr(buf, "conv_transpose2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 1, 1 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> out_padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			bool bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "out_padding=")) != NULL)
				{
					nparam = sscanf(token, "out_padding=(%d,%d)", &out_padding[0], &out_padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "out_padding=%d", &out_padding[0]);
						out_padding[1] = out_padding[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
						dilation[1] = dilation[0];
					}
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					int b;
					nparam = sscanf(token, "bias=%d", &b);
					if (b) bias = true;
					else bias = false;
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_conv_transpose2d(
					kernel_size[0],
					stride[0],
					padding[0],
					out_padding[0],
					dilation[0], bias);
			}
			else
			{
				model.get()->add_conv_transpose2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ out_padding[0],out_padding[1] },
					{ dilation[0],dilation[1] }, bias);
			}
			continue;
		}
		/* constructing max pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		**/
		if (strstr(buf, "maxpool2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 0, 0 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
					}
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[0] <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}

			if (stride[0] <= 0 || stride[1] <= 0)
			{
				stride[0] = kernel_size[0];
				stride[1] = kernel_size[1];
			}

			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_maxpool2d(
					kernel_size[0],
					stride[0],
					padding[0],
					dilation[0]
				);
			}
			else
			{
				model.get()->add_maxpool2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ dilation[0],dilation[1] }
				);
			}
			continue;
		}
		/**
		* constructing average pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		**/
		if (strstr(buf, "avgpool2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 0, 0 };
			std::vector<int> padding = { 0, 0 };
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (kernel_size[0] <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (stride[0] <= 0 || stride[1] <= 0)
			{
				stride[0] = kernel_size[0];
				stride[1] = kernel_size[1];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_avgpool2d(
					kernel_size[0],
					stride[0],
					padding[0]
				);
			}
			else
			{
				model.get()->add_avgpool2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] }
				);
			}
			continue;
		}

		if (strstr(buf, "conv_drop"))
		{
			float rate = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rate=")) != NULL)
				{
					nparam = sscanf(token, "rate=%f", &rate);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (rate <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_conv_drop(rate);
			continue;
		}

		if (strstr(buf, "dropout"))
		{
			float rate = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rate=")) != NULL)
				{
					nparam = sscanf(token, "rate=%f", &rate);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (rate <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_dropout(rate);
			continue;
		}
		if (strstr(buf, "pixel_shuffle"))
		{
			float upscale_factor = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "upscale_factor=")) != NULL)
				{
					nparam = sscanf(token, "upscale_factor=%f", &upscale_factor);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (upscale_factor <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_pixel_shuffle(upscale_factor);
			continue;
		}

		/**
		* constructing Batch Normalization layer
		* @param momentum        [in] momentum in the computation of the exponential
		* @param eos             [in] The epsilon value added for numerical stability.
		**/
		if (strstr(buf, "batchnorml2d"))
		{
			float momentum = 0.1;
			float eps = 1e-5;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "momentum=")) != NULL)
				{
					nparam = sscanf(token, "momentum=%f", &momentum);
				}
				if ((token = strstr(buf, "eps=")) != NULL)
				{
					nparam = sscanf(token, "eps=%f", &eps);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (eps <= 0 || momentum <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_bn2d(momentum, eps);
			continue;
		}

		/**
		* compute multi-layer long-short-term-memory (RNN)
		**/
		/**
		* @param rnn_type        [in] "rnn", "lstm", "gru"
		* @param sequence_length [in] number of sequence of length
		* @param hidden_size     [in] number of hidden of size
		* @param num_layers      [in] The number of recurrent layers (cells) to use.
		* @param dropout         [in] If non-zero, adds dropout with the given probability to the output of each RNN layer, except the final layer.
		**/
		if (strstr(buf, "recurrent"))
		{
			char rnn_type[32] = { '\0' };
			int sequence_length = 1;
			int hidden_size = 32;
			int num_layers = 1;
			float dropout = 0;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rnn_type=")) != NULL)
				{
					nparam = sscanf(token, "rnn_type=%s", rnn_type);
				}
				if ((token = strstr(buf, "sequence_length=")) != NULL)
				{
					nparam = sscanf(token, "sequence_length=%d", &sequence_length);
				}
				if ((token = strstr(buf, "hidden_size=")) != NULL)
				{
					nparam = sscanf(token, "hidden_size=%d", &hidden_size);
				}
				if ((token = strstr(buf, "num_layers=")) != NULL)
				{
					nparam = sscanf(token, "num_layers=%d", &num_layers);
				}
				if ((token = strstr(buf, "dropout=")) != NULL)
				{
					nparam = sscanf(token, "dropout=%f", &dropout);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (sequence_length <= 0 || hidden_size <= 0 || num_layers <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (std::string(rnn_type) != std::string("rnn") && std::string(rnn_type) != std::string("gru") && std::string(rnn_type) != std::string("lstm"))
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_recurrent(std::string(rnn_type), sequence_length, hidden_size, num_layers, dropout);
			continue;
		}
		if (strstr(buf, "tanh"))
		{
			model.get()->add_Tanh();
			continue;
		}
		if (strstr(buf, "leaky_relu"))
		{
			float negative_slope = 0.01;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "slope=")) != NULL)
				{
					nparam = sscanf(token, "slope=%f", &negative_slope);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			model.get()->add_LeakyReLU(negative_slope);
			continue;
		}
		if (strstr(buf, "relu"))
		{
			model.get()->add_ReLU();
			continue;
		}
		if (strstr(buf, "selu"))
		{
			model.get()->add_SELU();
			continue;
		}
		if (strstr(buf, "sigmoid"))
		{
			model.get()->add_Sigmoid();
			continue;
		}
		if (strstr(buf, "log_softmax"))
		{
			model.get()->add_LogSoftmax(1);
			continue;
		}
		if (strstr(buf, "softmax"))
		{
			model.get()->add_Softmax(1);
			continue;
		}
	}
	fclose(fp);

	if (model.get()->layer.size() == 0)
	{
		return -2;
	}
	size_t outsize =
		model.get()->layer[model.get()->layer.size() - 1].out_[0] *
		model.get()->layer[model.get()->layer.size() - 1].out_[1] *
		model.get()->layer[model.get()->layer.size() - 1].out_[2];

	if (outsize != train_labels[0].size())
	{
		printf("output size miss match %d != %d\n", outsize, train_labels[0].size());
		return -3;
	}
	//xavier
	for (auto w : model.get()->fc)
	{
		torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
		//auto &x = w->all_weights();
		//for (auto xx : x)
		//{
		//	torch::nn::init::xavier_uniform_(xx, torch::nn::init::calculate_gain(torch::kReLU));
		//}
	}
#if 0
	//uniform
	for (auto w : model.get()->fc)
	{
		torch::nn::init::uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
	}
	//gaussian
	for (auto w : model.get()->fc)
	{
		torch::nn::init::normal_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
	}
#endif

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	nn_->classification = (num_class > 2);
	nn_->batch_shuffle = batch_shuffle;
	printf("num_class=%d\n", num_class);

	torch::optim::Optimizer* optimizer = nullptr;

	float moment = 0.5;
	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	if (optimizer_name == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (optimizer_name == "adam")
	{
		optimizer = &optimizerAdam;
		std::cout << optimizer_name << std::endl;
	}

	if (!test_mode)
	{
		// train
		std::cout << "start training" << std::endl;
		nn_->fit(optimizer, train_images, train_labels, n_minibatch,
			n_train_epochs, on_enumerate_minibatch,
			on_enumerate_epoch);
		std::cout << "end training." << std::endl;
	}
	return 0;
}


extern "C" _LIBRARY_EXPORTS void Train(
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	std::vector<tiny_dnn::vec_t> dummy_x;
	std::vector<tiny_dnn::vec_t> dummy_y;
	return torch_train(
			dummy_x,
			dummy_y,
			n_minibatch,
			n_train_epochs,
			on_enumerate_minibatch,
			on_enumerate_epoch);
}

void add_activatin(cpp_torch::Net& model)
{
	if ( std::string(activation_fnc) == "tanh")model.get()->add_Tanh();
	if ( std::string(activation_fnc) == "relu")model.get()->add_ReLU();
	if ( std::string(activation_fnc) == "leakyrelu")model.get()->add_LeakyReLU(0.2);
	if (std::string(activation_fnc) == "selu")model.get()->add_SELU();
	if (std::string(activation_fnc) == "mish")model.get()->add_Mish();
}

extern "C" _LIBRARY_EXPORTS void torch_train(
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	if (train_images_.size() > 0 && train_labels_.size() > 0)
	{
		send_train_images(train_images_);
		send_train_labels(train_labels_);
	}

	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;

	cpp_torch::Net model;

	size_t input_size = train_images[0].size();
	int hidden_size = n_hidden_size;
	if (hidden_size <= 0) hidden_size = 64;

	model.get()->device = device;
	model.get()->pycode_dump_only = pycode_dump_only;

	model.get()->setInput(1, 1, input_size);

	size_t usize = input_size * 1;
	if (dropout > 0) model.get()->add_dropout(dropout*0.2);
	model.get()->add_fc(usize, false);
	add_activatin(model);

	model.get()->add_fc(input_size, false);
	add_activatin(model);

	bool use_residual = false;
	if (residual == 0)
	{
		use_residual = false;
		residual = 1;
	}
	else
	{
		use_residual = true;
	}
	//A Hybrid CNN-LSTM Model for Forecasting.	Particulate Matter(PM2.5)
	//Anomaly Detection in Control Valves by 1d CNN-LSTM
	if (use_cnn > 0)
	{
		int residual_in = model.get()->getOutputSize();
		for (int r = 0; r < residual; r++)
		{
			if (use_residual)
			{
				model.get()->add_residual_begin();
			}

			int kernel = 3;
			int stride = 2;
			int padding = 1;
			int dilation = 1;
			int in_ch = 1;
			int ot_ch = 3;

			int input_size2 = input_size;
			int size = (int)floor((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);
			if (size <= 0)
			{
				while (size <= 0)
				{
					input_size2 *= 2;
					input_size2 += 1;
					size = (int)ceil((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);

				}
				model.get()->add_fc(input_size2, false);
			}

			int in = model.get()->getOutputSize();
			if (padding_prm)
			{
				padding = ((in - 1)*stride + 1 + dilation * (kernel - 1) - in) / 2.0;
				kernel = padding * 2;
			}
			else
			{
				padding = 0;
			}
			for (int i = 0; i < use_cnn; i++)
			{
				model.get()->add_conv1d(in_ch, ot_ch, kernel, stride, padding);
				if (model.get()->getOutputSize() <= 0)
				{
					printf("number of cnn ERROR.\n"); fflush(stdout);
					throw cpp_torch::error_exception("number of cnn ERROR.");
				}
				if (use_cnn_add_bn)
				{
					model.get()->add_bn1d();
				}
				add_activatin(model);
				in_ch = ot_ch;
				ot_ch *= 2;

				// Aliasing is generated by downsampling, and the high frequency component is impaired, 
				// so xxx pooling not performed.

			}
			if (use_residual)
			{
				model.get()->add_fc(residual_in);
				add_activatin(model);

				model.get()->add_residual_end();
			}
		}
		if (dropout > 0) model.get()->add_dropout(dropout);
		model.get()->add_fc(input_size, false);
		add_activatin(model);
	}

	if (rnn_layers <= 0) rnn_layers = 1;
#ifdef RNN_LAYERS_OPT
	model.get()->add_recurrent(std::string(rnn_type), sequence_length, hidden_size, rnn_layers, dropout);
	add_activatin(model);

	if (use_attention)
	{
		model.get()->add_attaentinon(sequence_length*2, 7);
	}
#else
	model.get()->add_recurrent(std::string(rnn_type), sequence_length, hidden_size);
	add_activatin(model);
	for (int i = 1; i < rnn_layers; i++)
	{
		model.get()->add_recurrent(std::string(rnn_type), 1, hidden_size);
		add_activatin(model);
	}
	if (attention)
	{
		model.get()->add_attaentinon(sequence_length * 2);
	}
#endif

	int n_layers_tmp = n_layers;
	size_t sz = hidden_size / 2;

	//sz = train_labels[0].size() * 10;
	int coef = int(0.9 / log(train_labels[0].size() / 4.0 + 1.0)) + 1;
	sz = train_labels[0].size() * coef;

	for (int i = 0; i < n_layers_tmp; i++) {
		if (dropout > 0 && i == n_layers_tmp - 1) model.get()->add_dropout(dropout);
		if (fc_hidden_size > 0)
		{
			model.get()->add_fc(fc_hidden_size);
		}
		else
		{
			model.get()->add_fc(sz);
		}
		if (use_add_bn)
		{
			model.get()->add_bn1d();
		}
		add_activatin(model);
	}
	model.get()->add_fc(sz);
	add_activatin(model);

	if (classification >= 2)
	{
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_fc(std::min((int)sz, classification * 2));
		add_activatin(model);
		model.get()->add_fc(classification);
		model.get()->add_LogSoftmax(1);
	}
	else
	{
		model.get()->add_sampling();
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_fc(train_labels[0].size());
	}

	//xavier
	if (std::string(weight_init_type) == "xavier")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
		}
	}
#if 1
	//uniform
	if (std::string(weight_init_type) == "uniform")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::uniform_(w->weight);
		}
	}
	//gaussian
	if (std::string(weight_init_type) == "gaussian")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::normal_(w->weight);
		}
	}
	if (std::string(weight_init_type) == "constant")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::constant_(w->weight, 0);
		}
	}
	if (std::string(weight_init_type) == "he")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::kaiming_normal_(w->weight);
		}
	}
#endif
	for (auto w : model.get()->lstm)
	{
		//auto &x = w->all_weights();
		//for (auto xx : x)
		//{
		//	torch::nn::init::xavier_uniform_(xx, torch::nn::init::calculate_gain(torch::kReLU));
		//}
	}

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	//nn_->classification = false;
	nn_->classification = (classification >= 2);
	nn_->batch_shuffle = batch_shuffle;
	nn_->set_clip_grad_norm(clip_gradients);
	nn_->set_early_stopping(early_stopping);

	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	auto optimizerRMSprop =
		torch::optim::RMSprop(model.get()->parameters(), 
			torch::optim::RMSpropOptions(learning_rate));

	auto optimizerAdagrad =
		torch::optim::Adagrad(model.get()->parameters(),
			torch::optim::AdagradOptions(learning_rate));

	if (std::string(opt_type) == "sgd")
	{
		optimizer = &optimizerSGD;
	}
	if (std::string(opt_type) == "adam")
	{
		optimizer = &optimizerAdam;
	}
	if (std::string(opt_type) == "adagrad")
	{
		optimizer = &optimizerAdagrad;
	}
	if (std::string(opt_type) == "rmsprop")
	{
		optimizer = &optimizerRMSprop;
	}
	nn_->optimizer_name = opt_type;
	if (optimizer == nullptr)
	{
		nn_->optimizer_name = "adam";
		optimizer = &optimizerAdam;
	}

	if (!test_mode)
	{
		// train
		std::cout << "start training" << std::endl;
		nn_->fit(optimizer, train_images, train_labels, n_minibatch,
			n_train_epochs, on_enumerate_minibatch,
			on_enumerate_epoch);
		std::cout << "end training." << std::endl;
	}
}

extern "C" _LIBRARY_EXPORTS void torch_train_fc(
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	if (train_images_.size() > 0 && train_labels_.size() > 0)
	{
		send_train_images(train_images_);
		send_train_labels(train_labels_);
	}

	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;

	int hidden_size = train_images[0].size() * 50;

	cpp_torch::Net model;


	model.get()->device = device;
	model.get()->pycode_dump_only = pycode_dump_only;

	model.get()->setInput(1, 1, train_images[0].size());


	if (regression == "linear" || regression == "logistic")
	{
		/**/
	}
	else
	{
		model.get()->add_fc(input_size);
		add_activatin(model);

		//printf("@dropout:%f\n", dropout);

		for (int i = 0; i < n_layers; i++) {
			if (dropout > 0 && i == 0) model.get()->add_dropout(dropout*0.2);
			if (dropout > 0 && i == n_layers-2) model.get()->add_dropout(dropout);
			model.get()->add_fc(input_size);
			add_activatin(model);
			if (use_add_bn)
			{
				model.get()->add_bn1d();
			}
		}
	}

	bool use_residual = false;
	if (residual == 0)
	{
		use_residual = false;
		residual = 1;
	}
	else
	{
		use_residual = true;
	}
	//printf("--->classification=%d\n", classification);
	if (use_cnn > 0 && regression != "linear" && regression != "logistic")
	{
		int residual_in = model.get()->getOutputSize();
		for (int r = 0; r < residual; r++)
		{
			if (use_residual)
			{
				model.get()->add_residual_begin();
			}

			//int i_ch = 1;
			//int o_ch = 8;
			//int kernel = 3;
			//int stride = 2;
			//int padding = 1;
			//int dilation = 1;
			int kernel = 3;
			int stride = 2;
			int padding = 1;
			int dilation = 1;
			int i_ch = 1;
			int o_ch = 3;

			int input_size2 = input_size;
			int size = (int)ceil((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);
			if (size <= 0)
			{
				while( size <= 0)
				{
					input_size2 *= 2;
					input_size2 += 1;
					size = (int)ceil((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);

				}
				model.get()->add_fc(input_size2, false);
			}

			int in = model.get()->getOutputSize();
			if (padding_prm)
			{
				padding = ((in - 1)*stride + 1 + dilation * (kernel - 1) - in) / 2.0;
				kernel = padding * 2;
			}
			else
			{
				padding = 0;
			}
			for (int k = 0; k < use_cnn; k++)
			{

				if (model.get()->getOutputSize() <= 0)
				{
					printf("number of cnn ERROR.\n"); fflush(stdout);
					throw cpp_torch::error_exception("number of cnn ERROR.");
				}
				model.get()->add_conv1d(i_ch, o_ch, kernel, stride, padding, dilation);

				if (use_cnn_add_bn)
				{
					model.get()->add_bn1d();
				}
				add_activatin(model);
				model.get()->add_maxpool1d(kernel, stride, padding);
				i_ch = o_ch;
				o_ch *= 2;
			}

			if (use_residual)
			{
				model.get()->add_fc(residual_in);
				add_activatin(model);

				model.get()->add_residual_end();
			}
		}
	}

	if (classification >= 2)
	{
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_fc(std::min((int)input_size, classification * 2));
		add_activatin(model);
		model.get()->add_fc(classification);
	}
	else
	{
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_sampling();
		model.get()->add_fc(train_labels[0].size());
	}

	if (regression == "logistic")
	{
		model.get()->add_Sigmoid();
	}
	if (classification >= 2)
	{
		model.get()->add_LogSoftmax(1);
	}


	//xavier
	if (std::string(weight_init_type) == "xavier")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
		}
	}
#if 1
	//uniform
	if (std::string(weight_init_type) == "uniform")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::uniform_(w->weight);
		}
	}
	//gaussian
	if (std::string(weight_init_type) == "gaussian")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::normal_(w->weight);
		}
	}
	if (std::string(weight_init_type) == "constant")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::constant_(w->weight, 0);
		}
	}
	if (std::string(weight_init_type) == "he")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::kaiming_normal_(w->weight);
		}
	}
#endif

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	nn_->classification = (classification >= 2);
	nn_->batch_shuffle = batch_shuffle;
	nn_->shuffle_seed = shuffle_seed;
	nn_->set_early_stopping(early_stopping);
	nn_->set_L1_loss(L1_loss);


	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	auto optimizerAdam_ = torch::optim::Adam(
		model.get()->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

	auto optimizerRMSprop =
		torch::optim::RMSprop(model.get()->parameters(),
			torch::optim::RMSpropOptions(learning_rate));

	auto optimizerAdagrad =
		torch::optim::Adagrad(model.get()->parameters(),
			torch::optim::AdagradOptions(learning_rate));

	if (std::string(opt_type) == "sgd")
	{
		optimizer = &optimizerSGD;
	}
	if (std::string(opt_type) == "adam")
	{
		optimizer = &optimizerAdam;
	}
	if (std::string(opt_type) == "adam_")
	{
		optimizer = &optimizerAdam_;
	}
	if (std::string(opt_type) == "adagrad")
	{
		optimizer = &optimizerAdagrad;
	}
	if (std::string(opt_type) == "rmsprop")
	{
		optimizer = &optimizerRMSprop;
	}

	nn_->optimizer_name = opt_type;
	if (optimizer == nullptr)
	{
		nn_->optimizer_name = "adam";
		optimizer = &optimizerAdam;
	}

	if (!test_mode)
	{
		// train
		std::cout << "start training" << std::endl;
		nn_->fit(optimizer, train_images, train_labels, n_minibatch,
			n_train_epochs, on_enumerate_minibatch,
			on_enumerate_epoch);
		std::cout << "end training." << std::endl;
	}
}

void create_fc_model(
	cpp_torch::Net& model,
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression
	)
{

	int hidden_size = train_images_[0].size() * 50;


	model.get()->device = device;
	model.get()->pycode_dump_only = pycode_dump_only;

	model.get()->setInput(1, 1, train_images_[0].size());


	if (regression == "linear" || regression == "logistic")
	{
		/**/
	}
	else
	{
		model.get()->add_fc(input_size);
		add_activatin(model);

		//printf("@dropout:%f\n", dropout);

		for (int i = 0; i < n_layers; i++) {
			if (dropout > 0 && i == 0) model.get()->add_dropout(dropout * 0.2);
			if (dropout > 0 && i == n_layers - 2) model.get()->add_dropout(dropout);
			model.get()->add_fc(input_size);
			add_activatin(model);
			if (use_add_bn)
			{
				model.get()->add_bn1d();
			}
		}
	}

	bool use_residual = false;
	if (residual == 0)
	{
		use_residual = false;
		residual = 1;
	}
	else
	{
		use_residual = true;
	}
	//printf("--->classification=%d\n", classification);
	if (use_cnn > 0 && regression != "linear" && regression != "logistic")
	{
		int residual_in = model.get()->getOutputSize();
		for (int r = 0; r < residual; r++)
		{
			if (use_residual)
			{
				model.get()->add_residual_begin();
			}

			//int i_ch = 1;
			//int o_ch = 8;
			//int kernel = 3;
			//int stride = 2;
			//int padding = 1;
			//int dilation = 1;
			int kernel = 3;
			int stride = 2;
			int padding = 1;
			int dilation = 1;
			int i_ch = 1;
			int o_ch = 3;

			int input_size2 = input_size;
			int size = (int)ceil((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);
			if (size <= 0)
			{
				while (size <= 0)
				{
					input_size2 *= 2;
					input_size2 += 1;
					size = (int)ceil((double)(input_size2 + 2 * padding - dilation * (kernel - 1) - 1) / (double)stride + 1);

				}
				model.get()->add_fc(input_size2, false);
			}

			int in = model.get()->getOutputSize();
			if (padding_prm)
			{
				padding = ((in - 1) * stride + 1 + dilation * (kernel - 1) - in) / 2.0;
				kernel = padding * 2;
			}
			else
			{
				padding = 0;
			}
			for (int k = 0; k < use_cnn; k++)
			{

				if (model.get()->getOutputSize() <= 0)
				{
					printf("number of cnn ERROR.\n"); fflush(stdout);
					throw cpp_torch::error_exception("number of cnn ERROR.");
				}
				model.get()->add_conv1d(i_ch, o_ch, kernel, stride, padding, dilation);

				if (use_cnn_add_bn)
				{
					model.get()->add_bn1d();
				}
				add_activatin(model);
				model.get()->add_maxpool1d(kernel, stride, padding);
				i_ch = o_ch;
				o_ch *= 2;
			}

			if (use_residual)
			{
				model.get()->add_fc(residual_in);
				add_activatin(model);

				model.get()->add_residual_end();
			}
		}
	}

	if (classification >= 2)
	{
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_fc(std::min((int)input_size, classification * 2));
		add_activatin(model);
		model.get()->add_fc(classification);
	}
	else
	{
		//if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_sampling();
		model.get()->add_fc(train_labels_[0].size());
	}

	if (regression == "logistic")
	{
		model.get()->add_Sigmoid();
	}
	if (classification >= 2)
	{
		model.get()->add_LogSoftmax(1);
	}


	//xavier
	if (std::string(weight_init_type) == "xavier")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
		}
	}
#if 1
	//uniform
	if (std::string(weight_init_type) == "uniform")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::uniform_(w->weight);
		}
	}
	//gaussian
	if (std::string(weight_init_type) == "gaussian")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::normal_(w->weight);
		}
	}
	if (std::string(weight_init_type) == "constant")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::constant_(w->weight, 0);
		}
	}
	if (std::string(weight_init_type) == "he")
	{
		for (auto w : model.get()->fc)
		{
			torch::nn::init::kaiming_normal_(w->weight);
		}
	}
#endif
}
extern "C" _LIBRARY_EXPORTS void torch_train_post_fc(
	std::vector<tiny_dnn::vec_t>&train_images_,
	std::vector<tiny_dnn::vec_t>&train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	if (train_images_.size() > 0 && train_labels_.size() > 0)
	{
		send_train_images(train_images_);
		send_train_labels(train_labels_);
	}

	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;

	int hidden_size = train_images[0].size() * 50;

	cpp_torch::Net model;
	cpp_torch::Net model2;
	cpp_torch::Net model3;

	create_fc_model(
		model,
		train_images_,
		train_labels_,
		n_minibatch,
		n_train_epochs,
		regression
	);

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	nn_->classification = (classification >= 2);
	nn_->batch_shuffle = batch_shuffle;
	nn_->shuffle_seed = shuffle_seed;
	nn_->set_early_stopping(early_stopping);
	nn_->set_L1_loss(L1_loss);


	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	auto optimizerAdam_ = torch::optim::Adam(
		model.get()->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

	auto optimizerRMSprop =
		torch::optim::RMSprop(model.get()->parameters(),
			torch::optim::RMSpropOptions(learning_rate));

	auto optimizerAdagrad =
		torch::optim::Adagrad(model.get()->parameters(),
			torch::optim::AdagradOptions(learning_rate));

	if (std::string(opt_type) == "sgd")
	{
		optimizer = &optimizerSGD;
	}
	if (std::string(opt_type) == "adam")
	{
		optimizer = &optimizerAdam;
	}
	if (std::string(opt_type) == "adam_")
	{
		optimizer = &optimizerAdam_;
	}
	if (std::string(opt_type) == "adagrad")
	{
		optimizer = &optimizerAdagrad;
	}
	if (std::string(opt_type) == "rmsprop")
	{
		optimizer = &optimizerRMSprop;
	}

	nn_->optimizer_name = opt_type;
	if (optimizer == nullptr)
	{
		nn_->optimizer_name = "adam";
		optimizer = &optimizerAdam;
	}

	create_fc_model(
		model2,
		train_labels_,
		train_labels_,
		n_minibatch,
		n_train_epochs,
		regression
	);

	nn2_ = new cpp_torch::network_torch<cpp_torch::Net>(model2, device);

	nn2_->input_dim(1, 1, train_labels_[0].size());
	nn2_->output_dim(1, 1, train_labels[0].size());
	nn2_->classification = (classification >= 2);
	nn2_->batch_shuffle = batch_shuffle;
	nn2_->shuffle_seed = shuffle_seed;
	nn2_->set_early_stopping(early_stopping);
	nn2_->set_L1_loss(L1_loss);	

	torch::optim::Optimizer* optimizer2 = nullptr;

	auto optimizerSGD2 = torch::optim::SGD(
		model2.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam2 =
		torch::optim::Adam(model2.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	auto optimizerAdam2_ = torch::optim::Adam(
		model2.get()->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

	auto optimizerRMSprop2 =
		torch::optim::RMSprop(model2.get()->parameters(),
			torch::optim::RMSpropOptions(learning_rate));

	auto optimizerAdagrad2 =
		torch::optim::Adagrad(model2.get()->parameters(),
			torch::optim::AdagradOptions(learning_rate));

	if (std::string(opt_type) == "sgd")
	{
		optimizer2 = &optimizerSGD2;
	}
	if (std::string(opt_type) == "adam")
	{
		optimizer2 = &optimizerAdam2;
	}
	if (std::string(opt_type) == "adam_")
	{
		optimizer2 = &optimizerAdam2_;
	}
	if (std::string(opt_type) == "adagrad")
	{
		optimizer2 = &optimizerAdagrad2;
	}
	if (std::string(opt_type) == "rmsprop")
	{
		optimizer2 = &optimizerRMSprop2;
	}

	nn2_->optimizer_name = opt_type;
	if (optimizer2 == nullptr)
	{
		nn2_->optimizer_name = "adam";
		optimizer2 = &optimizerAdam2;
	}

	create_fc_model(
		model3,
		train_labels_,
		train_labels_,
		n_minibatch,
		n_train_epochs,
		regression
	);

	nn3_ = new cpp_torch::network_torch<cpp_torch::Net>(model3, device);

	nn3_->input_dim(1, 1, train_labels_[0].size());
	nn3_->output_dim(1, 1, train_labels[0].size());
	nn3_->classification = (classification >= 2);
	nn3_->batch_shuffle = batch_shuffle;
	nn3_->shuffle_seed = shuffle_seed;
	nn3_->set_early_stopping(early_stopping);
	nn3_->set_L1_loss(L1_loss);

	torch::optim::Optimizer* optimizer3 = nullptr;

	auto optimizerSGD3 = torch::optim::SGD(
		model3.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam3 =
		torch::optim::Adam(model3.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	auto optimizerAdam3_ = torch::optim::Adam(
		model3.get()->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

	auto optimizerRMSprop3 =
		torch::optim::RMSprop(model3.get()->parameters(),
			torch::optim::RMSpropOptions(learning_rate));

	auto optimizerAdagrad3 =
		torch::optim::Adagrad(model3.get()->parameters(),
			torch::optim::AdagradOptions(learning_rate));

	if (std::string(opt_type) == "sgd")
	{
		optimizer3 = &optimizerSGD3;
	}
	if (std::string(opt_type) == "adam")
	{
		optimizer3 = &optimizerAdam3;
	}
	if (std::string(opt_type) == "adam_")
	{
		optimizer3 = &optimizerAdam3_;
	}
	if (std::string(opt_type) == "adagrad")
	{
		optimizer3 = &optimizerAdagrad3;
	}
	if (std::string(opt_type) == "rmsprop")
	{
		optimizer3 = &optimizerRMSprop3;
	}

	nn3_->optimizer_name = opt_type;
	if (optimizer3 == nullptr)
	{
		nn3_->optimizer_name = "adam";
		optimizer3 = &optimizerAdam3;
	}

	if (!test_mode)
	{
		// train
		std::cout << "start training" << std::endl;
		nn_->post_fit(model2, model3, optimizer, optimizer2, optimizer3, train_images, train_labels, n_minibatch,
			n_train_epochs, on_enumerate_minibatch,
			on_enumerate_epoch);
		std::cout << "end training." << std::endl;
	}
}

_LIBRARY_EXPORTS int torch_train_init()
{
	torch::manual_seed(1);
	torch::cuda::manual_seed(1);
	torch::cuda::manual_seed_all(1);
	return 0;
}

_LIBRARY_EXPORTS int torch_train_init_seed(int seed)
{
	torch::manual_seed(seed);
	torch::cuda::manual_seed(seed);
	torch::cuda::manual_seed_all(seed);
	return 0;
}

