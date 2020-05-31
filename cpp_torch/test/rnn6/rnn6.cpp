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

	// The number of epochs to train.
	int64_t kNumberOfEpochs = 0;


	cpp_torch::network_torch<cpp_torch::Net>* nn_ = nullptr;
	torch::Device device(torch::kCPU);
	int test_mode = 0;
	float maxvalue = 0.0;
	float minvalue = 0.0;

	int x_dim = 0;
	int y_dim = 0;
	int sequence_length = 0;
	int rnn_layers = 0;
	int n_layers = 0;
	int n_hidden_size = 0;
	int fc_hidden_size = 0;
	float dropout = 0;
	float learning_rate = 0;
	char opt_type[32] = { '\0' };
	int n_train_epochs = 0;
	int n_minibatch = 0;
	int prophecy = 10;
	float tolerance = 1.0e-6;
	char rnn_type[16] = { '\0' };

	char regression[16] = { '\0' };
	int input_size = 0;
	int classification = 0;

	float moment = 0.01;
	float scale = 1.0;


	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;
};
using namespace rnn_dll_variables;

void state_reset(std::string& rnn_type , cpp_torch::Net& model)
{
	//if (rnn_type == "lstm")
	//{
	//	for (int i = 0; i < model.get()->lstm.size(); i++)
	//	{
	//		model.get()->lstm[i].get()->initial_state();
	//	}
	//}
	//if (rnn_type == "gru")
	//{
	//	for (int i = 0; i < model.get()->gru.size(); i++)
	//	{
	//		model.get()->gru[i].get()->reset();
	//	}
	//}
	//if (rnn_type == "rnn")
	//{
	//	for (int i = 0; i < model.get()->rnn.size(); i++)
	//	{
	//		model.get()->rnn[i].get()->reset();
	//	}
	//}
}


extern "C" _LIBRARY_EXPORTS void torch_getData(const char* filename, std::vector<tiny_dnn::vec_t>& data)
{
	CSVReader csv(filename, ',', false);
	data = csv.toTensor();
}
void scaling_data(std::vector<tiny_dnn::vec_t>& data, std::vector<tiny_dnn::vec_t>& dataset)
{
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

cpp_torch::network_torch<cpp_torch::Net>* toNet(void* nn)
{
	return (cpp_torch::network_torch<cpp_torch::Net>*)nn;
}

extern "C" _LIBRARY_EXPORTS void send_train_images(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(train_images, data);
}
extern "C" _LIBRARY_EXPORTS void send_train_labels(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(train_labels, data);
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

	scale = maxvalue;
	if (fabs(maxvalue) < fabs(minvalue))
	{
		scale = minvalue;
	}

	printf("scale:%f\n", scale);

	printf("x_dim:%d\n", x_dim);
	printf("y_dim:%d\n", y_dim);
	printf("sequence_length:%d\n", sequence_length);
	printf("rnn_layers:%d\n", rnn_layers);
	printf("n_layers:%d\n", n_layers);
	printf("n_hidden_size:%d\n", n_hidden_size);
	printf("fc_hidden_size:%d\n", fc_hidden_size);
	printf("dropout:%f\n", dropout);
	printf("learning_rate:%f\n", learning_rate);
	printf("opt_type:%s\n", opt_type);
	printf("n_train_epochs:%d\n", n_train_epochs);
	printf("n_minibatch:%d\n", n_minibatch);
	printf("rnn_type:%s\n", rnn_type);

	printf("(fc)input_size:%d\n", input_size);
	printf("(fc)regression:%s\n", regression);
	printf("(fc)classification:%d\n", classification);
	//
	printf("test_mode:%d\n", test_mode);
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

	result = toNet(nn)->get_accuracy(train_images_, train_labels_);

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

	return (void*)nn2;
}


extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x)
{
	for (auto& xx : x) xx /= scale;
	tiny_dnn::vec_t y = nn_->predict(x);
	for (auto& yy : y) yy *= scale;
	return y;
}
extern _LIBRARY_EXPORTS tiny_dnn::vec_t torch_model_predict(const void* nn, tiny_dnn::vec_t x)
{
	cpp_torch::network_torch<cpp_torch::Net>* nn2
		= (cpp_torch::network_torch<cpp_torch::Net>*)(nn);

	for (auto& xx : x) xx /= scale;

	tiny_dnn::vec_t y = nn2->predict(x);
	for (auto& yy : y) yy *= scale;
	return y;
}

extern "C" _LIBRARY_EXPORTS void torch_delete_model()
{
	if (nn_ == nullptr) return;
	delete nn_;

	nn_ = nullptr;
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
extern "C" _LIBRARY_EXPORTS void* torch_setDevice(const char* device_name)
{
	torch::DeviceType device_type;

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
	{
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	device = torch::Device(device_type);
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
}
extern "C" _LIBRARY_EXPORTS void* torch_load_new(const char* name)
{
	void* nn2 = torch_model(std::string(name));
	return nn2;
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
	
	model.get()->setInput(1, 1, input_size);

	size_t usize = input_size * 1;
	model.get()->add_fc(usize, false);
	model.get()->add_Tanh();

	model.get()->add_fc(input_size, false);
	model.get()->add_Tanh();

	if (rnn_layers <= 0) rnn_layers = 1;
#ifdef RNN_LAYERS_OPT
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size, rnn_layers, 0.1);
	model.get()->add_Tanh();
#else
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size);
	model.get()->add_Tanh();
	for (int i = 1; i < rnn_layers; i++)
	{
		model.get()->add_recurrent(std::string("lstm"), 1, hidden_size);
		model.get()->add_Tanh();
	}
#endif

	int n_layers_tmp = n_layers;
	size_t sz = hidden_size / 2;

	sz = train_labels[0].size() * 10;
	for (int i = 0; i < n_layers_tmp; i++) {
		if (dropout && i == n_layers_tmp - 1) model.get()->add_dropout(dropout);
		if (fc_hidden_size > 0)
		{
			model.get()->add_fc(fc_hidden_size);
		}
		else
		{
			model.get()->add_fc(sz);
		}
		model.get()->add_Tanh();
	}
	model.get()->add_fc(train_labels[0].size());
	model.get()->add_Tanh();

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
	nn_->classification = false;
	nn_->batch_shuffle = false;


	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	if (std::string(opt_type) == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (std::string(opt_type) == "adam")
	{
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

	model.get()->setInput(1, 1, train_images[0].size());


	if (regression == "linear" || regression == "logistic")
	{
		/**/
	}
	else
	{
		model.get()->add_fc(input_size);
		model.get()->add_Tanh();

		for (int i = 0; i < n_layers; i++) {
			if (dropout && i == n_layers - 1) model.get()->add_dropout(dropout);
			model.get()->add_fc(input_size);
			model.get()->add_Tanh();
		}
	}
	if (classification >= 2)
	{
		if (dropout) model.get()->add_dropout(dropout);
		model.get()->add_fc(std::min((int)input_size, classification * 2));
		model.get()->add_Tanh();
		model.get()->add_fc(classification);
	}
	else
	{
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
	for (auto w : model.get()->fc)
	{
		torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
#if 0
	//uniform
	for (auto w : model.get()->fc)
	{
		torch::nn::init::uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	//gaussian
	for (auto w : model.get()->fc)
	{
		torch::nn::init::normal_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
#endif

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	nn_->classification = (classification >= 2);
	nn_->batch_shuffle = true;


	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	if (std::string(opt_type) == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (std::string(opt_type) == "adam")
	{
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

_LIBRARY_EXPORTS int torch_train_init()
{
	torch::manual_seed(1);
	return 0;
}

