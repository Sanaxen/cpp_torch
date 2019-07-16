/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include <torch/torch.h>

//#include <cstddef>
//#include <cstdio>
//#include <iostream>
#include <string>
#include <vector>

#include "libtorch_utils.h"
#include "libtorch_sequential_layer_model.h"
#include "util/Progress.hpp"
#include "util/download_data_set.h"
#include "csvreader.h"
#include "libtorch_link_libs.hpp"


//#define TEST1
#define TEST2

#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 32;

// The batch size for testing.
const int64_t kTestBatchSize = 10;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 800;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int sequence_length = 20;
const int out_sequence_length = 1;
const int hidden_size = 64;
int x_dim = 1;
int y_dim = 3;

#define TEST

struct NetImpl : torch::nn::Module {
	NetImpl()
		: 
		lstm({ nullptr }),
		fc1(sequence_length*hidden_size, 128),
		fc2(128, 128),
		fc3(128, 50),
		fc4(50, y_dim*out_sequence_length)
	{
		auto opt = torch::nn::LSTMOptions(y_dim, hidden_size);
		opt = opt.batch_first(true);

		lstm = torch::nn::LSTM(opt);
		lstm.get()->options.batch_first(true);
		fc1.get()->options.with_bias(false);
		fc2.get()->options.with_bias(false);
		fc3.get()->options.with_bias(false);
		fc4.get()->options.with_bias(false);

		register_module("lstm", lstm);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("fc3", fc3);
		register_module("fc4", fc4);
	}

	torch::Tensor forward(torch::Tensor x) {

		int batch = x.sizes()[0];
		//cpp_torch::dump_dim("X", x);
		x = x.view({ batch, -1, y_dim});
		//cpp_torch::dump_dim("X", x);
		x = lstm->forward(x).output;
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim("X", x);
		x = x.view({ batch,  -1});
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(fc1(x));
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(fc2(x));
		x = torch::tanh(fc3(x));
		x = torch::tanh(fc4(x));

		return x;
	}

	torch::nn::LSTM lstm;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
	torch::nn::Linear fc4;
};
TORCH_MODULE(Net); // creates module holder for NetImpl


// load MNIST dataset
std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;
std::vector<tiny_dnn::vec_t> iX, iY;
std::vector<tiny_dnn::vec_t> nX, nY;
std::vector<tiny_dnn::vec_t> dataset;
tiny_dnn::vec_t dataset_min, dataset_maxmin;

void add_seq(tiny_dnn::vec_t& y, tiny_dnn::vec_t& Y)
{
	for (int i = 0; i < y.size(); i++)
	{
		Y.push_back(y[i]);
	}
}
tiny_dnn::vec_t seq_vec(tiny_dnn::tensor_t& ny, int start)
{

	tiny_dnn::vec_t seq;
	for (int k = 0; k < sequence_length; k++)
	{
		add_seq(ny[start + k], seq);
	}
	return seq;
}

int error = 0;
int data_set(float test = 0.3f)
{
	train_images.clear();
	train_labels.clear();
	test_images.clear();
	test_images.clear();

	printf("n_minibatch:%d sequence_length:%d\n", kTrainBatchSize, sequence_length);
	printf("out_sequence_length:%d\n", out_sequence_length);

	size_t dataAll = iY.size() - sequence_length - out_sequence_length;
	printf("dataset All:%d->", dataAll);
	size_t test_Num = dataAll*test;
	int datasetNum = dataAll - test_Num;
	//datasetNum = (size_t)((float)datasetNum / (float)kTrainBatchSize);
	//datasetNum = kTrainBatchSize*datasetNum;
	//test_Num = dataAll - datasetNum;

	//datasetNum = datasetNum - datasetNum % n_minibatch;
	if (datasetNum == 0 || datasetNum < kTrainBatchSize)
	{
		printf("Too many min_batch or Sequence length\n");
		error = -1;
		return error;
	}
	size_t train_num_max = datasetNum;
	printf("train:%d test:%d\n", datasetNum, test_Num);

	for (int i = 0; i < train_num_max; i++)
	{
		train_images.push_back(seq_vec(nY, i));


		tiny_dnn::vec_t y;
		for (int j = 0; j < out_sequence_length; j++)
		{
			const auto& ny = nY[i + sequence_length + j];
			for (int k = 0; k < y_dim; k++)
			{
				y.push_back(ny[k]);
			}
		}

		train_labels.push_back(y);
	}

	for (int i = train_num_max; i < dataAll; i++)
	{
		test_images.push_back(seq_vec(nY, i));

		tiny_dnn::vec_t y;
		for (int j = 0; j < out_sequence_length; j++)
		{
			const auto& ny = nY[i + sequence_length + j];
			for (int k = 0; k < y_dim; k++)
			{
				y.push_back(ny[k]);
			}
		}

		test_labels.push_back(y);
	}
	printf("train:%d test:%d\n", train_images.size(), test_images.size());
	return 0;
}

void read_rnn_dataset(const std::string &data_dir_path)
{
	CSVReader csv(data_dir_path + "/sample.csv", ',', false);
	dataset = csv.toTensor();

	for (int i = 0; i < dataset.size(); i++)
	{
		tiny_dnn::vec_t image;
		image.push_back(dataset[i][0]);
		iX.push_back(image);

		tiny_dnn::vec_t label;
		for (int k = 1; k < dataset[i].size(); k++)
		{
			label.push_back(dataset[i][k]);
		}
		iY.push_back(label);
	}
	printf("y_dim:%d == %d\n", y_dim, iY[0].size());

	nY = iY;
	//data normalize
	tiny_dnn::cpp_torch::normalizeMinMax(nY, dataset_min, dataset_maxmin);

	data_set(0.3);
}

void sequence_test(
#ifndef TEST
cpp_torch::network_torch<Net> nn
#else
cpp_torch::network_torch<cpp_torch::Net> nn
#endif
)
{
	std::vector<tiny_dnn::vec_t> predict_y;
	FILE* fp = fopen("predict.dat", "w");
	float dt = iX[1][0] - iX[0][0];
	for (int i = 0; i < train_images.size(); i++)
	{
		tiny_dnn::vec_t y = nn.predict(train_images[i]);
		predict_y.push_back(y);

		fprintf(fp, "%f", iX[i + sequence_length][0]);
		for (int k = 0; k < 3; k++)
		{
			fprintf(fp, " %f %f", y[k] * dataset_maxmin[k] + dataset_min[k], iY[i + sequence_length][k]);
		}
		fprintf(fp, "\n");
	}

	float t = iX[train_images.size()+sequence_length][0];
	for (int i = train_images.size(); i < iY.size() - sequence_length; i++)
	{
		tiny_dnn::vec_t pre_y = seq_vec(predict_y, i - sequence_length);

		tiny_dnn::vec_t y = nn.predict(pre_y);
		predict_y.push_back(y);

		fprintf(fp, "%f",t);
		for (int k = 0; k < 3; k++)
		{
			fprintf(fp, " %f %f", y[k] * dataset_maxmin[k] + dataset_min[k], 0.0);
		}
		fprintf(fp, "\n");
		t += dt;
	}

	fclose(fp);
}

void learning_and_test_rnn_dataset(torch::Device device)
{

#ifndef TEST
	Net model;
#endif

#ifdef TEST
	cpp_torch::Net model;

	model.get()->device = device;
	model.get()->setInput(1, 1, y_dim*sequence_length);

	model.get()->add_lstm(sequence_length, hidden_size);
	model.get()->add_Tanh();
	model.get()->add_fc(128);
	model.get()->add_Tanh();
	model.get()->add_fc(128);
	model.get()->add_Tanh();
	model.get()->add_fc(y_dim*out_sequence_length);
#endif


#ifndef TEST
	cpp_torch::network_torch<Net> nn(model, device);
#else
	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);
#endif

	nn.input_dim(1, 1, y_dim*sequence_length);
	nn.output_dim(1, 1, y_dim*out_sequence_length);
	nn.classification = false;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;


	torch::optim::SGD optimizer(
		model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
	//auto optimizer =
	//	torch::optim::Adam(model.get()->parameters(),
	//		torch::optim::AdamOptions(0.01));

	FILE* lossfp = fopen("loss.dat", "w");
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (epoch % kLogInterval == 0)
		{
			float loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
			std::cout << "loss :" << loss << std::endl;
			fprintf(lossfp, "%f\n", loss);
			fflush(lossfp);

			sequence_test(nn);
		}
		++epoch;

		if (epoch <= kNumberOfEpochs)
		{
			disp.restart(train_images.size());
		}
		t.restart();
	};

	int batch = 1;
	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
		batch++;
	};

	nn.set_tolerance(0.01, 0.001, 5);

	// train
	nn.fit(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	fclose(lossfp);

	float_t loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
	printf("loss:%f\n", loss);
	std::vector<int>& res = nn.test_tolerance(test_images, test_labels);
	{
		cpp_torch::textColor color("YELLOW");
		cpp_torch::print_ConfusionMatrix(res, nn.get_tolerance());
	}

	nn.test(test_images, test_labels, kTestBatchSize);


}


auto main() -> int {

	torch::manual_seed(1);

	torch::DeviceType device_type;
#ifdef USE_CUDA
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else
#endif
	{
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	read_rnn_dataset(std::string(kDataRoot));

	learning_and_test_rnn_dataset(device);
}
