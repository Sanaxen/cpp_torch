/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../../torch_util/libtorch_utils.h"
#include "../../torch_util/libtorch_sequential_layer_model.h"
#include "../../torch_util/csvreader.h"

#include "../../torch_util/libtorch_link_libs.hpp"


#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 128;

// The batch size for testing.
const int64_t kTestBatchSize = 20;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 200;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

#define TEST

struct NetImpl : torch::nn::Module {
	NetImpl()
		: 
		fc1(1, 1000),
		fc2(1000, 1000),
		fc3(1000, 100),
		fc4(100, 50), 
		fc5(50, 3)
	{
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("fc3", fc3);
		register_module("fc4", fc4);
		register_module("fc5", fc5);
	}

	torch::Tensor forward(torch::Tensor x) {
		//cpp_torch::dump_dim(std::string("input"), x);
		x = fc1->forward(x);
		//cpp_torch::dump_dim(std::string("fc"), x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim(std::string("tanh"), x);
		x = fc2->forward(x);
		//cpp_torch::dump_dim(std::string("fc2"), x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim(std::string("tanh"), x);
		x = fc3->forward(x);
		//cpp_torch::dump_dim(std::string("fc3"), x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim(std::string("tanh"), x);
		x = fc4->forward(x);
		//cpp_torch::dump_dim(std::string("fc4"), x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim(std::string("tanh"), x);
		x = fc5->forward(x);
		//cpp_torch::dump_dim(std::string("fc5"), x);
		return x;
	}
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
	torch::nn::Linear fc4;
	torch::nn::Linear fc5;
};
TORCH_MODULE(Net); // creates module holder for NetImpl


// load csv dataset
std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;
std::vector<tiny_dnn::vec_t> dataset;
tiny_dnn::vec_t dataset_min, dataset_maxmin;

void read_dataset(const std::string &data_dir_path)
{
	CSVReader csv(data_dir_path + "/sample.csv", ',', false);

	dataset = csv.toTensor();

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_int_distribution<> rand_index(0, (int)dataset.size() - 1);

	std::random_device rnd2;
	std::mt19937 mt2(rnd2());
	std::uniform_real_distribution<> rand(0, 1.0);

	for ( int i = 0; i < dataset.size(); i++)
	{
		const int index = rand_index(mt);
		if (rand(mt2) < 0.8)
		{
			tiny_dnn::vec_t label;
			tiny_dnn::vec_t image;

			image.push_back(dataset[index][0]);
			train_images.push_back(image);

			for (int k = 1; k < dataset[index].size(); k++)
			{
				label.push_back(dataset[index][k]);
			}
			train_labels.push_back(label);
		}
		else
		{
			tiny_dnn::vec_t label;
			tiny_dnn::vec_t image;

			image.push_back(dataset[index][0]);
			test_images.push_back(image);

			for (int k = 1; k < dataset[index].size(); k++)
			{
				label.push_back(dataset[index][k]);
			}
			test_labels.push_back(label);
		}
	}
	tiny_dnn::cpp_torch::normalizeMinMax(train_labels, dataset_min, dataset_maxmin);
	tiny_dnn::cpp_torch::normalizeMinMax(test_labels, dataset_min, dataset_maxmin);
}

void learning_and_test_dataset(torch::Device device)
{

#ifndef TEST
	Net model;
#endif

#ifdef TEST
	cpp_torch::Net model;

	model.get()->setInput(1, 1, 1);
	model.get()->add_fc(1000);
	model.get()->add_Tanh();
	model.get()->add_fc(1000);
	model.get()->add_Tanh();
	model.get()->add_fc(100);
	model.get()->add_Tanh();
	model.get()->add_fc(50);
	model.get()->add_Tanh();
	model.get()->add_fc(3);
#endif


#ifndef TEST
	cpp_torch::network_torch<Net> nn(model, device);
#else
	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);
#endif

	nn.input_dim(1,1,1);
	nn.output_dim(1,1,3);
	nn.classification = false;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	torch::optim::SGD optimizer(
		model.get()->parameters(), torch::optim::SGDOptions(0.1).momentum(0.5));
	//auto optimizer =
	//	torch::optim::Adam(model.get()->parameters(),
	//		torch::optim::AdamOptions(0.01));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		++epoch;
		float loss = nn.get_loss(train_images, train_labels);
		std::cout << "loss :" << loss << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
	};

	nn.set_tolerance(0.1, 0.001, 5);

	// fit
	nn.fit(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	float_t loss = nn.get_loss(train_images, train_labels);
	printf("loss:%f\n", loss);

	std::vector<int>& res = nn.test_tolerance(test_images, test_labels);
	cpp_torch::print_ConfusionMatrix(res, nn.get_tolerance());


	FILE* fp = fopen("predict.dat", "w");
	for (int i = 0; i < dataset.size(); i++)
	{
		tiny_dnn::vec_t x;
		x.push_back(dataset[i][0]);
		tiny_dnn::vec_t y = nn.predict(x);

		fprintf(fp, "%f", x[0]);
		for (int k = 0; k < 3; k++)
		{
			fprintf(fp, " %f %f", y[k] * dataset_maxmin[k] + dataset_min[k], dataset[i][k+1]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
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


	read_dataset(std::string(kDataRoot));

	learning_and_test_dataset(device);
}
