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

#include "test/include/seqence_data.h"
#include "libtorch_link_libs.hpp"


//#define TEST1
#define TEST2

//#define USE_CUDA

#define ZERO_TOL 0.0001

const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 32;

// The batch size for testing.
const int64_t kTestBatchSize = 10;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 2000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int sequence_length = 20;
const int out_sequence_length = 3;
const int hidden_size = 64;
const int x_dim = 1;
const int y_dim = 3;

#define TEST

struct NetImpl : torch::nn::Module {
	NetImpl()
		: 
		gru({ nullptr }),
		fc1(sequence_length*hidden_size, 128),
		fc2(128, 128),
		fc3(128, 50),
		fc4(50, y_dim*out_sequence_length)
	{
		auto opt = torch::nn::GRUOptions(y_dim, hidden_size);
		opt = opt.batch_first(true);

		gru = torch::nn::GRU(opt);
		gru.get()->options.batch_first(true);
		fc1.get()->options.with_bias(false);
		fc2.get()->options.with_bias(false);
		fc3.get()->options.with_bias(false);
		fc4.get()->options.with_bias(false);

		register_module("gru", gru);
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
		x = gru->forward(x).output;
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

	torch::nn::GRU gru;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
	torch::nn::Linear fc4;
};
TORCH_MODULE(Net); // creates module holder for NetImpl

std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;


void read_rnn_dataset(cpp_torch::test::SeqenceData& seqence_data, const std::string &data_dir_path)
{
	seqence_data.Initialize(data_dir_path);

	seqence_data.get_train_data(train_images, train_labels);
	seqence_data.get_test_data(test_images, test_labels);
}

void learning_and_test_rnn_dataset(cpp_torch::test::SeqenceData& seqence_data, torch::Device device)
{

#ifndef TEST
	Net model;
#endif

#ifdef TEST
	cpp_torch::Net model;

	model.get()->device = device;
	model.get()->setInput(1, 1, y_dim*sequence_length);

	model.get()->add_recurrent(std::string("gru"), sequence_length, hidden_size);
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

	cpp_torch::progress_display2 disp(train_images.size());
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

		float loss = 0;
		if (epoch % kLogInterval == 0)
		{
			loss = nn.get_loss(train_images, train_labels, kTestBatchSize);

			seqence_data.sequence_test(nn);
			if (loss < ZERO_TOL)
			{
				nn.stop_ongoing_training();
			}
		}
		++epoch;

		if (epoch <= kNumberOfEpochs)
		{
			std::string h = "[" + std::to_string(epoch) + "/" + std::to_string(kNumberOfEpochs) + "]" ;
			if ((epoch - 1) % kLogInterval == 0)
			{
				h += " loss:" + std::to_string(loss);
			}
			disp.restart(train_images.size(), h);
		}
		t.restart();
	};

	int batch = 1;
	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
		batch++;
	};

	nn.set_tolerance(0.01, 0.0001, 5);

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

	cpp_torch::test::SeqenceData seqence_data;
	seqence_data.x_dim = x_dim;
	seqence_data.y_dim = y_dim;
	seqence_data.sequence_length = sequence_length;
	seqence_data.out_sequence_length = out_sequence_length;
	seqence_data.n_minibatch = kTrainBatchSize;

	read_rnn_dataset(seqence_data, std::string(kDataRoot));

	learning_and_test_rnn_dataset(seqence_data, device);
}
