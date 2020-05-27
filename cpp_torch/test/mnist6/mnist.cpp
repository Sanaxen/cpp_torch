/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"


#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

#define TEST

struct NetImpl : torch::nn::Module {
	NetImpl()
		: conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
		conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
		fc1(320, 50),
		fc2(50, 10) {
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv2_drop", conv2_drop);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	torch::Tensor forward(torch::Tensor x) {

#if 0
		cpp_torch::dump_dim(std::string(""), x);
		x = conv1->forward(x);
		cpp_torch::dump_dim(std::string("conv"), x);
		x = torch::max_pool2d(x, 2);
		cpp_torch::dump_dim(std::string("maxpool"), x);
		x = torch::relu(x);
		cpp_torch::dump_dim(std::string("relu"), x);
		x = conv2->forward(x);
		cpp_torch::dump_dim(std::string("conv"), x);
		x = conv2_drop->forward(x);
		cpp_torch::dump_dim(std::string("comv_drop"), x);
		x = torch::max_pool2d(x, 2);
		cpp_torch::dump_dim(std::string("maxpool"), x);
		x = torch::relu(x);
		cpp_torch::dump_dim(std::string("relu"), x);

		//cpp_torch::dump_dim(std::string("xx"),x);
		x = x.view({ -1, 320 });
		cpp_torch::dump_dim(std::string("view"), x);
		x = fc1->forward(x);
		cpp_torch::dump_dim(std::string("fc"), x);
		x = torch::relu(x);
		cpp_torch::dump_dim(std::string(""), x);
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		cpp_torch::dump_dim(std::string("drop"), x);
		x = fc2->forward(x);
		cpp_torch::dump_dim(std::string("fc"), x);
		//return x;
		return torch::log_softmax(x, /*dim=*/1);
#else
		//cpp_torch::dump_dim(std::string("xx"), conv1->forward(x));
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		x = torch::relu(
			torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));

		//cpp_torch::dump_dim(std::string("xx"),x);
		x = x.view({ -1, 320 });
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		x = fc2->forward(x);
		//return x;
		return torch::log_softmax(x, /*dim=*/1);
#endif
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};
TORCH_MODULE(Net); // creates module holder for NetImpl


// load MNIST dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void read_mnist_dataset(const std::string &data_dir_path)
{
	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels-idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images-idx3-ubyte",
		&train_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels-idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images-idx3-ubyte",
		&test_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);
}

void learning_and_test_mnist_dataset(torch::Device device)
{

#ifndef TEST
	Net model;
#endif

#ifdef TEST
	cpp_torch::Net model;

	model.get()->setInput(1, 28, 28);
	model.get()->add_conv2d(1, 10, 5);
	model.get()->add_maxpool2d(2);
	model.get()->add_ReLU();

	model.get()->add_conv2d(10, 20, 5);
	model.get()->add_conv_drop(0.5);
	model.get()->add_maxpool2d(2);
	model.get()->add_ReLU();
	model.get()->add_fc(50);
	model.get()->add_ReLU();
	model.get()->add_dropout(0.5);
	model.get()->add_fc(10);
	model.get()->add_LogSoftmax(1);
#endif


#ifndef TEST
	cpp_torch::network_torch<Net> nn(model, device);
#else
	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);
#endif

	nn.input_dim(1, 28, 28);
	nn.output_dim(1, 1, 10);
	nn.classification = true;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;


	torch::optim::SGD optimizer(
		model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
	//auto optimizer =
	//	torch::optim::Adam(model.get()->parameters(),
	//		torch::optim::AdamOptions(0.01));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (epoch % kLogInterval == 0)
		{
			tiny_dnn::result res = nn.test(test_images, test_labels);
			std::cout << res.num_success << "/" << res.num_total << std::endl;
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

	//progress.start();
	// train
	nn.train(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	float_t loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
	printf("loss:%f\n", loss);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	{
		cpp_torch::textColor color("YELLOW");
		cpp_torch::print_ConfusionMatrix(res);
	}

	nn.test(test_images, test_labels, kTestBatchSize);

	//nn.save(std::string("model1.pt"));

	//Net model2;
	//cpp_torch::network_torch<Net> nn2(model2, device);
	//nn2 = nn;

	//nn2.load(std::string("model1.pt"));
	//nn2.test(test_images, test_labels, kTrainBatchSize);

	//tiny_dnn::result res2 = nn2.test(test_images, test_labels);
	//cpp_torch::print_ConfusionMatrix(res2);

}


auto main() -> int {

#if 0
	cpp_torch::url_download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "./data/train-images-idx3-ubyte.gz");
	cpp_torch::url_download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "./data/train-labels-idx1-ubyte.gz");
	cpp_torch::url_download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "./data/t10k-images-idx3-ubyte.gz");
	cpp_torch::url_download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz");
	
	cpp_torch::file_uncompress("./data/train-images-idx3-ubyte.gz", true);
	cpp_torch::file_uncompress("./data/train-labels-idx1-ubyte.gz", true);
	cpp_torch::file_uncompress("./data/t10k-images-idx3-ubyte.gz", true);
	cpp_torch::file_uncompress("./data/t10k-labels-idx1-ubyte.gz", true);
#else
	std::string url = "http://yann.lecun.com/exdb/mnist/";
	std::vector<std::string> files = {
			"train-images-idx3-ubyte.gz",
			"train-labels-idx1-ubyte.gz",
			"t10k-images-idx3-ubyte.gz",
			"t10k-labels-idx1-ubyte.gz"
	};
	std::string dir = std::string(kDataRoot) + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir );
#endif

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

	read_mnist_dataset(std::string(kDataRoot));

	learning_and_test_mnist_dataset(device);
}
