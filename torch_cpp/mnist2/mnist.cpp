#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../torch_util/libtorch_utils.h"
#include "../torch_util/libtorch.hpp"

//#define TEST1
#define TEST2

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
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
	return x;
	//return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
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

	Net model;

	torch::optim::SGD optimizer(
		model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
	//auto optimizer = torch::optim::Adam(model.get()->parameters(), torch::optim::AdamOptions(1));

	cpp_torch::network_torch<Net> nn(model, device);
	nn.input_dim(1, 28, 28);
	nn.out_dim(1, 1, 10);
	//nn.classification = true;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		epoch++;

		nn.test(test_images, test_labels, kTrainBatchSize);

		if ( epoch <= kNumberOfEpochs) disp.restart(train_images.size());
		t.restart();
	};

	int batch = 1;
	auto on_enumerate_minibatch = [&]() {
		batch++;
		disp += kTrainBatchSize;
	};

	nn.batch_shuffle = true;
	nn.train(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch, on_enumerate_epoch);
	std::cout << "\ntrain " << " finished. " << nn.time_measurement.elapsed() << "s elapsed." << std::endl;

	nn.test(test_images, test_labels, kTrainBatchSize);

	nn.save(std::string("model1.pt"));

	Net model2;
	cpp_torch::network_torch<Net> nn2(model2, device);
	nn2.input_dim(1, 28, 28);
	nn2.out_dim(1, 10, 1);
	nn2.classification = nn.classification;

	nn2.load(std::string("model1.pt"));
	nn2.test(test_images, test_labels, kTrainBatchSize);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	cpp_torch::print_ConfusionMatrix(res);

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


  read_mnist_dataset(std::string(kDataRoot));

  learning_and_test_mnist_dataset(device);
}
