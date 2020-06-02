/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#define USE_TORCHVISION
#include "cpp_torch.h"
#include <torch/script.h> // One-stop header.

//#define USE_CUDA

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

//https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


auto main() -> int {

	std::string url = "https://download.pytorch.org/models/";
	std::vector<std::string> files = {
			"resnet18-5c106cde.pth"
	};
	std::string dir = std::string(kDataRoot) + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir );
	// convert.py [resnet18-5c106cde.pth -> script_module.pt]

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


	auto model = vision::models::ResNet18();
	model->eval();
	torch::load(model, "data/script_module.pt");


	//224 x 224
	// Create a random input tensor and run it through the model.
	auto in = torch::rand({ 1, 3, 224, 224 });
	auto out = model->forward(in.to(device));
	//std::cout << out;
	std::cout << out.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

	cpp_torch::network_torch<vision::models::ResNet18> nn(model, device);
	nn.model.get()->forward(in);
	auto out2 = model->forward(in.to(device));
	//std::cout << out;
	std::cout << out2.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

	nn.model.get()->pretty_print(std::cout);
	std::cout << out;

#if 0
	auto module = torch::jit::load("./data/script_module.pt");
	std::cout << "load ok\n";
	module.to(device);
	assert(module != nullptr);
	std::cout << "ok\n";

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(in.to(device));

	out = module.forward(inputs).toTensor().to(torch::kCPU);

	std::cout << out;
#endif
}
