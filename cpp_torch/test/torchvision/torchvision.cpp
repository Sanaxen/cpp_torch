/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#define USE_TORCHVISION
#include "cpp_torch.h"
#include <torch/script.h> // One-stop header.

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

//https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void load_image_(std::vector<std::string>& files, int label)
{
	size_t input_image_size = 224;
	for (int i = 0; i < files.size(); i++)
	{
		try
		{
			cpp_torch::Image& img = cpp_torch::readImage(files[i].c_str());
			cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);

			cv::resize(cvmat, cvmat, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
			//std::cout << "size " << cvmat.size() << std::endl;

			cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(cvmat);
			tiny_dnn::vec_t& vx = image2vec_t(&imgx, 3, input_image_size, input_image_size, 1.0/255.0);
			train_images.push_back(vx);
			train_labels.push_back(label);
		}
		catch (...)
		{
		}
	}
}
void load_images()
{
	std::vector<std::string>& apple = cpp_torch::getImageFiles(kDataRoot + std::string("/apple"));
	std::vector<std::string>& banana = cpp_torch::getImageFiles(kDataRoot + std::string("/banana"));
	std::vector<std::string>& orange = cpp_torch::getImageFiles(kDataRoot + std::string("/orange"));
	
	load_image_(apple, 1);
	load_image_(banana, 2);
	load_image_(orange, 3);
}

auto main() -> int {

	std::string url = "https://download.pytorch.org/models/";
	std::vector<std::string> files = {
			"resnet18-5c106cde.pth"
	};
	std::string dir = std::string(kDataRoot) + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir );

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
	//model->eval();
	//torch::load(model, "data/script_module.pt");

	model->fc.get()->options.out_features(3);

	load_images();
	//model->fc.get()->reset_parameters();
	
	cpp_torch::network_torch<vision::models::ResNet18> nn(model, device);

	nn.input_dim(3, 224, 224);
	nn.output_dim(1, 1, 1000);

	auto optimizer =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(0.0025));
	
	// create callback
	tiny_dnn::timer t;
	int epoch = 1;
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;


		if (epoch % kLogInterval == 0)
		{
			tiny_dnn::result res = nn.test(train_images, train_labels);
			std::cout << res.num_success << "/" << res.num_total << std::endl;
		}
		++epoch;
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
	};

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_int_distribution<> rand_index(0, (int)train_images.size() - 1);

	std::vector<int> index;

	index.resize(train_images.size());
#pragma omp parallel for
	for (int i = 0; i < train_images.size(); i++)
	{
		index[i] = rand_index(mt);
	}
	auto images = train_images;
	auto labels = train_labels;
#pragma omp parallel for
	for (int i = 0; i < train_images.size(); i++)
	{
		train_images[index[i]] = images[i];
		train_labels[index[i]] = labels[i];
	}
	//train_images.resize(640);
	//train_labels.resize(640);
	nn.train(&optimizer, train_images, train_labels, 128, 100, on_enumerate_minibatch, on_enumerate_epoch);
	
	//for (int i = 0; i < train_images.size(); i++)
	//{
	//	auto p = nn.predict(train_images[i]);
	//	printf("%d\n", max_index(p));
	//}
	tiny_dnn::result res = nn.test(train_images, train_labels);
	std::cout << res.num_success << "/" << res.num_total << std::endl;
	res.print_detail(std::cout);

#if 0
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
#endif

}
