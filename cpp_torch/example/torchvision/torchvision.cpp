/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_TORCHVISION_0100
#define USE_OPENCV_UTIL
#define USE_TORCHVISION
#include "cpp_torch.h"
#include <torch/script.h> // One-stop header.

#define USE_CUDA

/* 2020/12/19 libtorch1.7.1
loss= 0.0378258
1277/1278
accuracy:99.9218% (1277/1278)
	*     0     1     2
	0   385     1     0
	1     0   155     0
	2     0     0   737
******************************/

/* 2020/12/19 libtorch1.9.0
loss= 0.0957799
1260/1278
accuracy:98.5916% (1260/1278)
	*     0     1     2
	0   324    10     4
	1     2   141     1
	2     1     0   795
******************************/

// Where to find the dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 16;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 500;

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
			//cv::imshow("", cvmat);
			//cv::waitKey(1);

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
	std::vector<std::string>& pineapple = cpp_torch::getImageFiles(kDataRoot + std::string("/pineapple"));
	std::vector<std::string>& banana = cpp_torch::getImageFiles(kDataRoot + std::string("/banana"));
	std::vector<std::string>& orange = cpp_torch::getImageFiles(kDataRoot + std::string("/orange"));

	load_image_(pineapple, 0);
	load_image_(banana, 1);
	load_image_(orange, 2);
}

auto main() -> int {

	std::string url = "https://download.pytorch.org/models/";
	std::vector<std::string> files = {
			"resnet18-5c106cde.pth"
	};
	std::string dir = std::string(kDataRoot) + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir );
	
	// python convert.py
	// data/script_module.pt -> data/script_module.pt

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
	{
		FILE* fp = fopen("data/script_module.pt", "r");
		if (fp == NULL)
		{
			std::cout << "python convert.py" << std::endl;
			std:cout << "data/script_module.pt -> data/script_module.pt" << std::endl;
			exit(1);
		}
		else
		{
			fclose(fp);
		}
	}
	torch::load(model, "data/script_module.pt");


	int num_class = 1000;

	//Transfer Learning
	bool TransferLearning = true;
	std::vector<torch::Tensor> trainable_params;
	if (TransferLearning)
	{
		num_class = 3;
		model->unregister_module("fc");

		model->fc = torch::nn::Linear(torch::nn::LinearImpl(512, num_class));
		model->register_module("fc", model->fc);

		auto params = model->named_parameters(true /*recurse*/);
		for (auto& param : params)
		{
			auto layer_name = param.key();
			if ("fc.weight" == layer_name || "fc.bias" == layer_name)
			{
				param.value().set_requires_grad(true);
				trainable_params.push_back(param.value());
			}
			else
			{
				param.value().set_requires_grad(false);
			}
		}
	}
	else
	{
		num_class = 3;
		model->unregister_module("fc");

		model->fc = torch::nn::Linear(torch::nn::LinearImpl(512, num_class));
		model->register_module("fc", model->fc);
		trainable_params = model.get()->parameters();
	}

	load_images();
	
	cpp_torch::network_torch<vision::models::ResNet18> nn(model, device);

	//nn.classification = true;
	nn.input_dim(3, 224, 224);
	nn.output_dim(1, 1, num_class);

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	auto optimizer =
		torch::optim::SGD(trainable_params,
			torch::optim::SGDOptions(0.001).weight_decay(1.0e-6).momentum(0.9).nesterov(true));

	// create callback
	int epoch = 1;
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;


		if (epoch % kLogInterval == 0)
		{
			float loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
			std::cout << "loss= " << loss << std::endl;
		}
		++epoch;
		if (epoch <= kNumberOfEpochs)
		{
			disp.restart(train_images.size());
		}
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
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

	std::cout << "train_images " << train_images.size();
	index.resize(1000);
	std::cout << "-> " << train_images.size() << std::endl;

#pragma omp parallel for
	for (int i = 0; i < train_images.size(); i++)
	{
		train_images[i] = images[index[i]];
		train_labels[i] = labels[index[i]];
	}

	printf("train start\n"); fflush(stdout);
	nn.train(&optimizer, train_images, train_labels, kTrainBatchSize, kNumberOfEpochs, on_enumerate_minibatch, on_enumerate_epoch);
	

	tiny_dnn::result res = nn.test(train_images, train_labels);
	std::cout << res.num_success << "/" << res.num_total << std::endl;
	res.print_detail(std::cout);

	return 0;
}
