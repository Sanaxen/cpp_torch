/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"

/* CIFAR-10 dataset
 * The classes are completely mutually exclusive.
 * There is no overlap between automobiles and trucks.
 * "Automobile" includes sedans, SUVs, things of that sort.
 * "Truck" includes only big trucks. Neither includes pickup trucks.
 * https://www.cs.toronto.edu/~kriz/cifar.html
 * (https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)
 */

#define USE_CUDA

//Data Augment
const bool kDataAugment = true;

// Where to find the CIFAR10 dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 256;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 100;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


// load MNIST dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void read_cifar10_dataset(const std::string &data_dir_path)
{
	for (int i = 1; i <= 5; i++) {
		if (kDataAugment)
		{
			tiny_dnn::parse_cifar10(data_dir_path + "/data_batch_" + std::to_string(i) + ".bin",
				&train_images, &train_labels, 0.0, 1.0, 0, 0, 0.0, 1);
		}
		else
		{
			tiny_dnn::parse_cifar10(data_dir_path + "/data_batch_" + std::to_string(i) + ".bin",
				&train_images, &train_labels, 0.0, 1.0, 0, 0, 0.4913, 0.247);
		}
	}

	tiny_dnn::parse_cifar10(data_dir_path + "/test_batch.bin", &test_images, &test_labels,
		0.0, 1.0, 0, 0, 0.4913, 0.247);

	if (kDataAugment)
	{
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::uniform_int_distribution<> rand(0, 5);
		std::uniform_int_distribution<> rand_index(0, train_images.size() - 1);

		const size_t sz = train_images.size();
		for (int i = 0; i < sz * 2; i++)
		{
			const int index = rand_index(mt);
			tiny_dnn::vec_t& u = train_images[index];

			tiny_dnn::vec_t v(u.size());
			transform(u.begin(), u.end(), u.begin(),
				[=](float_t c) {return (c * CHANNEL_RANGE); }
			);
			std::string func = "";
			switch (rand(mt))
			{
			case 0:func = "GAMMA"; break;
			case 1:func = "RL"; break;
			case 2:func = "COLOR_NOIZE"; break;
			case 3:func = "NOIZE"; break;
			case 4:func = "ROTATION"; break;
			case 5:func = "SIFT"; break;
			}
			cpp_torch::ImageAugmentation(v, 32, 32, func);
			tiny_dnn::vec_t v2(v.size());
			transform(v.begin(), v.end(), v2.begin(),
				[=](float_t c) {return (c / CHANNEL_RANGE); }
			);
			train_images.push_back(v2);
			train_labels.push_back(train_labels[index]);

			//{
			//	Image* bmp = vec_t2image(v, 3, 32, 32);
			//	ImageWrite("aaa.bmp", bmp);
			//	delete bmp;
			//	exit(0);
			//}
		}
		const size_t sz2 = train_images.size();
#pragma omp parallel for
		for (int i = 0; i < sz2; i++)
		{
			for (int j = 0; j < train_images[i].size(); j++)
			{
				train_images[i][j] = (train_images[i][j] - 0.4913) / 0.247;
			}
		}
		printf("Augmentation:%d -> %d\n", sz, sz2);
	}
}

void learning_and_test_cifar10_dataset(torch::Device device)
{

	cpp_torch::Net model;

	model.get()->setInput(3, 32, 32);
#if 0
	model.get()->add_conv2d(3, 32, 5);
	model.get()->add_ReLU();
	model.get()->add_maxpool2d(2);

	model.get()->add_conv2d(32, 64, 5);
	model.get()->add_ReLU();
	model.get()->add_maxpool2d(2);

	model.get()->add_fc(256);
	model.get()->add_ReLU();
	model.get()->add_fc(128);
	model.get()->add_ReLU();
	model.get()->add_fc(128);
	model.get()->add_ReLU();
	model.get()->add_fc(64);
	model.get()->add_ReLU();
	model.get()->add_fc(10);
#else
	model.get()->add_conv2d(3, 32, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_conv2d(32, 32, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_maxpool2d(2);
	model.get()->add_conv_drop(0.2);

	model.get()->add_conv2d(32, 64, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_conv2d(64, 64, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_maxpool2d(2);
	model.get()->add_conv_drop(0.3);

	model.get()->add_conv2d(64, 128, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_conv2d(128, 128, 3, 3, 3);
	model.get()->add_ReLU();
	model.get()->add_bn();
	model.get()->add_maxpool2d(2);
	model.get()->add_conv_drop(0.4);

	model.get()->add_fc(10);
#endif
	model.get()->add_LogSoftmax(1);

	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);

	nn.input_dim(3, 32, 32);
	nn.output_dim(1, 1, 10);
	nn.classification = true;
	nn.batch_shuffle = true;
	nn.pre_make_batch = true;

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	auto optimizer =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(0.0005));
	//torch::optim::SGD optimizer(
	//	model.get()->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

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

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
	};

	// train
	nn.train(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	float_t loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
	printf("loss:%f\n", loss);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	cpp_torch::print_ConfusionMatrix(res);

	nn.test(test_images, test_labels, kTrainBatchSize);

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

	std::string url = "https://www.cs.toronto.edu/~kriz/";
	std::vector<std::string> files = {
			"cifar-10-binary.tar.gz"
	};
	std::string dir = std::string(kDataRoot) + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir);

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


	read_cifar10_dataset(std::string(kDataRoot));

	learning_and_test_cifar10_dataset(device);
}
