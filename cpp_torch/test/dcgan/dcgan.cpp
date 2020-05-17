/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "dcgan.h"
#include "test/include/images_mormalize.h"

#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 300;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int kRndArraySize = 100;


struct GeneratorImpl : torch::nn::Module {
	GeneratorImpl(): 
		conv1(torch::nn::Conv2dOptions(kRndArraySize, 256,4).with_bias(false).transposed(true).stride(1).padding(0)),
		batchnml1(torch::nn::BatchNormOptions(256)),
		conv2(torch::nn::Conv2dOptions(256, 128, 4).bias(false).transposed(true).stride(2).padding(1)),
		batchnml2(torch::nn::BatchNormOptions(128)),
		conv3(torch::nn::Conv2dOptions(128, 64, 4).bias(false).transposed(true).stride(2).padding(1)),
		batchnml3(torch::nn::BatchNormOptions(64)),
		conv4(torch::nn::Conv2dOptions(64, 32, 4).bias(false).transposed(true).stride(2).padding(1)),
		batchnml4(torch::nn::BatchNormOptions(32)),
		conv5(torch::nn::Conv2dOptions(32, 3, 4).bias(false).transposed(true).stride(2).padding(1))
		{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("conv5", conv5);
		register_module("batchnml1", batchnml1);
		register_module("batchnml2", batchnml2);
		register_module("batchnml3", batchnml3);
		register_module("batchnml4", batchnml4);
	}

	torch::Tensor forward(torch::Tensor x) {

		x = conv1->forward(x);
		x = torch::relu_(batchnml1->forward(x));

		x = conv2->forward(x);
		x = torch::relu_(batchnml2->forward(x));
		
		x = conv3->forward(x);
		x = torch::relu_(batchnml3->forward(x));

		x = conv4->forward(x);
		x = torch::relu_(batchnml4->forward(x));

		x = conv5->forward(x);
		x = torch::tanh(x);
		//cpp_torch::dump_dim("x", x);
		return x;
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Conv2d conv4;
	torch::nn::Conv2d conv5;
	torch::nn::BatchNorm2d batchnml1;
	torch::nn::BatchNorm2d batchnml2;
	torch::nn::BatchNorm2d batchnml3;
	torch::nn::BatchNorm2d batchnml4;
};
TORCH_MODULE(Generator); // creates module holder for NetImpl

struct DiscriminatorImpl : torch::nn::Module {
	DiscriminatorImpl() :
		conv1(torch::nn::Conv2dOptions(3, 32, 4).bias(false).stride(2).padding(1)),
		conv2(torch::nn::Conv2dOptions(32, 64, 4).bias(false).stride(2).padding(1)),
		batchnml1(torch::nn::BatchNormOptions(64)),
		conv3(torch::nn::Conv2dOptions(64, 128, 4).bias(false).stride(2).padding(1)),
		batchnml2(torch::nn::BatchNormOptions(128)),
		conv4(torch::nn::Conv2dOptions(128, 256, 4).bias(false).stride(2).padding(1)),
		batchnml3(torch::nn::BatchNormOptions(256)),
		conv5(torch::nn::Conv2dOptions(256, 1, 4).bias(false).stride(1).padding(0))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("conv5", conv5);
		register_module("batchnml1", batchnml1);
		register_module("batchnml2", batchnml2);
		register_module("batchnml3", batchnml3);
	}

	torch::Tensor forward(torch::Tensor x) {

		x = conv1->forward(x);
		x = torch::leaky_relu_(x, 0.2);

		x = conv2->forward(x);
		x = batchnml1->forward(x);
		x = torch::leaky_relu_(x, 0.2);

		x = conv3->forward(x);
		x = batchnml2->forward(x);
		x = torch::leaky_relu_(x, 0.2);
		x = torch::dropout(x, 0.2, is_training());

		x = conv4->forward(x);
		x = batchnml3->forward(x);
		x = torch::leaky_relu_(x, 0.2);
		x = torch::dropout(x, 0.2, is_training());
		//cpp_torch::dump_dim("x", x);

		x = conv5->forward(x);
		x = torch::dropout(x, 0.2, is_training());
		//cpp_torch::dump_dim("x", x);
		//x = fc1->forward(x.view({ -1, 256 * 4 * 4 }));
		//cpp_torch::dump_dim("x", x);

		x = x.squeeze();
		//cpp_torch::dump_dim("squeeze->x", x);
#ifdef USE_LOSS_BCE
		x = torch::sigmoid(x);
#endif
		return x;
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Conv2d conv4;
	torch::nn::Conv2d conv5;
	torch::nn::BatchNorm2d batchnml1;
	torch::nn::BatchNorm2d batchnml2;
	torch::nn::BatchNorm2d batchnml3;
};
TORCH_MODULE(Discriminator); // creates module holder for NetImpl

// load  dataset
std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void learning_and_test_dcgan_dataset(torch::Device device)
{
	train_images.clear();
	printf("load images start\n");
	std::vector<std::string>& image_files = cpp_torch::getImageFiles(kDataRoot + std::string("/image"));

	cpp_torch::progress_display2 loding(image_files.size() + 1);

	tiny_dnn::vec_t& real_label = std::vector<float_t>(1, 1.0);
	for (int i = 0; i < image_files.size(); i++)
	{
		cpp_torch::Image img = cpp_torch::readImage(image_files[i].c_str());
		tiny_dnn::vec_t& v = image2vec_t(&img, 3, img.height, img.width/*, 1.0/255.0*/);
		train_images.push_back(v);
		train_labels.push_back(real_label);
		loding += 1;
	}
	loding.end();
	printf("load images:%d\n", train_images.size());

#if 0
	//image normalize (mean 0 and variance 1)
	float mean = 0.0;
	float stddiv = 0.0;
	cpp_torch::test::images_normalize(train_images, mean, stddiv);
	printf("mean:%f stddiv:%f\n", mean, stddiv);
#else
	//image normalize [-1, 1]
	cpp_torch::test::images_normalize_11(train_images);
#endif

	Generator g_model;
	Discriminator d_model;

	cpp_torch::network_torch<Generator> g_nn(g_model, device);
	cpp_torch::network_torch<Discriminator> d_nn(d_model, device);

	const int nz = kRndArraySize;

	//Generator  100 -> 3 x 64 x 64
	g_nn.input_dim(nz, 1, 1);
	g_nn.output_dim(3, 64, 64);

	//Discriminator  3 x 64 x 64 -> 0(fake) or 1(real)
	d_nn.input_dim(3, 64, 64);
	d_nn.output_dim(1, 1, 1);
	d_nn.classification = false;
	d_nn.batch_shuffle = true;

	//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
	torch::Tensor check_z = torch::randn({ kTrainBatchSize, nz, 1, 1 }).to(device);

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	auto g_optimizer =
		torch::optim::Adam(g_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));
	auto d_optimizer =
		torch::optim::Adam(d_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));


	cpp_torch::DCGAN<Generator, Discriminator> dcgan(g_nn, d_nn, device);


	FILE* fp = fopen("loss.dat", "w");
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;

		fprintf(fp, "%f %f\n", dcgan.get_generator_loss(), dcgan.get_discriminator_loss());
		fflush(fp);

		g_nn.model.get()->train(false);
		torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
		g_nn.model.get()->train(true);

		//generated_img = (mean + generated_img.mul(stddiv)).clamp(0, 255);
		//generated_img = ((1+generated_img).mul(128)).clamp(0, 255);
		float min = 0.0;
		float max = 255.0;
		generated_img = ((1 + generated_img).mul(0.5*(max - min)) + min).clamp(0, 255);

		if (epoch % kLogInterval == 0)
		{
			if (epoch == kNumberOfEpochs)
			{
#ifdef USE_OPENCV_UTIL
#pragma omp parallel for
				for (int i = 0; i < kTrainBatchSize; i++)
				{
					char fname[32];
					sprintf(fname, "generated_images/gen%d.bmp", i);
					cv::Mat& cv_mat = cpp_torch::cvutil::tensorToMat(generated_img[i], 1);
					cv::imwrite(fname, cv_mat);
				}
				cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, "image_array.bmp", 2);
				cv::imshow("image_array.bmp", img);
				cv::waitKey();
#else
#pragma omp parallel for
				for (int i = 0; i < kTrainBatchSize; i++)
				{
					char fname[32];
					sprintf(fname, "generated_images/gen%d.bmp", i);
					cpp_torch::TensorToImageFile(generated_img[i], fname, 255.0);
				}
				char cmd[32];
				sprintf(cmd, "cmd.exe /c make_image_array.bat %d", epoch);
				system(cmd);
#endif
			}
			char model_name[256];
			sprintf(model_name, "generate_model%03d.pt", epoch - 1);
			g_nn.save(std::string(model_name));

			sprintf(model_name, "discriminator_model%03d.pt", epoch - 1);
			d_nn.save(std::string(model_name));
		}
		else
		{
#ifdef USE_OPENCV_UTIL
			//cv::Mat& img = cpp_torch::cvutil::TensorToImageFile(generated_img[0], "gen0.bmp", 255.0);
			//cv::imshow("gen0.bmp", img);
			//cv::waitKey(500);

			cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, "image_array.bmp", 2);
			cv::imshow("image_array.bmp", img);
			cv::waitKey(500);

#else
			cpp_torch::TensorToImageFile(generated_img[0], "gen0.bmp", 255.0);
#endif
		}

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

	dcgan.train(&g_optimizer, &d_optimizer, train_images, std::vector<tiny_dnn::vec_t>(), kTrainBatchSize, kNumberOfEpochs, nz, on_enumerate_minibatch, on_enumerate_epoch);
	std::cout << "end training." << std::endl;
	fclose(fp);

	g_nn.save(std::string("generate_model.pt"));
	d_nn.save(std::string("discriminator_model.pt"));
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

	learning_and_test_dcgan_dataset(device);
}
