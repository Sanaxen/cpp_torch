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

const int64_t kNumberOfTrainImages = 4256;
//#define TEST


// load  dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void learning_and_test_dcgan_dataset(torch::Device device)
{
	train_images.clear();
	printf("load images start\n");
	cpp_torch::progress_display2 loding(kNumberOfTrainImages+1);
	for (int i = 0; i <= kNumberOfTrainImages; i++)
	{
		char buf[32];
		sprintf(buf, "%d", i);
		std::string& file = kDataRoot + std::string("/jpg/") + std::string(buf) + ".jpg";

		cpp_torch::Image* img = cpp_torch::readImage(file.c_str());
		tiny_dnn::vec_t& v = image2vec_t(img, 3, img->height, img->width/*, 1.0/255.0*/);
		delete img;

		train_images.push_back(v);
		loding += 1;
	}

	//image normalize (mean 0 and variance 1)
	float mean = 0.0;
	float stddiv = 0.0;
	cpp_torch::test::images_normalize(train_images, mean, stddiv);
	printf("mean:%f stddiv:%f\n", mean, stddiv);

	loding.end();
	printf("load images:%d\n", train_images.size());

	const int nz = 100;

	cpp_torch::Net  g_model;
	g_model.get()->setInput(nz, 1, 1);
	g_model.get()->add_conv_transpose2d(100, 256, 4, 1, 0);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(256, 128, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(128, 64, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(64, 32, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(32, 3, 4, 2, 1);
	g_model.get()->add_Tanh();

	cpp_torch::Net  d_model;
	d_model.get()->setInput(3, 64, 64);
	d_model.get()->add_conv2d(3, 32, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_conv2d(32, 64, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_conv2d(64, 128, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_conv2d(128, 256, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_conv2d(256, 1, 4, 1, 0);
	d_model.get()->add_Squeeze();
#ifdef USE_LOSS_BCE
	d_model.get()->add_Sigmoid();
#endif

	cpp_torch::network_torch<cpp_torch::Net> g_nn(g_model, device);
	cpp_torch::network_torch<cpp_torch::Net> d_nn(d_model, device);

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


	cpp_torch::DCGAN<cpp_torch::Net, cpp_torch::Net> dcgan(g_nn, d_nn, device);


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
					cv::Mat& cv_mat = cpp_torch::cvutil::tensorToMat(generated_img[i], 255.0);
					cv::imwrite(fname, cv_mat);
				}
				cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, "image_array.bmp");
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
		}
		else
		{
#ifdef USE_OPENCV_UTIL
			//cv::Mat& img = cpp_torch::cvutil::TensorToImageFile(generated_img[0], "gen0.bmp", 255.0);
			//cv::imshow("gen0.bmp", img);
			//cv::waitKey(500);

			char fname[64];
			sprintf(fname, "generated_images/image_array%d.png", epoch);
			cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, fname);
			cv::imshow("", img);
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

	dcgan.train(&g_optimizer, &d_optimizer, train_images, kTrainBatchSize, kNumberOfEpochs, nz, on_enumerate_minibatch, on_enumerate_epoch);
	std::cout << "end training." << std::endl;
	fclose(fp);

	g_nn.save(std::string("g_model.pt"));
	d_nn.save(std::string("d_model.pt"));
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
