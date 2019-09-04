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
#include "util/command_line.h"

#define USE_CUDA
#define IMAGE_SIZE	64
#define IMAGE_CHANNEL	3

// The batch size for testing.
int64_t kTestBatchSize = 64;

const int kRndArraySize = 100;
const int ngf = 64;

bool gpu = false;
int  seed = -1;
void test_dcgan(torch::Device device)
{
	const int nz = kRndArraySize;

	cpp_torch::Net  g_model;
	g_model.get()->setInput(nz, 1, 1);
	g_model.get()->add_conv_transpose2d(nz, ngf * 8, 4, 1, 0);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 8, ngf * 4, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 4, ngf * 2, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 2, ngf, 4, 2, 1);
	g_model.get()->add_bn();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf, IMAGE_CHANNEL, 4, 2, 1);
	g_model.get()->add_Tanh();

	//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
	torch::Tensor check_z = torch::randn({ kTestBatchSize, nz, 1, 1 }).to(device);

	cpp_torch::network_torch<cpp_torch::Net> g_nn(g_model, device);
	g_nn.load(std::string("g_model.pt"));
	g_nn.model.get()->to(device);

	g_nn.model.get()->train(false);
	torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
	g_nn.model.get()->train(true);

	const float min = 0.0;
	const float max = 255.0;

	generated_img = ((1 + generated_img).mul(0.5*(max - min)) + min).clamp(0, 255);

#pragma omp parallel for
	for (int i = 0; i < kTestBatchSize; i++)
	{
		char fname[32];
		sprintf(fname, "generated_images/gen%d.bmp", i);
		cv::Mat& cv_mat = cpp_torch::cvutil::tensorToMat(generated_img[i], 1);
		cv::imwrite(fname, cv_mat);
	}

	char fname[64];
	sprintf(fname, "generated_images/image_array%d.png", 99999);
	int r = (int)(sqrt((float)kTestBatchSize));

	cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, r, r, fname, 2);
	if (seed < 0)
	{
		cv::imshow("", img);
		cv::waitKey();
	}

}


auto main(int argc, char** argv) -> int {

	for (int i = 1; i < argc; i++)
	{
		BOOL_OPT(i, gpu, "--gpu");
		INT_OPT(i, seed, "--seed");
		INT_OPT(i, kTestBatchSize, "--batch");
	}
	printf("--gpu:%d\n", gpu);
	printf("--batch:%d\n", kTestBatchSize);

	torch::manual_seed( abs(seed));

	torch::DeviceType device_type;
#ifdef USE_CUDA
	if (gpu && torch::cuda::is_available()) {
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

	test_dcgan(device);
}
