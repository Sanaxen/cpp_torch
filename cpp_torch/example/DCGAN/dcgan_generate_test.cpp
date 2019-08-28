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

// The batch size for testing.
int64_t kTestBatchSize = 64;

const int kRndArraySize = 100;

int  seed = -1;
void test_dcgan(torch::Device device)
{
	const int nz = kRndArraySize;

	cpp_torch::Net  g_model;
	g_model.get()->setInput(nz, 1, 1);
	g_model.get()->add_conv_transpose2d(nz, 256, 4, 1, 0);
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

	//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
	torch::Tensor check_z = torch::randn({ kTestBatchSize, nz, 1, 1 }).to(device);

	cpp_torch::network_torch<cpp_torch::Net> g_nn(g_model, device);
	g_nn.load(std::string("g_model.pt"));
	g_nn.model.get()->train(false);
	torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
	g_nn.model.get()->train(true);

	generated_img = ((1 + generated_img).mul(128)).clamp(0, 255);

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

	if (argc < 1)
	{
		printf("dcgan_generate_test.exe seed num\n");
	}
	if (argc > 1) seed = atoi(argv[1]);
	if (argc > 2) kTestBatchSize = atoi(argv[2]);

	printf("kTestBatchSize:%d\n", kTestBatchSize);

	torch::manual_seed( abs(seed));

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

	test_dcgan(device);
}
