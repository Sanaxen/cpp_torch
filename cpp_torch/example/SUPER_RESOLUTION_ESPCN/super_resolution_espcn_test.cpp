/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "test/include/images_mormalize.h"

//#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 60;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int kDataAugment_crop_num_factor = 10;
const int upscale_factor = 3;

int calculate_valid_crop_size(const int crop_size, const int upscale_factor)
{
	return crop_size - (crop_size % upscale_factor);
}


int input_image_size;
void test_super_resolution_dataset(torch::Device device, const std::string& test_image)
{
	printf("input[%s]\n", test_image.c_str());
	cv::Mat org_image = cv::imread(test_image.c_str());

	int input_image_size = std::max(org_image.size[0], org_image.size[1]);
	printf("org_size:%d\n", input_image_size);

	cpp_torch::Net model;

	model.get()->setInput(1, input_image_size, input_image_size);
	model.get()->add_conv2d(1, 64, 5, 1, 2);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 64, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 32, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(32, pow(upscale_factor, 2), 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_pixel_shuffle(upscale_factor);

	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);

	nn.input_dim(1, input_image_size, input_image_size);
	nn.output_dim(1, input_image_size * upscale_factor, input_image_size * upscale_factor);

	nn.load(std::string("model.pt"));


	nn.model.get()->train(false);

	char imgfname[256];

	cv::resize(org_image, org_image, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
	sprintf(imgfname, "super_resolution_ref_%03d.png", 999);
	cv::imwrite(imgfname, org_image);

	cv::Mat input = org_image.clone();
	//cv::resize(org_image, input, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);
	sprintf(imgfname, "super_resolution_befor_%03d.png", 999);
	cv::imwrite(imgfname, input);

	cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(input);
	ImageRGB2YCbCr(&imgx);
	tiny_dnn::vec_t& vx = image_channel2vec_t(&imgx, 1, input_image_size, input_image_size);
	torch::Tensor x = cpp_torch::toTorchTensors(vx).view({ 1,1,input_image_size,input_image_size });

	torch::Tensor generated_img = nn.model.get()->forward(x.to(device));

	cv::Mat resizeImage;
	cv::resize(input, resizeImage, cv::Size(input_image_size* upscale_factor, input_image_size* upscale_factor), 0, 0, INTER_CUBIC);
	sprintf(imgfname, "super_resolution_bicubic_%03d.png", 999);
	cv::imwrite(imgfname, resizeImage);

	cpp_torch::Image& image_CbCr = cpp_torch::cvutil::cvMatToImage(resizeImage);
	cpp_torch::ImageRGB2YCbCr(&image_CbCr);
	cpp_torch::Image ycbcr[3];
	ycbcr[0] = image_CbCr.clone();
	ycbcr[1] = image_CbCr.clone();
	ycbcr[2] = image_CbCr.clone();
	cpp_torch::ImageGetChannel(&ycbcr[0], 1);
	cpp_torch::ImageGetChannel(&ycbcr[1], 2);
	cpp_torch::ImageGetChannel(&ycbcr[2], 3);
	tiny_dnn::vec_t& xx = cpp_torch::toTensor_t(generated_img[0], (input_image_size* upscale_factor)*(input_image_size* upscale_factor));

	cpp_torch::Image& predict_Y = cpp_torch::ToImage(&xx[0], input_image_size* upscale_factor, input_image_size* upscale_factor, 1);
	cpp_torch::ImageGetChannel(&predict_Y, 1);
	cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[1], 2);
	cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[2], 3);
	cpp_torch::ImageYCbCr2RGB(&predict_Y);
	cv::Mat output = cpp_torch::cvutil::ImgeTocvMat(&predict_Y);
	sprintf(imgfname, "super_resolution_test_%03d.png", 999);
	cv::imwrite(imgfname, output);

	float loss = cv::norm((resizeImage - output)) / (input_image_size* upscale_factor*input_image_size* upscale_factor);
	float psnr = 10.0*log10(1.0 / loss);
	printf("psnr:%.4f dB  %.4f\n", psnr , loss);
}


int main(int argc, char** argv)
{

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

	test_super_resolution_dataset(device, argv[1]);

	return 0;
}
