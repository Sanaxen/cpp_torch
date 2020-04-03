/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "test/include/images_normalize.h"
#include "util/command_line.h"

#define USE_CUDA
bool gpu = false;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data/image";

int upscale_factor = 3;

int calculate_valid_crop_size(const int crop_size, const int upscale_factor)
{
	return crop_size - (crop_size % upscale_factor);
}


int input_image_size;
void test_super_resolution_dataset(torch::Device device, const std::string& test_image, const int scale)
{
	upscale_factor = scale;

	printf("input[%s]\n", test_image.c_str());
	cv::Mat org_image = cv::imread(test_image.c_str());

	int input_image_size = std::max(org_image.size[0], org_image.size[1]);
	printf("org_size:%d\n", input_image_size);

	cpp_torch::Net model;

	model.get()->setInput(1, input_image_size, input_image_size);
#if 0
	model.get()->add_conv2d(1, 64, 5, 1, 2);
	model.get()->add_Tanh();
	model.get()->add_conv2d(64, 64, 3, 1, 1);
	model.get()->add_Tanh();
	model.get()->add_conv2d(64, 32, 3, 1, 1);
	model.get()->add_Tanh();
	model.get()->add_conv2d(32, pow(upscale_factor, 2), 3, 1, 1);
	model.get()->add_Tanh();
	model.get()->add_pixel_shuffle(upscale_factor);
	model.get()->add_Sigmoid();
#else
	model.get()->add_conv2d(1, 64, 5, 1, 2);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 64, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 32, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(32, pow(upscale_factor, 2), 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_pixel_shuffle(upscale_factor);
#endif

	char load_model[256];
	sprintf(load_model, "model_scale%d.pt", upscale_factor);
	torch::load(model, load_model);

	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);

	nn.input_dim(1, input_image_size, input_image_size);
	nn.output_dim(1, input_image_size * upscale_factor, input_image_size * upscale_factor);

	//nn.load(std::string(load_model));

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
	printf("psnr:%.4f dB  %.4f\n", psnr, loss);
	//getc(stdin);
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("super_resolution_test.exe image_file upscale\n");
	}

	float  upscale = 2;

	char* filename = "";
	for (int i = 1; i < argc; i++)
	{
		BOOL_OPT(i, gpu, "--gpu");
		CSTR_OPT(i, filename, "--input");
		FLOAT_OPT(i, upscale, "--upscale");
	}
	printf("--input:%s\n", filename);
	printf("--gpu:%d\n", gpu);
	printf("--upscale:%.2f\n", upscale);
	if (upscale <= 0)
	{
		printf("error [upscale <= 0] -> upscale = 2\n");
		upscale = 2;
	}
	if (upscale > 12)
	{
		printf("error [upscale > 12] -> upscale = 12\n");
		upscale = 12;
	}
	printf("%s -> upscale:%.2f\n", argv[1], upscale);

	if (upscale <= 1)
	{
		cv::Mat img = imread(filename);
		cv::resize(img, img, cv::Size(img.size().width * upscale, img.size().height * upscale), 0, 0, INTER_CUBIC);
		cv::imwrite("downsize_img.png", img);
		return 0;
	}

	torch::manual_seed(1);

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

	test_super_resolution_dataset(device, filename, (int)upscale);

	return 0;
}
