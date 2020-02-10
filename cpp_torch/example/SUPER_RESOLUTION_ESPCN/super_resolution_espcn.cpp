/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "test/include/images_mormalize.h"
#include "util/command_line.h"

#define USE_CUDA
bool gpu = true;

// dataset.
char* kDataRoot = "./data/BSDS300/images/train";
char* kTestDataRoot1 = "./data/BSDS300/images/test";
char* kTestDataRoot2 = "./data/Set5";

// The batch size for training.
int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 32;

// The number of epochs to train.
int64_t kNumberOfEpochs = 60;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

int64_t kImage_size = 128;// 256;

int kDataAugment_crop_num_factor = 10;
int upscale_factor = 3;

int calculate_valid_crop_size(const int crop_size, const int upscale_factor)
{
	return crop_size - (crop_size % upscale_factor);
}

// load  dataset
std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

int input_image_size;
void learning_and_test_super_resolution_dataset(torch::Device device)
{
	train_images.clear();
	printf("load images start\n");
	std::vector<std::string>& image_train_files = cpp_torch::getImageFiles(kDataRoot);
	std::vector<std::string>& image_test_files = cpp_torch::getImageFiles(kTestDataRoot1);

	input_image_size = calculate_valid_crop_size(kImage_size, upscale_factor);
	//printf("input_image_size:%d\n", input_image_size);

	cpp_torch::progress_display2 loding(image_train_files.size() + image_test_files.size() + 1);

	std::random_device rnd;
	std::mt19937 mt(rnd());

	for (int i = 0; i < image_train_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_train_files[i].c_str());

		cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);

		const int pad = 20 + kDataAugment_crop_num_factor;
		if (cvmat.size().width <= input_image_size+pad || cvmat.size().height <= input_image_size + pad)
		{
			cv::resize(cvmat, cvmat, cv::Size(input_image_size + pad, input_image_size + pad), 0, 0, INTER_CUBIC);
		}
		std::uniform_int_distribution<> rand_w(0, (int)cvmat.size().width - input_image_size - 1);
		std::uniform_int_distribution<> rand_h(0, (int)cvmat.size().height - input_image_size - 1);
		const int retry_max = 20;
		for (int k = 0; k < kDataAugment_crop_num_factor; k++)
		{
			int w = 0;
			int h = 0;

			w = rand_w(mt);
			h = rand_h(mt);
			int count = 0;
			while (w + input_image_size >= cvmat.size().width - 1 || h + input_image_size >= cvmat.size().height - 1)
			{
				w = rand_w(mt);
				h = rand_h(mt);
				count++;
				if (count == retry_max)
				{
					w = 0;
					h = 0;
					break;
				}
			}
			cv::Mat cut_img(cvmat, cv::Rect(w, h, input_image_size,input_image_size));
			cv::resize(cut_img, cut_img, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);

			cv::Mat traget_image = cut_img.clone();
			cv::resize(cut_img, cut_img, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);


			cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(cut_img);
			cpp_torch::Image& imgy = cpp_torch::cvutil::cvMatToImage(traget_image);

			ImageRGB2YCbCr(&imgx);
			ImageRGB2YCbCr(&imgy);

			cpp_torch::Image ycbcr[2];
			ycbcr[0] = imgx.clone();
			cpp_torch::ImageGetChannel(&ycbcr[0], 1);

			ycbcr[1] = imgy.clone();
			cpp_torch::ImageGetChannel(&ycbcr[1], 1);

			tiny_dnn::vec_t& vx = image_channel2vec_t(&ycbcr[0], 1, input_image_size / upscale_factor, input_image_size / upscale_factor);
			tiny_dnn::vec_t& vy = image_channel2vec_t(&ycbcr[1], 1, input_image_size, input_image_size);

			train_images.push_back(vx);
			train_labels.push_back(vy);
			if (count == retry_max) break;
		}
		loding += 1;
	}

	for (int i = 0; i < image_test_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_test_files[i].c_str());

		cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);
		cv::resize(cvmat, cvmat, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);

		cv::Mat traget_image = cvmat.clone();
		cv::resize(cvmat, cvmat, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);

		cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(cvmat);
		cpp_torch::Image& imgy = cpp_torch::cvutil::cvMatToImage(traget_image);
		ImageRGB2YCbCr(&imgx);
		ImageRGB2YCbCr(&imgy);


		cpp_torch::Image ycbcr[2];
		ycbcr[0] = imgx.clone();
		cpp_torch::ImageGetChannel(&ycbcr[0], 1);

		ycbcr[1] = imgy.clone();
		cpp_torch::ImageGetChannel(&ycbcr[1], 1);

		tiny_dnn::vec_t& vx = image_channel2vec_t(&ycbcr[0], 1, input_image_size / upscale_factor, input_image_size / upscale_factor);
		tiny_dnn::vec_t& vy = image_channel2vec_t(&ycbcr[1], 1, input_image_size, input_image_size);

		test_images.push_back(vx);
		test_labels.push_back(vy);
		loding += 1;
	}
	loding.end();
	printf("load images:%d %d\n", train_images.size(), test_images.size());

	cpp_torch::Net model;

	model.get()->setInput(1, input_image_size / upscale_factor, input_image_size / upscale_factor);
	model.get()->add_conv2d(1, 64, 5, 1, 2);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 64, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(64, 32, 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_conv2d(32, pow(upscale_factor, 2), 3, 1, 1);
	model.get()->add_ReLU();
	model.get()->add_pixel_shuffle(upscale_factor);

	for (auto w : model.get()->conv2d)
	{
		torch::nn::init::orthogonal_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
		//torch::nn::init::orthogonal_(w->weight, torch::nn::init::calculate_gain(torch::nn::init::Nonlinearity::ReLU));	//1.3
	}
	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);

	nn.input_dim(1, input_image_size / upscale_factor, input_image_size / upscale_factor);
	nn.output_dim(1, input_image_size, input_image_size);
	//nn.batch_shuffle = true;
	//nn.pre_make_batch = false;

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	auto optimizer =
		torch::optim::Adam(model.get()->parameters(), torch::optim::AdamOptions(0.0001));
	//auto optimizer =
	//	torch::optim::Adagrad (model.get()->parameters(), torch::optim::AdagradOptions(0.01));

	FILE* fp = fopen("loss.dat", "w");

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;

		nn.model.get()->train(false);

		float loss = nn.get_loss(test_images, test_labels, kTestBatchSize);
		fprintf(fp, "%f %f\n", loss/ kTestBatchSize);
		fflush(fp);
		printf("loss %.4f\n", loss); fflush(stdout);

		if (epoch % kLogInterval == 0)
		{
			std::vector<std::string>& image_files = cpp_torch::getImageFiles(kTestDataRoot2);


			float mse_loss = 0;
			float psnr = 0;
			for (int i = 0; i < image_files.size(); i++)
			{
				char imgfname[256];

				cv::Mat org_image = cv::imread(image_files[i].c_str());
				cv::resize(org_image, org_image, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
				sprintf(imgfname, "super_resolution_ref_%03d.png", i);
				cv::imwrite(imgfname, org_image);

				cv::Mat input;
				cv::resize(org_image, input, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);
				sprintf(imgfname, "super_resolution_befor_%03d.png", i);
				cv::imwrite(imgfname, input);

				cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(input);
				ImageRGB2YCbCr(&imgx);
				tiny_dnn::vec_t& vx = image_channel2vec_t(&imgx, 1, input_image_size / upscale_factor, input_image_size / upscale_factor);
				torch::Tensor x = cpp_torch::toTorchTensors(vx).view({ 1,1,input_image_size / upscale_factor,input_image_size / upscale_factor });

				torch::Tensor generated_img = nn.model.get()->forward(x.to(device));

				nn.model.get()->train(true);

				cv::Mat resizeImage;
				cv::resize(input, resizeImage, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
				sprintf(imgfname, "super_resolution_bicubic_%03d.png", i);
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
				tiny_dnn::vec_t& xx = cpp_torch::toTensor_t(generated_img[0], input_image_size*input_image_size);

				cpp_torch::Image& predict_Y = cpp_torch::ToImage(&xx[0], input_image_size, input_image_size, 1);
				cpp_torch::ImageGetChannel(&predict_Y, 1);
				cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[1], 2);
				cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[2], 3);
				cpp_torch::ImageYCbCr2RGB(&predict_Y);
				cv::Mat output = cpp_torch::cvutil::ImgeTocvMat(&predict_Y);
				sprintf(imgfname, "super_resolution_test_%03d.png", i);
				cv::imwrite(imgfname, output);

				float loss = cv::norm((org_image - output)) / (input_image_size*input_image_size);
				mse_loss += loss;
				psnr += 10.0*log10(1.0 / loss);
			}
			printf("psnr:%.4fdB  %.4f\n", psnr / image_files.size(), mse_loss / image_files.size());
			
			char save_model[256];
			sprintf(save_model, "model_scale%d.pt", upscale_factor);
			nn.save(std::string(save_model));
			//nn.load(std::string(save_model));
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

	nn.fit(&optimizer, train_images, train_labels, kTrainBatchSize, kNumberOfEpochs, on_enumerate_minibatch, on_enumerate_epoch);
	std::cout << "end training." << std::endl;
	fclose(fp);
}


auto main(int argc, char** argv) -> int {

	bool help = false;
	for (int i = 1; i < argc; i++)
	{
		BOOL_OPT(i, gpu, "--gpu");
		INT_OPT(i, kDataAugment_crop_num_factor, "--augment_crop");
		INT_OPT(i, kNumberOfEpochs, "--epoch");
		INT_OPT(i, kTrainBatchSize, "--batch");
		INT_OPT(i, kImage_size, "--image_size");
		FLOAT_OPT(i, upscale_factor, "--upscale");
		CSTR_OPT(i, kDataRoot, "--data_root");
		CSTR_OPT(i, kTestDataRoot1, "--testdata_root1");
		CSTR_OPT(i, kTestDataRoot2, "--testdata_root2");
		HELP_OPT(i, help, "--help");

	}
	printf("--data_root:%s\n", kDataRoot);
	printf("--gpu:%d\n", gpu);
	printf("--epoch:%d\n", kNumberOfEpochs);
	printf("--augment_crop:%d\n", kDataAugment_crop_num_factor);
	printf("--batch:%d\n", kTrainBatchSize);
	printf("--image_size:%d\n", kImage_size);
	printf("--testdata_root1:%s\n", kTestDataRoot1);
	printf("--testdata_root2:%s\n", kTestDataRoot2);
	if (help) exit(0);

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

	learning_and_test_super_resolution_dataset(device);
}
