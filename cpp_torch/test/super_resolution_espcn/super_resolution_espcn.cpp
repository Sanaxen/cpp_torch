/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "test/include/images_mormalize.h"

#define USE_CUDA

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

const int64_t kImage_size = 256;

const int kDataAugment_crop_num_factor = 10;
const int upscale_factor = 3;
struct NetImpl : torch::nn::Module {
	NetImpl() :
		conv1(torch::nn::Conv2dOptions(1, 64, 5).with_bias(false).stride(1).padding(2)),
		conv2(torch::nn::Conv2dOptions(64, 64, 3).with_bias(false).stride(1).padding(1)),
		conv3(torch::nn::Conv2dOptions(64, 32, 3).with_bias(false).stride(1).padding(1)),
		conv4(torch::nn::Conv2dOptions(32, pow(upscale_factor, 2), 3).with_bias(false).stride(1).padding(1))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);

		torch::nn::init::orthogonal_(conv1->weight, torch::nn::init::calculate_gain(torch::nn::init::Nonlinearity::ReLU));
		torch::nn::init::orthogonal_(conv2->weight, torch::nn::init::calculate_gain(torch::nn::init::Nonlinearity::ReLU));
		torch::nn::init::orthogonal_(conv3->weight, torch::nn::init::calculate_gain(torch::nn::init::Nonlinearity::ReLU));
		torch::nn::init::orthogonal_(conv4->weight);
	}

	torch::Tensor forward(torch::Tensor x) {

		if (x.sum().template item<float_t>() == 0.0)
		{
			//abort();
		}
		//cpp_torch::dump_dim("x1", x);
		x = torch::relu(conv1->forward(x));
		//cpp_torch::dump_dim("x2", x);
		if (x.sum().template item<float_t>() == 0.0)
		{
			//abort();
		}
		x = torch::relu(conv2->forward(x));
		//cpp_torch::dump_dim("x3", x);
		if (x.sum().template item<float_t>() == 0.0)
		{
			//abort();
		}
		x = torch::relu(conv3->forward(x));
		//cpp_torch::dump_dim("x4", x);
		if (x.sum().template item<float_t>() == 0.0)
		{
			//abort();
		}
		x = torch::pixel_shuffle(conv4->forward(x), upscale_factor);
		//cpp_torch::dump_dim("x5", x);
		if (x.sum().template item<float_t>() == 0.0)
		{
			//abort();
		}
		return x;
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Conv2d conv4;
};
TORCH_MODULE(Net); // creates module holder for NetImpl

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
	std::vector<std::string>& image_train_files = cpp_torch::getImageFiles(kDataRoot + std::string("/BSDS300/images/train"));
	std::vector<std::string>& image_test_files = cpp_torch::getImageFiles(kDataRoot + std::string("/BSDS300/images/test"));

	input_image_size = calculate_valid_crop_size(kImage_size, upscale_factor);
	//printf("input_image_size:%d\n", input_image_size);

	cpp_torch::progress_display2 loding(image_train_files.size() + image_test_files.size() + 1);

	std::random_device rnd;
	std::mt19937 mt(rnd());

	for (int i = 0; i < image_train_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_train_files[i].c_str());

		cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);

		if (cvmat.size().width <= input_image_size || cvmat.size().height <= input_image_size)
		{
			cv::resize(cvmat, cvmat, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
		}
		std::uniform_int_distribution<> rand_w(0, (int)cvmat.size().width - input_image_size - 1);
		std::uniform_int_distribution<> rand_h(0, (int)cvmat.size().height - input_image_size - 1);
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
				if (count == 20)
				{
					w = 0;
					h = 0;
					break;
				}
			}
			cv::Mat cut_img(cvmat, cv::Rect(w, h, input_image_size,input_image_size));

			cv::resize(cut_img, cut_img, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
			//std::cout << "size " << cvmat.size() << std::endl;

			cv::Mat traget_image = cut_img.clone();

			cv::resize(cut_img, cut_img, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);
			//std::cout << "size " << cvmat.size() << std::endl;


			cpp_torch::Image& imgx = cpp_torch::cvutil::cvMatToImage(cut_img);
			cpp_torch::Image& imgy = cpp_torch::cvutil::cvMatToImage(traget_image);

			ImageRGB2YCbCr(&imgx);
			ImageRGB2YCbCr(&imgy);

			cpp_torch::Image ycbcr[2];
			ycbcr[0] = imgx.clone();
			cpp_torch::ImageGetChannel(&ycbcr[0], 1);

			ycbcr[1] = imgy.clone();
			cpp_torch::ImageGetChannel(&ycbcr[1], 1);

			//cpp_torch::ImageWrite("qqq.bmp", ycbcr[0]);
			//cpp_torch::ImageWrite("ppp.bmp", ycbcr[1]);

			tiny_dnn::vec_t& vx = image_channel2vec_t(&ycbcr[0], 1, input_image_size / upscale_factor, input_image_size / upscale_factor);
			tiny_dnn::vec_t& vy = image_channel2vec_t(&ycbcr[1], 1, input_image_size, input_image_size);

			train_images.push_back(vx);
			train_labels.push_back(vy);
			if (count == 20) break;
		}
		loding += 1;
	}

	for (int i = 0; i < image_test_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_test_files[i].c_str());

		cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);

		cv::resize(cvmat, cvmat, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
		//std::cout << "size " << cvmat.size() << std::endl;

		cv::Mat traget_image = cvmat.clone();

		cv::resize(cvmat, cvmat, cv::Size(input_image_size / upscale_factor, input_image_size / upscale_factor), 0, 0, INTER_CUBIC);
		//std::cout << "size " << cvmat.size() << std::endl;

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

#if 0
	//image normalize (mean 0 and variance 1)
	float mean = 0.0;
	float stddiv = 1.0;
	cpp_torch::test::images_normalize(train_images, mean, stddiv);
	cpp_torch::test::images_normalize_(train_labels, mean, stddiv);
	cpp_torch::test::images_normalize_(test_images, mean, stddiv);
	cpp_torch::test::images_normalize_(test_labels, mean, stddiv);
	printf("mean:%f stddiv:%f\n", mean, stddiv);
#else
	//image normalize [-1, 1]
	//float max_v = 0.0;
	//float min_v = 0.0;
	//cpp_torch::test::images_normalize_11(train_images, max_v, min_v);
	//cpp_torch::test::images_normalize_11_(train_labels, max_v, min_v);
	//cpp_torch::test::images_normalize_11_(test_images, max_v, min_v);
	//cpp_torch::test::images_normalize_11_(test_labels, max_v, min_v);

	//cpp_torch::test::images_normalize_01(train_images, max_v, min_v);
	//cpp_torch::test::images_normalize_01_(train_labels, max_v, min_v);
	//cpp_torch::test::images_normalize_01_(test_images, max_v, min_v);
	//cpp_torch::test::images_normalize_01_(test_labels, max_v, min_v);
	//printf("max_v:%f min_v:%f\n", max_v, min_v);

#endif

	Net model;

	cpp_torch::network_torch<Net> nn(model, device);

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

		float loss = nn.get_loss(test_images, test_labels, 1);
		fprintf(fp, "%f %f\n", loss/ test_images.size());
		fflush(fp);
		printf("loss %.4f\n", loss); fflush(stdout);


		std::vector<std::string>& image_files = cpp_torch::getImageFiles(kDataRoot + std::string("/Set5"));


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
			//cpp_torch::ImageWrite("www2.bmp", &imgx);

			torch::Tensor x = cpp_torch::toTorchTensors(vx).view({ 1,1,input_image_size / upscale_factor,input_image_size / upscale_factor });

			//tiny_dnn::vec_t& vv = cpp_torch::toTensor_t(x, (input_image_size / upscale_factor)*(input_image_size / upscale_factor));
			//cpp_torch::Image tttt = cpp_torch::vec_t2image(vv, 1, input_image_size / upscale_factor, input_image_size / upscale_factor);
			//cpp_torch::ImageWrite("www.bmp", &tttt);

			torch::Tensor generated_img = nn.model.get()->forward(x.to(device));
			//cpp_torch::dump_dim("x1", x);
			//cpp_torch::dump_dim("x1", generated_img);

			nn.model.get()->train(true);

			cv::Mat resizeImage;
			cv::resize(input, resizeImage, cv::Size(input_image_size, input_image_size), 0, 0, INTER_CUBIC);
			sprintf(imgfname, "super_resolution_bicubic_%03d.png", i);
			cv::imwrite(imgfname, resizeImage);

			//cv::imwrite("aaa.bmp", resizeImage);
			cpp_torch::Image& image_CbCr = cpp_torch::cvutil::cvMatToImage(resizeImage);

			//cpp_torch::ImageWrite("ddd.bmp", image_CbCr);
			cpp_torch::ImageRGB2YCbCr(&image_CbCr);
			//cpp_torch::ImageWrite("bbb.bmp", image_CbCr);
			cpp_torch::Image ycbcr[3];
			ycbcr[0] = image_CbCr.clone();
			ycbcr[1] = image_CbCr.clone();
			ycbcr[2] = image_CbCr.clone();
			cpp_torch::ImageGetChannel(&ycbcr[0], 1);
			cpp_torch::ImageGetChannel(&ycbcr[1], 2);
			cpp_torch::ImageGetChannel(&ycbcr[2], 3);
			//cpp_torch::ImageYCbCr2RGB(image_CbCr);
			//cpp_torch::ImageWrite("ccc.bmp", image_CbCr);


			//tiny_dnn::vec_t& xx = cpp_torch::toTensor_t((generated_img[0] * stddiv + mean), input_image_size*input_image_size);
			//tiny_dnn::vec_t& xx = cpp_torch::toTensor_t(((generated_img[0]+1) * (max_v -min_v)*0.5).clamp(0,255), input_image_size*input_image_size);
			//tiny_dnn::tensor_t& xx = cpp_torch::toTensor_t((generated_img[0] * (max_v -min_v) + min_v), 1, input_image_size, input_image_size);
			//tiny_dnn::tensor_t& xx = cpp_torch::toTensor_t((generated_img[0]+1)*128, 1, input_image_size, input_image_size);
			tiny_dnn::vec_t& xx = cpp_torch::toTensor_t(generated_img[0], input_image_size*input_image_size);

			//cpp_torch::Image* tmp = cpp_torch::cvutil::cvMatToImage(TestImagesY[0]);
			//cpp_torch::ImageWrite("eee.bmp", tmp);
			//tiny_dnn::vec_t tmptmp = cpp_torch::image_channel2vec_t(tmp, 1, 64, 64);
			//torch::Tensor xxx = cpp_torch::toTorchTensors(tmptmp).view({ 1,64,64 });

			//tiny_dnn::tensor_t& xx = cpp_torch::toTensor_t(xxx, 1, input_image_size, input_image_size);

			cpp_torch::Image& predict_Y = cpp_torch::ToImage(&xx[0], input_image_size, input_image_size, 1);
			//cpp_torch::ImageWrite("fff.bmp", &predict_Y);

			cpp_torch::ImageGetChannel(&predict_Y, 1);
			//cpp_torch::ImageWrite("ggg.bmp", &predict_Y);
			//cpp_torch::ImageChgChannel(predict_Y, ycbcr[0], 1);
			cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[1], 2);
			cpp_torch::ImageChgChannel(&predict_Y, &ycbcr[2], 3);
			//cpp_torch::ImageWrite("hhh.bmp", &predict_Y);
			cpp_torch::ImageYCbCr2RGB(&predict_Y);
			//cpp_torch::ImageWrite("iii.bmp", &predict_Y);
			cv::Mat output = cpp_torch::cvutil::ImgeTocvMat(&predict_Y);
			sprintf(imgfname, "super_resolution_test_%03d.png", i);
			cv::imwrite(imgfname, output);

			float loss = cv::norm((org_image - output)) / (input_image_size*input_image_size);
			mse_loss += loss;
			psnr += 10.0*log10(1.0 / loss);
		}
		printf("psnr:%.4f  %.4fdB\n", psnr / image_files.size(), mse_loss / image_files.size());

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

	learning_and_test_super_resolution_dataset(device);
}
