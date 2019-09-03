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

#define IMAGE_SIZE	64
#define IMAGE_CHANNEL	3

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int kRndArraySize = 100;

const bool kDataAugment = true;
const float drop_rate = 0.2;
const int ngf = 64;
const int ndf = 64;

const float discriminator_flip = 0.0;
const bool discriminator_noise = true;

// load  dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void learning_and_test_dcgan_dataset(torch::Device device)
{
	train_images.clear();
	printf("load images start\n");
	std::vector<std::string>& image_files = cpp_torch::getImageFiles(kDataRoot + std::string("/image"));

	cpp_torch::progress_display2 loding(image_files.size() + 1);
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> rand_flip(0.0, 1.0);

#ifdef USE_CV_SUPERRES
	cv::Ptr<cv::superres::SuperResolution> superResolution = cv::superres::createSuperResolution_BTVL1();
	superResolution->setTemporalAreaRadius(1);
	superResolution->setIterations(2);
#endif

	const int base_size = 256;
	const int random_sift = 15;	//base_size < 10%
	const int extend_base_size = base_size + random_sift;
	const int upsample = 5;
	for (int i = 0; i < image_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_files[i].c_str());

		if (!kDataAugment)
		{
			tiny_dnn::vec_t& v = image2vec_t(&img, IMAGE_CHANNEL, img.height, img.width/*, 1.0/255.0*/);
			train_images.push_back(v);
		}
		else
		{
			cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);

#ifdef USE_CV_SUPERRES
			if (cvmat.rows < extend_base_size || cvmat.cols < extend_base_size)
			{
				float scale1 = (extend_base_size) / (float)cvmat.rows;
				float scale2 = (extend_base_size) / (float)cvmat.cols;
				superResolution->setScale(std::max(scale1, scale2));

				try
				{
					superResolution->nextFrame(cvmat);
				}
				catch (cv::Exception& err)
				{
					cout << "error " << err.what() << endl;
				}
			}
#endif
			cv::resize(cvmat, cvmat, cv::Size(extend_base_size, extend_base_size), 0, 0, INTER_LANCZOS4);

			//Cut one image a little while shifting it to multiple sheets
			std::uniform_int_distribution<> rand_w(0, random_sift);
			std::uniform_int_distribution<> rand_h(0, random_sift);
			for (int k = 0; k < upsample; k++)
			{
				int w = 0;
				int h = 0;

				w = rand_w(mt);
				h = rand_h(mt);
				cv::Mat cut_img(cvmat, cv::Rect(w, h, base_size, base_size));
				cv::resize(cut_img, cut_img, cv::Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, INTER_CUBIC);

				if (rand_flip(mt) > 0.5)
				{
					cv::Mat dest;
					cv::flip(cut_img, dest, 1);
					cut_img = dest.clone();
				}
				char fnm[256];
				sprintf(fnm, "dump\\%04d.png", i * upsample + k);
				cv::imwrite(fnm, cut_img);

				cpp_torch::Image& img = cpp_torch::cvutil::cvMatToImage(cut_img);
				tiny_dnn::vec_t& v = image2vec_t(&img, IMAGE_CHANNEL, img.height, img.width/*, 1.0/255.0*/);

				train_images.push_back(v);
			}
		}
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
	//cpp_torch::test::images_normalize_11(train_images);
	float min = 0.0;
	float max = 255.0;
	cpp_torch::test::images_normalize_11(train_images, max, min);

#endif

	loding.end();
	printf("load images:%d\n", train_images.size());

	const int nz = kRndArraySize;

	cpp_torch::Net  g_model;
	g_model.get()->setInput(nz, 1, 1);
	g_model.get()->add_conv_transpose2d(nz, ngf*8, 4, 1, 0);
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

	cpp_torch::Net  d_model;
	d_model.get()->setInput(IMAGE_CHANNEL, ndf, ndf);
	d_model.get()->add_conv2d(IMAGE_CHANNEL, ndf, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf, ndf*2, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*2, ndf*4, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*4, ndf*8, 4, 2, 1);
	d_model.get()->add_bn();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*8, 1, 4, 1, 0);
	d_model.get()->add_Squeeze();
#ifdef USE_LOSS_BCE
	d_model.get()->add_Sigmoid();
#endif

	cpp_torch::network_torch<cpp_torch::Net> g_nn(g_model, device);
	cpp_torch::network_torch<cpp_torch::Net> d_nn(d_model, device);

	//Generator  100 -> 3 x 64 x 64
	g_nn.input_dim(nz, 1, 1);
	g_nn.output_dim(IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE);

	//Discriminator  3 x 64 x 64 -> 0(fake) or 1(real)
	d_nn.input_dim(IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE);
	d_nn.output_dim(1, 1, 1);
	d_nn.classification = false;
	d_nn.batch_shuffle = true;

	//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
	torch::Tensor check_z = torch::randn({ kTrainBatchSize, nz, 1, 1 }).to(device);

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	//Adam !! Radford et. al. 2015
	auto g_optimizer =
		torch::optim::Adam(g_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));
	auto d_optimizer =
		torch::optim::Adam(d_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));


	cpp_torch::DCGAN<cpp_torch::Net, cpp_torch::Net> dcgan(g_nn, d_nn, device);
	dcgan.discriminator_flip = discriminator_flip;
	dcgan.discriminator_noise = discriminator_noise;

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

			char fname[64];
			sprintf(fname, "generated_images/image_array%d.png", epoch);
			cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, fname, 2);
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
