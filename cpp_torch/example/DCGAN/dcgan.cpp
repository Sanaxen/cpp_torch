/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#define USE_OPENCV_UTIL
#include "cpp_torch.h"
#include "dcgan.h"
#include "test/include/images_normalize.h"
#include "util/command_line.h"

#define USE_CUDA

#define IMAGE_SIZE	64
#define IMAGE_CHANNEL	3

// the path to the root of the dataset folder.
char* kDataRoot = "./data/image";

// The batch size for training.
int64_t kTrainBatchSize = 64;

// The batch size for testing.
int64_t kTestBatchSize = 1000;

// The number of epochs to train.
int64_t kNumberOfEpochs = 1000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int kRndArraySize = 100;

bool gpu = true;
int kDataAugment = -1;
float drop_rate = 0.2;

//relates to the depth of feature maps carried through the generator
const int ngf = 64;
//sets the depth of feature maps propagated through the discriminator
const int ndf = 64;

//learning rate for training. As described in the DCGAN paper, this number should be 0.0002
float lr = 0.0002;
//beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
float beta1 = 0.5;
float beta2 = 0.5;//0.999

float discriminator_flip = 0.0;
float discriminator_range = 0.0;
bool discriminator_noise = false;

int checkStart = 150;

// load  dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void learning_and_test_dcgan_dataset(torch::Device device)
{
	train_images.clear();
	printf("load images start\n");
	std::vector<std::string>& image_files = cpp_torch::getImageFiles(kDataRoot);

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
	const int upsample = kDataAugment - 1;
	for (int i = 0; i < image_files.size(); i++)
	{
		cpp_torch::Image& img = cpp_torch::readImage(image_files[i].c_str());

		if (kDataAugment <= 1)
		{
			cv::Mat cvmat = cpp_torch::cvutil::ImgeTocvMat(&img);
			cv::resize(cvmat, cvmat, cv::Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, INTER_CUBIC);
			cpp_torch::Image& img = cpp_torch::cvutil::cvMatToImage(cvmat);

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
	g_model.get()->add_conv_transpose2d(nz, ngf*8, 4, 1, 0, 0, 1, false);
	g_model.get()->add_bn2d();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 8, ngf * 4, 4, 2, 1, 0, 1, false);
	g_model.get()->add_bn2d();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 4, ngf * 2, 4, 2, 1, 0, 1, false);
	g_model.get()->add_bn2d();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf * 2, ngf, 4, 2, 1, 0, 1, false);
	g_model.get()->add_bn2d();
	g_model.get()->add_ReLU();
	g_model.get()->add_conv_transpose2d(ngf, IMAGE_CHANNEL, 4, 2, 1, 0, 1, false);
	g_model.get()->add_Tanh();

	cpp_torch::Net  d_model;
	d_model.get()->setInput(IMAGE_CHANNEL, ndf, ndf);
	d_model.get()->add_conv2d(IMAGE_CHANNEL, ndf, 4, 2, 1, 1, false);
	d_model.get()->add_bn2d();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf, ndf*2, 4, 2, 1, 1, false);
	d_model.get()->add_bn2d();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*2, ndf*4, 4, 2, 1, 1, false);
	d_model.get()->add_bn2d();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*4, ndf*8, 4, 2, 1, 1, false);
	d_model.get()->add_bn2d();
	d_model.get()->add_LeakyReLU(0.2);
	d_model.get()->add_dropout(drop_rate);

	d_model.get()->add_conv2d(ndf*8, 1, 4, 1, 0, 1, false);
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
			torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)));
	auto d_optimizer =
		torch::optim::Adam(d_model.get()->parameters(),
			torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)));


	cpp_torch::DCGAN<cpp_torch::Net, cpp_torch::Net> dcgan(g_nn, d_nn, device);
	dcgan.discriminator_flip = discriminator_flip;
	dcgan.discriminator_noise = discriminator_noise;
	dcgan.noize_range = discriminator_range;

	float loss_min = 99999.0;
	int none_update_count = 0;

	FILE* fp = fopen("loss.dat", "w");
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;

		float g_loss = dcgan.get_generator_loss();
		float d_loss = dcgan.get_discriminator_loss();
		fprintf(fp, "%f %f\n", g_loss, d_loss);
		fflush(fp);

		g_nn.model.get()->train(false);
		torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
		g_nn.model.get()->train(true);

		//generated_img = (mean + generated_img.mul(stddiv)).clamp(0, 255);
		//generated_img = ((1+generated_img).mul(128)).clamp(0, 255);
		generated_img = ((1 + generated_img).mul(0.5*(max - min)) + min).clamp(0, 255);

		if (epoch > checkStart)
		{
			if (loss_min > fabs(g_loss - d_loss))
			{
				loss_min = fabs(g_loss - d_loss);
				g_nn.save(std::string("g_model.pt"));
				d_nn.save(std::string("d_model.pt"));
				cv::Mat& img = cpp_torch::cvutil::ImageWrite(generated_img, 8, 8, "image_array_bst.bmp", 2);
				cv::imshow("image_array_bst.bmp", img);
				cv::waitKey(5000);
				none_update_count = 0;
			}
			else
			{
				none_update_count++;
			}
			if (none_update_count > 100)
			{
				throw "stopping";
			}
		}
			
#if 0
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
				cv::waitKey(5000);
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
#endif
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

	try
	{
		dcgan.train(&g_optimizer, &d_optimizer, train_images, tiny_dnn::tensor_t{}, kTrainBatchSize, kNumberOfEpochs, nz, on_enumerate_minibatch, on_enumerate_epoch);
	}
	catch (...)
	{

	}
	std::cout << "end training." << std::endl;
	fclose(fp);

	//g_nn.save(std::string("g_model.pt"));
	//d_nn.save(std::string("d_model.pt"));
}


int main(int argc, char** argv)
{
	bool help = false;
	for (int i = 1; i < argc; i++)
	{
		BOOL_OPT(i, gpu, "--gpu");
		INT_OPT(i, kDataAugment, "--augment");
		INT_OPT(i, kNumberOfEpochs, "--epoch");
		INT_OPT(i, kTrainBatchSize, "--batch");
		INT_OPT(i, checkStart, "--checkStart");
		FLOAT_OPT(i, drop_rate, "--drop_rate");
		FLOAT_OPT(i, discriminator_flip, "--d_flip");
		BOOL_OPT(i, discriminator_noise, "--d_noise");
		FLOAT_OPT(i, discriminator_range, "--d_noise_range");
		CSTR_OPT(i, kDataRoot, "--data_root");
		FLOAT_OPT(i, lr, "--lr");
		FLOAT_OPT(i, beta1, "--beta1");
		HELP_OPT(i, help, "--help");
	}
	printf("--data_root:%s\n", kDataRoot);
	printf("--gpu:%d\n", gpu);
	printf("--epoch:%d\n", kNumberOfEpochs);
	printf("--augment:%d\n", kDataAugment);
	printf("--batch:%d\n", kTrainBatchSize);
	printf("--d_flip:%f\n", discriminator_flip);
	printf("--d_noise:%s\n", discriminator_noise ? "true" : "false");
	printf("--d_noise_range:%f\n", discriminator_range);
	printf("--drop_rate:%f\n", drop_rate);
	printf("--lr:%f\n", lr);
	printf("--beta1:%f\n", beta1);
	printf("--checkStart:%d\n", checkStart);
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

	learning_and_test_dcgan_dataset(device);

	return 0;
}
