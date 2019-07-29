/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"
#include "dcgan.h"

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



struct GeneratorImpl : torch::nn::Module {
	GeneratorImpl(): 
		conv1(torch::nn::Conv2dOptions(100, 256,4).with_bias(false).transposed(true).stride(1).padding(0)),
		batchnml1(torch::nn::BatchNormOptions(256)),
		conv2(torch::nn::Conv2dOptions(256, 128, 4).with_bias(false).transposed(true).stride(2).padding(1)),
		batchnml2(torch::nn::BatchNormOptions(128)),
		conv3(torch::nn::Conv2dOptions(128, 64, 4).with_bias(false).transposed(true).stride(2).padding(1)),
		batchnml3(torch::nn::BatchNormOptions(64)),
		conv4(torch::nn::Conv2dOptions(64, 32, 4).with_bias(false).transposed(true).stride(2).padding(1)),
		batchnml4(torch::nn::BatchNormOptions(32)),
		conv5(torch::nn::Conv2dOptions(32, 3, 4).with_bias(false).transposed(true).stride(2).padding(1))
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
		x = torch::relu(batchnml1->forward(x));

		x = conv2->forward(x);
		x = torch::relu(batchnml2->forward(x));
		
		x = conv3->forward(x);
		x = torch::relu(batchnml3->forward(x));

		x = conv4->forward(x);
		x = torch::relu(batchnml4->forward(x));

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
	torch::nn::BatchNorm batchnml1;
	torch::nn::BatchNorm batchnml2;
	torch::nn::BatchNorm batchnml3;
	torch::nn::BatchNorm batchnml4;
};
TORCH_MODULE(Generator); // creates module holder for NetImpl

struct DiscriminatorImpl : torch::nn::Module {
	DiscriminatorImpl() :
		conv1(torch::nn::Conv2dOptions(3, 32, 4).with_bias(false).stride(2).padding(1)),
		conv2(torch::nn::Conv2dOptions(32, 64, 4).with_bias(false).stride(2).padding(1)),
		batchnml1(torch::nn::BatchNormOptions(64)),
		conv3(torch::nn::Conv2dOptions(64, 128, 4).with_bias(false).stride(2).padding(1)),
		batchnml2(torch::nn::BatchNormOptions(128)),
		conv4(torch::nn::Conv2dOptions(128, 256, 4).with_bias(false).stride(2).padding(1)),
		batchnml3(torch::nn::BatchNormOptions(256)),
		conv5(torch::nn::Conv2dOptions(256, 1, 4).with_bias(false).stride(1).padding(0))
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
		x = torch::leaky_relu(x, 0.2);

		x = conv2->forward(x);
		x = batchnml1->forward(x);
		x = torch::leaky_relu(x, 0.2);

		x = conv3->forward(x);
		x = batchnml2->forward(x);
		x = torch::leaky_relu(x, 0.2);

		x = conv4->forward(x);
		x = batchnml3->forward(x);
		x = torch::leaky_relu(x, 0.2);

		x = conv5->forward(x);
		//cpp_torch::dump_dim("x", x);
		x = x.squeeze();
		//cpp_torch::dump_dim("squeeze->x", x);

		return x;
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Conv2d conv4;
	torch::nn::Conv2d conv5;
	torch::nn::BatchNorm batchnml1;
	torch::nn::BatchNorm batchnml2;
	torch::nn::BatchNorm batchnml3;
};
TORCH_MODULE(Discriminator); // creates module holder for NetImpl

// load  dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;


void learning_and_test_dcgan_dataset(torch::Device device)
{
	train_images.clear();
	for (int i = 0; i <= kNumberOfTrainImages; i++)
	{
		char buf[32];
		sprintf(buf, "%d", i);
		std::string& file = kDataRoot + std::string("/") + std::string(buf) + ".jpg";

		cpp_torch::Image* img = cpp_torch::readImage(file.c_str());
		tiny_dnn::vec_t& v = image2vec_t(img, 3, 64, 64);
		delete img;

		train_images.push_back(v);
	}
	printf("load images:%d\n", train_images.size());

	Generator g_model;
	Discriminator d_model;

	cpp_torch::network_torch<Generator> g_nn(g_model, device);
	cpp_torch::network_torch<Discriminator> d_nn(d_model, device);

	//100 -> 3 x 64 x 64
	g_nn.input_dim(1, 1, 100);
	g_nn.output_dim(3, 64, 64);
	g_nn.classification = false;
	g_nn.batch_shuffle = false;

	//3 x 64 x 64 -> 0 or 1
	d_nn.input_dim(3, 64, 64);
	d_nn.output_dim(1, 1, 1);
	d_nn.classification = true;
	d_nn.batch_shuffle = false;

	const int nz = 100;
	torch::Tensor check_z = torch::rand({ kTrainBatchSize, nz, 1, 1 }).to(device);

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;

		if (epoch % kLogInterval == 0)
		{
			g_nn.model.get()->train(false);
			torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
			g_nn.model.get()->train(true);

			//char fname[32];
			//for (int i = 0; i < kTrainBatchSize; i++)
			//{
			//	sprintf(fname, "gen%d.bmp", i);
			//	cpp_torch::TensorToImageFile(generated_img[i], fname, 255.0);
			//}
			cpp_torch::TensorToImageFile(generated_img[0], "gen0.bmp", 255.0);
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

	auto g_optimizer =
		torch::optim::Adam(g_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));
	auto d_optimizer =
		torch::optim::Adam(d_model.get()->parameters(),
			torch::optim::AdamOptions(0.0002).beta1(0.5).beta2(0.999));

#if 10

	cpp_torch::DCGAN<Generator, Discriminator> dcgan(g_nn, d_nn, device);

	dcgan.train(&g_optimizer, &d_optimizer, train_images, kTrainBatchSize, kNumberOfEpochs, 100, on_enumerate_minibatch, on_enumerate_epoch);

#else
	torch::Tensor ones = torch::ones(kTrainBatchSize).to(device);
	torch::Tensor zeros = torch::zeros(kTrainBatchSize).to(device);
	torch::Tensor loss_g;
	torch::Tensor loss_d;

	std::vector<torch::Tensor> images_torch;
	cpp_torch::toTorchTensors(train_images, images_torch);

	const int batchNum = images_torch.size() / kTrainBatchSize; ;

	std::vector< torch::Tensor> batch_x;

	d_nn.generate_BATCH(images_torch, batch_x);


	torch::Tensor loss_G;
	torch::Tensor loss_D;
	for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch)
	{
		//std::cout << epoch << "/" << kNumberOfEpochs << std::endl;

		for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
		{
			//std::cout << "batch " << batch_idx << "/" << batchNum << std::endl;
			
			//Generate fake image
			torch::Tensor z = torch::rand({ kTrainBatchSize, nz, 1, 1 }).to(device);
			torch::Tensor fake_img = g_model.get()->forward(z);
			//cpp_torch::dump_dim("fake_img", fake_img);
			torch::Tensor fake_img_tensor = fake_img.detach().to(device);

			//	Calculate loss to make fake image look like real image(label 1)
			torch::Tensor out = d_model.get()->forward(fake_img);

			//cpp_torch::dump_dim("out", out);
			//cpp_torch::dump_dim("ones", ones);
			loss_G = torch::mse_loss(out, ones);
			
			g_optimizer.zero_grad();
			d_optimizer.zero_grad();
			loss_G.backward();
			g_optimizer.step();

			// ======= Discriminator training =========
			//Real image
			torch::Tensor real_img = batch_x[batch_idx].to(device);

			//	Calculate loss to distinguish real image from real image(label 1)
			torch::Tensor real_out = d_model.get()->forward(real_img);
			torch::Tensor loss_D_real = torch::mse_loss(real_out, ones);

			fake_img = fake_img_tensor;

			//Calculate the loss so that fake images can be identified as fake images (label 0)
			torch::Tensor fake_out = d_model.get()->forward(fake_img_tensor.to(device));

			//cpp_torch::dump_dim("fake_out", fake_out);
			//cpp_torch::dump_dim("zeros", zeros);
			torch::Tensor loss_D_fake = torch::mse_loss(fake_out, zeros);

			//Total loss of real and fake images
			loss_D = loss_D_real + loss_D_fake;

			g_optimizer.zero_grad();
			d_optimizer.zero_grad();
			loss_D.backward();
			d_optimizer.step();

			if (epoch % 10 == 0)
			{
				torch::Tensor generated_img = g_model.get()->forward(check_z);
#if 10			
				cpp_torch::TensorToImageFile(generated_img[0], "aaa.bmp", 255.0);
#else
				//cpp_torch::dump_dim("generated_img", generated_img[0]);
				tiny_dnn::tensor_t& img = cpp_torch::toTensor_t(generated_img[0], 3, 64, 64);
				
				for (int i = 0; i < img[0].size(); i++)
				{
					img[0][i] *= 255;
				}
				cpp_torch::Image* rgb_img = cpp_torch::vec_t2image(img[0], 3, 64, 64);

				//printf("ImageWrite\n");
				cpp_torch::ImageWrite("aaa.bmp", rgb_img);
				delete rgb_img;
				//printf("ImageWrite end.\n");
#endif
			}
			on_enumerate_minibatch();
		}
		on_enumerate_epoch();
	}
#endif

	std::cout << "end training." << std::endl;

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

	learning_and_test_dcgan_dataset(device);
}
