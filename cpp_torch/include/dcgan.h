#ifndef _DCGAN_H_

#define _DCGAN_H_
/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/

/* Generative Adversarial Nets
*/
#define USE_LOSS_BCE
//#define USE_LOSS_MSE
//#define __Experiment

namespace cpp_torch
{
#ifdef 	__Experiment
	/*
	Because there is no BCEWithLogitsLoss in libtorch
	How do I define BCEWithLogitsLoss?
	*/
	inline torch::Tensor safe_log(torch::Tensor x)
	{
		torch::Tensor y = log(abs(x) + 1.0e-12);
		return y;
	}

	torch::Tensor BCEWithLogitsLoss(torch::Tensor o, torch::Tensor t)
	{
		//Specify the target of automatic differentiation with True
		if (!o.requires_grad()) o.set_requires_grad(true);
		//o.options().requires_grad(true);

		torch::Tensor y = -(t*safe_log(torch::sigmoid(o)) + (1 - t)*safe_log(1 - torch::sigmoid(o))).mean();
		
		return y;
	}
#define LOSS_FUNC BCEWithLogitsLoss
#endif

#ifndef 	__Experiment
#ifdef USE_LOSS_BCE
#define LOSS_FUNC	torch::binary_cross_entropy
#endif

#ifdef USE_LOSS_MSE
#define LOSS_FUNC	torch::mse_loss
#endif
#endif


	template <
		typename G_Model, typename D_Model>
	class DCGAN
	{
		cpp_torch::network_torch<G_Model> g_nn;
		cpp_torch::network_torch<D_Model> d_nn;
		torch::Tensor loss_G;
		torch::Tensor loss_D;
		int NZ = 100;
		tiny_dnn::timer time_measurement;
	public:
		torch::Device device;

		DCGAN(
			cpp_torch::network_torch<G_Model> g_nn_, 
			cpp_torch::network_torch<D_Model> d_nn_, torch::Device device_)
			:g_nn(g_nn_), d_nn(d_nn_), device(device_)
		{
		}

		/**
		* @param g_optimizer        Generator optimizing algorithm for training
		* @param d_optimizer        Discriminator optimizing algorithm for training
		* @param images             array of input data
		* @param kTrainBatchSize    number of mini-batch
		* @param kNumberOfEpochs    number of training epochs
		* @param nz                 number of random number
		* @param on_batch_enumerate callback for each mini-batch enumerate
		* @param on_epoch_enumerate callback for each epoch
		* assume
		*/
		bool train(
			torch::optim::Optimizer* g_optimizer,
			torch::optim::Optimizer* d_optimizer,
			std::vector<torch::Tensor> &images,
			int kTrainBatchSize,
			int kNumberOfEpochs,
			const int nz = 100,
			std::function <void(void)> on_batch_enumerate = {},
			std::function <void(void)> on_epoch_enumerate = {}
		)
		{
			if (images.size() < kTrainBatchSize ) {
				return false;
			}

			NZ = nz;
			time_measurement.start();

			//real
			torch::Tensor ones = torch::ones(kTrainBatchSize).to(device);
			
			//fake
			torch::Tensor zeros = torch::zeros(kTrainBatchSize).to(device);

			int batchNum;

			std::vector< torch::Tensor> batch_x;

			if (d_nn.pre_make_batch)
			{
				d_nn.generate_BATCH(images, batch_x);
				batchNum = batch_x.size();
			}

			//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
			torch::Tensor check_z = torch::randn({ kTrainBatchSize, nz, 1, 1 }).to(device);

			for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch)
			{
				if (!d_nn.pre_make_batch)
				{
					d_nn.generate_BATCH(images, batch_x);
					batchNum = batch_x.size();
				}

				for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
				{
					// ======= Generator training =========
					//Generate fake image

					//random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
					torch::Tensor z = torch::randn({ kTrainBatchSize, nz, 1, 1 }).to(device);
					torch::Tensor fake_img = g_nn.model.get()->forward(z);
					//cpp_torch::dump_dim("fake_img", fake_img);
					torch::Tensor fake_img_tensor = fake_img.detach();

					//	Calculate loss to make fake image look like real image(label 1)
					torch::Tensor out = d_nn.model.get()->forward(fake_img.to(device));

					//cpp_torch::dump_dim("out", out);
					//cpp_torch::dump_dim("ones", ones);
					loss_G = LOSS_FUNC(out.to(device), ones);
					AT_ASSERT(!std::isnan(loss_G.template item<float_t>()));

					d_optimizer->zero_grad();
					g_optimizer->zero_grad();
					loss_G.backward();
					g_optimizer->step();

					// ======= Discriminator training =========
					//Real image
					torch::Tensor real_img = batch_x[batch_idx].to(device);

					//	Calculate loss to distinguish real image from real image(label 1)
					torch::Tensor real_out = d_nn.model.get()->forward(real_img);

					torch::Tensor loss_D_real = LOSS_FUNC(real_out.to(device), ones);
					AT_ASSERT(!std::isnan(loss_D_real.template item<float_t>()));

					
					fake_img = fake_img_tensor;
					//fake_img = g_nn.model.get()->forward(z);

					//Calculate the loss so that fake images can be identified as fake images (label 0)
					torch::Tensor fake_out = d_nn.model.get()->forward(fake_img.to(device));

					//cpp_torch::dump_dim("fake_out", fake_out);
					//cpp_torch::dump_dim("zeros", zeros);
					torch::Tensor loss_D_fake = LOSS_FUNC(fake_out.to(device), zeros);
					AT_ASSERT(!std::isnan(loss_D_fake.template item<float_t>()));

					//Total loss of real and fake images
					loss_D = loss_D_real + loss_D_fake;

					d_optimizer->zero_grad();
					g_optimizer->zero_grad();
					loss_D.backward();
					d_optimizer->step();

					on_batch_enumerate();
				}
				on_epoch_enumerate();
				//printf("%f %f", get_generator_loss(), get_discriminator_loss());
			}
			time_measurement.stop();
			return true;
		}

		/**
		* @param g_optimizer        Generator optimizing algorithm for training
		* @param d_optimizer        Discriminator optimizing algorithm for training
		* @param images             array of input data
		* @param kTrainBatchSize    number of mini-batch
		* @param kNumberOfEpochs    number of training epochs
		* @param nz                 number of random number
		* @param on_batch_enumerate callback for each mini-batch enumerate
		* @param on_epoch_enumerate callback for each epoch
		* assume
		*/
		bool train(
			torch::optim::Optimizer* g_optimizer,
			torch::optim::Optimizer* d_optimizer,
			tiny_dnn::tensor_t &images, 
			int kTrainBatchSize,
			int kNumberOfEpochs,
			const int nz = 100,
			std::function <void(void)> on_batch_enumerate = {},
			std::function <void(void)> on_epoch_enumerate = {}
		)
		{
			std::vector<torch::Tensor> images_torch;
			toTorchTensors(images, images_torch);

			return train(g_optimizer, d_optimizer, images_torch, kTrainBatchSize, kNumberOfEpochs, nz, on_batch_enumerate, on_epoch_enumerate);
		}

		torch::Tensor generate_rand(int num)
		{
			torch::Tensor check_z = torch::rand({ num, NZ, 1, 1 }).to(device);
			return g_model.get()->forward(check_z);
		}
		torch::Tensor generate(torch::Tensor generate_seed)
		{
			generate_seed = generate_seed.to(device);
			return g_model.get()->forward(generate_seed);
		}

		float get_generator_loss()
		{
			return loss_G.sum().template item<float_t>();
			//float sum_loss = 0;
			//const int batchNum = loss_G.sizes()[0];
			//for (size_t i = 0; i < batchNum; i++) 
			//{
			//	sum_loss += loss_G[i].template item<float_t>();
			//}
			//return sum_loss;
		}
		float get_discriminator_loss()
		{
			return loss_D.sum().template item<float_t>();
			//float sum_loss = 0;
			//const int batchNum = loss_D.sizes()[0];
			//for (size_t i = 0; i < batchNum; i++) {
			//	sum_loss += loss_D[i].template item<float_t>();
			//}
			//return sum_loss;
		}

	};
}
#endif
