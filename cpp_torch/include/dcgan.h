#ifndef _DCGAN_H_

#define _DCGAN_H_

namespace cpp_torch
{
	template <
		typename G_Model, typename D_Model>
	class DCGAN
	{
		cpp_torch::network_torch<G_Model> g_nn;
		cpp_torch::network_torch<D_Model> d_nn;
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

			torch::Tensor ones = torch::ones(kTrainBatchSize).to(device);
			torch::Tensor zeros = torch::zeros(kTrainBatchSize).to(device);

			const int batchNum = images.size() / kTrainBatchSize; ;

			std::vector< torch::Tensor> batch_x;

			if (d_nn.pre_make_batch)
			{
				d_nn.generate_BATCH(images, batch_x);
			}

			torch::Tensor check_z = torch::rand({ kTrainBatchSize, nz, 1, 1 }).to(device);

			torch::Tensor loss_G;
			torch::Tensor loss_D;
			for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch)
			{
				if (!d_nn.pre_make_batch)
				{
					d_nn.generate_BATCH(images, batch_x);
				}

				for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
				{
					// ======= Generator training =========
					//Generate fake image
					torch::Tensor z = torch::rand({ kTrainBatchSize, nz, 1, 1 }).to(device);
					torch::Tensor fake_img = g_nn.model.get()->forward(z);
					//cpp_torch::dump_dim("fake_img", fake_img);
					torch::Tensor fake_img_tensor = fake_img.detach().to(device);

					//	Calculate loss to make fake image look like real image(label 1)
					torch::Tensor out = d_nn.model.get()->forward(fake_img);

					//cpp_torch::dump_dim("out", out);
					//cpp_torch::dump_dim("ones", ones);
					loss_G = torch::mse_loss(out, ones);

					g_optimizer->zero_grad();
					d_optimizer->zero_grad();
					loss_G.backward();
					g_optimizer->step();

					// ======= Discriminator training =========
					//Real image
					torch::Tensor real_img = batch_x[batch_idx].to(device);

					//	Calculate loss to distinguish real image from real image(label 1)
					torch::Tensor real_out = d_nn.model.get()->forward(real_img);
					torch::Tensor loss_D_real = torch::mse_loss(real_out, ones);

					fake_img = fake_img_tensor;

					//Calculate the loss so that fake images can be identified as fake images (label 0)
					torch::Tensor fake_out = d_nn.model.get()->forward(fake_img_tensor.to(device));

					//cpp_torch::dump_dim("fake_out", fake_out);
					//cpp_torch::dump_dim("zeros", zeros);
					torch::Tensor loss_D_fake = torch::mse_loss(fake_out, zeros);

					//Total loss of real and fake images
					loss_D = loss_D_real + loss_D_fake;

					g_optimizer->zero_grad();
					d_optimizer->zero_grad();
					loss_D.backward();
					d_optimizer->step();

//					if (epoch % 10 == 0)
//					{
//						g_nn.model.get()->train(false);
//						torch::Tensor generated_img = g_nn.model.get()->forward(check_z);
//#if 10			
//						cpp_torch::TensorToImageFile(generated_img[0], "aaa.bmp", 255.0);
//#else
//						//cpp_torch::dump_dim("generated_img", generated_img[0]);
//						tiny_dnn::tensor_t& img = cpp_torch::toTensor_t(generated_img[0], 3, 64, 64);
//
//						for (int i = 0; i < img[0].size(); i++)
//						{
//							img[0][i] *= 255;
//						}
//						cpp_torch::Image* rgb_img = cpp_torch::vec_t2image(img[0], 3, 64, 64);
//
//						//printf("ImageWrite\n");
//						cpp_torch::ImageWrite("aaa.bmp", rgb_img);
//						delete rgb_img;
//						//printf("ImageWrite end.\n");
//#endif
//					}
					on_batch_enumerate();
				}
				on_epoch_enumerate();
				g_nn.model.get()->train(true);
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
	};
}
#endif
