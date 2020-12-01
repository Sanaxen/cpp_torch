#ifndef libtorch_UTILS
#define libtorch_UTILS
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#include <random>
#include "util/utils.h"
#ifdef USE_IMAGE_UTIL
#include "util/Image.hpp"
#endif

namespace cpp_torch
{
	inline void nop() {
		// do nothing
	}
	/**
	* error exception class
	**/
	class error_exception : public std::exception {
	public:
		explicit error_exception(const std::string &msg) : msg_(msg) {
			fprintf(stderr, "ERROR:%s\n", msg.c_str());
			fflush(stderr);
		}
		const char *what() const throw() override { return msg_.c_str(); }

	private:
		std::string msg_;
	};

	inline size_t tensor_flatten_size(torch::Tensor& t)
	{
		size_t s = 1;
		if (t.dim())
		{
			for (int i = 0; i < t.sizes().size(); i++)
			{
				s *= t.sizes()[i];
			}
			return s;
		}
		return 0;
	}

	inline void dump_dim(const std::string & s, torch::Tensor& t)
	{
		printf("%s dim:%d ", s.c_str(), t.dim());
		if (t.dim())
		{
			for (int i = 0; i < t.sizes().size() - 1; i++)
			{
				printf("%d x", t.sizes()[i]);
			}
			printf("%d\n", t.sizes()[t.sizes().size() - 1]);
		}
		fflush(stdout);
	}
	inline void dump_dim(char* s, torch::Tensor& t)
	{
		dump_dim(std::string(s), t);
	}
	void label2vec(const std::vector<tiny_dnn::label_t>& labels, std::vector<tiny_dnn::vec_t>& vec, int max_label)
	{
		vec.clear();
		vec.resize(labels.size());
#pragma omp parallel for
		for (int i = 0; i < labels.size(); i++)
		{
			tiny_dnn::vec_t t(max_label, 0);
			t[labels[i]] = 1.0;
			vec[i] = t;
		}
	}

	template <
		typename initial_vector>
		inline void toTorchTensors(initial_vector& vec, std::vector<torch::Tensor>& tensor_vect)
	{
		tensor_vect.resize(vec.size());
#pragma omp parallel for
		for (int i = 0; i < vec.size(); i++)
		{
			//torch::Tensor& tensor = torch::tensor({ vec[i] });	1.3
			torch::Tensor& tensor = torch::tensor(vec[i]);
			tensor_vect[i] = tensor;
		}
	}
	template <
		typename initial_vector>
		inline torch::Tensor toTorchTensors(initial_vector& vec)
	{
		//return torch::tensor({ vec });	 1.3
		return torch::tensor( vec );
	}

	inline std::vector<tiny_dnn::tensor_t> toTensor_t(torch::Tensor& x, int batch, int channel, int h, int w)
	{
		std::vector<tiny_dnn::tensor_t> y;
		const int size = channel * h*w;
		torch::Tensor& xx = x.view({ batch, 1,1, size });

		y.resize(batch);
#pragma omp parallel for
		for (int i = 0; i < batch; i++)
		{
			tiny_dnn::tensor_t t;
			tiny_dnn::vec_t v(size);
			for (int j = 0; j < size; j++)
			{
				v[j] = xx[i][0][0][j].template item<float_t>();
			}
			t.push_back(v);
			y[i] = t;
		}
		return y;
	}
	inline tiny_dnn::tensor_t toTensor_t(torch::Tensor& x, int channel, int h, int w)
	{
		const int size = channel * h*w;
		torch::Tensor& xx = x.view({ 1, 1,1, size });

		tiny_dnn::tensor_t t;
		tiny_dnn::vec_t v(size);
#if 0
#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{
			v[j] = xx[0][0][0][j].template item<float_t>();
		}
#else
		const float* p = xx.cpu().data<float>();
		v.assign(p, p + size);
#endif
		t.push_back(v);
		return t;
	}
	inline tiny_dnn::vec_t toTensor_t(torch::Tensor& x, int size)
	{
		torch::Tensor& xx = x.view({ 1, 1,1, size });

		tiny_dnn::vec_t v(size);
#if 0
#pragma omp parallel for
		for (int j = 0; j < size; j++)
		{
			v[j] = xx[0][0][0][j].template item<float_t>();
		}
#else
		const float* p = xx.cpu().data<float>();
		v.assign(p, p + size);
#endif
		return v;
	}

	inline int get_BATCH(const std::vector<torch::Tensor>& images, torch::Tensor& batch_images, const int batchSize, std::vector<int>& index)
	{
		int batchNum = images.size() / batchSize;
		if (batchNum == 0)
		{
			throw error_exception("input size < batch size");
		}

		batch_images = images[index[0]];
		for (int i = 1; i < index.size(); i++)
		{
			batch_images = torch::cat({ batch_images, images[index[i]] }, 0);
		}
		return batchNum;
	}

	void TensorToImageFile(torch::Tensor image_tensor, const std::string& filename, const int scale = 1.0)
	{
#ifdef USE_IMAGE_UTIL
		const int channels = image_tensor.sizes()[0];
		const int h = image_tensor.sizes()[1];
		const int w = image_tensor.sizes()[2];

		if (channels == 0 || channels > 3)
		{
			dump_dim("image_tensor", image_tensor);
			throw error_exception("tensor dimension != CxHxW");
		}
		tiny_dnn::tensor_t& img = toTensor_t(image_tensor.view({ channels, h,w }), channels, h, w);

		const int sz = img[0].size();
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			img[0][i] *= scale;
		}
		Image rgb_img = vec_t2image(img[0], channels, h, w);

		ImageWrite(filename.c_str(), &rgb_img);
#else
		throw error_exception("undefined USE_IMAGE_UTIL");
#endif
	}

	inline int get_BATCH(const std::vector<torch::Tensor>& images, const std::vector<torch::Tensor>& labels, torch::Tensor& batch_images, torch::Tensor& batch_labels, const int batchSize, std::vector<int>& index)
	{
		int batchNum = images.size() / batchSize;
		if (batchNum == 0)
		{
			throw error_exception("input size < batch size");
		}

		batch_images = images[index[0]];
		batch_labels = labels[index[0]];
		for (int i = 1; i < index.size(); i++)
		{
			batch_images = torch::cat({ batch_images, images[index[i]] }, 0);
			batch_labels = torch::cat({ batch_labels, labels[index[i]] }, 0);
		}
		return batchNum;
	}

	template <
		typename Model>
		class network_torch
	{
		std::vector<float_t> Tolerance_Set;
		float loss_value = 0.0;
		float clip_grad_value = 0.0;
	public:
		int in_channels = 1;
		int in_H = 1;
		int in_W = 1;
		int out_channels = 1;
		int out_H = 1;
		int out_W = 1;
		tiny_dnn::timer time_measurement;

		torch::Device device;
		Model model;

		/**
		 * @param model_             model of neural networks
		 * @param optimizer_         optimizing algorithm for training
		 * @param device_            Device Type(kCPU, kCUDA)
		 * assume
		 */
		network_torch(Model& model_, torch::Device device_)
			:model(model_), device(device_)
		{
			try
			{
				model.get()->to(device);
			}
			catch (std::exception& e)
			{
				printf("%s\n", e.what());
				exit(0);
			}
		}

		inline void set_clip_grad_norm(float v)
		{
			clip_grad_value = v;
		}
		inline float get_clip_grad_norm()
		{
			return clip_grad_value;
		}
		inline void input_dim(int c, int w, int h)
		{
			in_channels = c;
			in_H = h;
			in_W = w;
		}
		inline void output_dim(int c, int w, int h)
		{
			out_channels = c;
			out_H = h;
			out_W = w;
		}

		bool classification = false;
		bool batch_shuffle = true;
		bool pre_make_batch = true;

		bool stop_training_ = false;
		/**
		 * request to finish an ongoing training
		 *
		 * It is safe to test the current network performance in @a
		 * on_batch_enumerate
		 * and
		 * @a on_epoch_enumerate callbacks during training.
		 */
		inline void stop_ongoing_training() { stop_training_ = true; }

		/**
		* @param images             array of input data
		* @param batch_x            input data batch
		* assume
		*/
		inline void generate_BATCH(
			std::vector<torch::Tensor> &images,
			std::vector< torch::Tensor>& batch_x
		)
		{
			bool shuffle = batch_shuffle;
			const int batchNum = (int64_t)((float)images.size() / (float)kTrainBatchSize + 0.5);

			batch_x = std::vector< torch::Tensor>(batchNum);

			std::random_device rnd;
			std::mt19937 mt(rnd());
			std::uniform_int_distribution<> rand_index(0, (int)images.size() - 1);

#pragma omp parallel for
			for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
			{
				std::vector<int> index(kTrainBatchSize);
				if (shuffle)
				{
					for (int k = 0; k < kTrainBatchSize; k++)
					{
						index[k] = rand_index(mt);
					}
				}
				else
				{
					for (int k = 0; k < kTrainBatchSize; k++)
					{
						index[k] = (batch_idx*kTrainBatchSize + k) % images.size();
					}
				}
				get_BATCH(images, batch_x[batch_idx], kTrainBatchSize, index);

				if (tensor_flatten_size(batch_x[batch_idx]) < kTrainBatchSize*in_channels*in_H*in_W)
				{
					dump_dim("batch_x", batch_x[batch_idx]);
					std::cout << tensor_flatten_size(batch_x[batch_idx])
						<< " < " << kTrainBatchSize << "*" << in_channels << "*"
						<< in_H << "* " << in_W << "="
						<< kTrainBatchSize*in_channels*in_H*in_W << std::endl;
					throw error_exception("tensor size error.");
				}
				batch_x[batch_idx] = batch_x[batch_idx].view({ kTrainBatchSize, in_channels, in_H, in_W });
			}
		}


		/**
		* @param images             array of input data
		* @param labels             array of labels output
		* @param batch_x            input data batch
		* @param batch_y            output  data batch
		* assume
		*/
		inline void generate_BATCH(
			std::vector<torch::Tensor> &images,
			std::vector<torch::Tensor> &labels,
			std::vector< torch::Tensor>& batch_x,
			std::vector< torch::Tensor>& batch_y
		)
		{
			bool shuffle = batch_shuffle;

			int batch_tmp = kTrainBatchSize;
			if (batch_tmp > images.size())
			{
				batch_tmp = images.size();
			}

			//printf("%d -> lost:%d\n", images.size(), images.size() % kTrainBatchSize);
			//for (int i = batch_tmp; i >= 2; i--)
			//{
			//	if (images.size() % i == 0)
			//	{
			//		batch_tmp = i;
			//		break;
			//	}
			//}
			//printf("Please change:kTrainBatchSize:%d -> %d\n", kTrainBatchSize, batch_tmp);

			const int batchNum = (int64_t)((float)images.size() / (float)kTrainBatchSize + 0.5);

			batch_x = std::vector< torch::Tensor>(batchNum);
			batch_y = std::vector< torch::Tensor>(batchNum);

			std::random_device rnd;
			std::mt19937 mt(rnd());
			std::uniform_int_distribution<> rand_index(0, (int)images.size() - 1);

#pragma omp parallel for
			for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
			{
				std::vector<int> index(kTrainBatchSize);
				if (shuffle)
				{
					for (int k = 0; k < kTrainBatchSize; k++)
					{
						index[k] = rand_index(mt);
					}
				}
				else
				{
					for (int k = 0; k < kTrainBatchSize; k++)
					{
						index[k] = (batch_idx*kTrainBatchSize + k) % images.size();
					}
				}
				get_BATCH(images, labels, batch_x[batch_idx], batch_y[batch_idx], kTrainBatchSize, index);

				if (tensor_flatten_size(batch_x[batch_idx]) < kTrainBatchSize*in_channels*in_H*in_W)
				{
					dump_dim("batch_x", batch_x[batch_idx]);
					std::cout << tensor_flatten_size(batch_x[batch_idx])
						<< " < " << kTrainBatchSize << "*" << in_channels << "*" 
						<< in_H << "* " << in_W << "=" 
						<<kTrainBatchSize*in_channels*in_H*in_W << std::endl;
					throw error_exception("tensor size error.");
				}
				if (tensor_flatten_size(batch_y[batch_idx]) < kTrainBatchSize*out_channels*out_H*out_W)
				{
					dump_dim("batch_y", batch_y[batch_idx]);
					std::cout << tensor_flatten_size(batch_y[batch_idx])
						<< " < " << kTrainBatchSize << "*" << out_channels << "*"
						<< out_H << "* " << out_W << "="
						<< kTrainBatchSize*out_channels*out_H*out_W << std::endl;
					throw error_exception("tensor size error.");
				}
				batch_x[batch_idx] = batch_x[batch_idx].view({ kTrainBatchSize, in_channels, in_H, in_W });
				batch_y[batch_idx] = batch_y[batch_idx].view({ kTrainBatchSize, out_channels, out_H, out_W });
			}
		}

		/**
		 * @param optimizer          optimizing algorithm for training
		 * @param images             array of input data
		 * @param labels             array of labels output
		 * @param kTrainBatchSize    number of mini-batch
		 * @param kNumberOfEpochs    number of training epochs
		 * @param on_batch_enumerate callback for each mini-batch enumerate
		 * @param on_epoch_enumerate callback for each epoch
		 * assume
		 */
		bool fit(
			torch::optim::Optimizer* optimizer,
			std::vector<torch::Tensor> &images,
			std::vector<torch::Tensor> &labels,
			int kTrainBatchSize,
			int kNumberOfEpochs,
			std::function <void(void)> on_batch_enumerate = {},
			std::function <void(void)> on_epoch_enumerate = {}
		)
		{
			if (images.size() != labels.size()) {
				return false;
			}
			if (images.size() < kTrainBatchSize || labels.size() < kTrainBatchSize) {
				return false;
			}

			time_measurement.start();

			int batchNum;

			std::vector< torch::Tensor> batch_x;
			std::vector< torch::Tensor> batch_y;

			if (pre_make_batch)
			{
				generate_BATCH(images, labels, batch_x, batch_y);
				batchNum = batch_x.size();

				for (int i = 0; i < batchNum; i++)
				{
					batch_x[i] = batch_x[i].to(device);
					batch_y[i] = batch_y[i].to(device);
				}
			}
			
			std::vector<int> batch_idx_list;
			for (int i = 0; i < batchNum; i++)
			{
				batch_idx_list.push_back(i);
			}
			std::mt19937 get_rand_mt;

			optimizer->zero_grad();
			stop_training_ = false;
			model.get()->train(true);
			for (size_t epoch = 0; epoch < kNumberOfEpochs && !stop_training_; ++epoch)
			{
				if (!pre_make_batch)
				{
					generate_BATCH(images, labels, batch_x, batch_y);
					batchNum = batch_x.size();
					for (int i = 0; i < batchNum; i++)
					{
						batch_x[i] = batch_x[i].to(device);
						batch_y[i] = batch_y[i].to(device);
					}
				}
				if (this->batch_shuffle)
				{
					std::shuffle(batch_idx_list.begin(), batch_idx_list.end(), get_rand_mt);
				}
				loss_value = 0.0;

				float loss_ave = 0.0;
				for (int b_idx = 0; b_idx < batchNum && !stop_training_; b_idx++)
				{
					const int batch_idx = batch_idx_list[b_idx];

					torch::Tensor& data = batch_x[batch_idx];
					torch::Tensor& targets = batch_y[batch_idx];

					//data = data.to(device);
					//targets = targets.to(device);

					optimizer->zero_grad();
					auto output = model.get()->forward(data);
					//dump_dim("output", output);
					//dump_dim("targets", targets);

					targets = targets.reshape_as(output);

					torch::Tensor loss;
					if (classification)
					{
						loss = torch::nll_loss(output, targets.argmax(1));
					}
					else
					{
						loss = torch::mse_loss(output, targets);
					}
					if (std::isnan(loss.template item<float_t>()))
					{
						std::cout << "loss value is nan" << std::endl;
					}
					AT_ASSERT(!std::isnan(loss.template item<float_t>()));

					loss.backward();

					if ( fabs(clip_grad_value) > 0.0)
					{
						torch::nn::utils::clip_grad_norm_(model->parameters(), clip_grad_value);
					}
					optimizer->step();

					loss_value = loss.template item<float_t>();
					loss_ave += loss_value;
					on_batch_enumerate();
					model.get()->train(true);
				}

				if (stop_training_) break;
				loss_value = loss_ave / kTrainBatchSize;
				on_epoch_enumerate();
				model.get()->train(true);
			}
			time_measurement.stop();
			return true;
		}

		/**
		 * @param optimizer          optimizing algorithm for training
		 * @param images             array of input data
		 * @param labels             array of labels output
		 * @param kTrainBatchSize    number of mini-batch
		 * @param kNumberOfEpochs    number of training epochs
		 * @param on_batch_enumerate callback for each mini-batch enumerate
		 * @param on_epoch_enumerate callback for each epoch
		 * assume
		 */
		bool fit(
			torch::optim::Optimizer* optimizer,
			tiny_dnn::tensor_t &images,
			tiny_dnn::tensor_t &labels,
			int kTrainBatchSize,
			int kNumberOfEpochs,
			std::function <void(void)> on_batch_enumerate = {},
			std::function <void(void)> on_epoch_enumerate = {}
		)
		{
			std::vector<torch::Tensor> images_torch;
			std::vector<torch::Tensor> labels_torch;
			toTorchTensors(images, images_torch);
			toTorchTensors(labels, labels_torch);

			return fit(optimizer, images_torch, labels_torch, kTrainBatchSize, kNumberOfEpochs, on_batch_enumerate, on_epoch_enumerate);
		}

		/**
		 * @param optimizer          optimizing algorithm for training
		 * @param images             array of input data
		 * @param class_labels       array of label-id for each input data(0-origin) label-id=on-hot-vector
		 * @param kTrainBatchSize    number of mini batch
		 * @param kNumberOfEpochs    number of training epochs
		 * @param on_batch_enumerate callback for each mini-batch enumerate
		 * @param on_epoch_enumerate callback for each epoch
		 * assume
		 */
		bool train(
			torch::optim::Optimizer* optimizer,
			tiny_dnn::tensor_t &images,
			std::vector<tiny_dnn::label_t> &class_labels,
			int kTrainBatchSize,
			int kNumberOfEpochs,
			std::function <void(void)> on_batch_enumerate = {},
			std::function <void(void)> on_epoch_enumerate = {}
		)
		{
			std::vector<tiny_dnn::vec_t> one_hot_vec;
			label2vec(class_labels, one_hot_vec);

			std::vector<torch::Tensor> images_torch;
			std::vector<torch::Tensor> labels_torch;
			toTorchTensors(images, images_torch);
			toTorchTensors(one_hot_vec, labels_torch);

			return fit(optimizer, images_torch, labels_torch, kTrainBatchSize, kNumberOfEpochs, on_batch_enumerate, on_epoch_enumerate);
		}

		bool test(
			std::vector<torch::Tensor> &images,
			std::vector<torch::Tensor> &labels,
			int kTestBatchSize
		)
		{
			if (images.size() != labels.size()) {
				return false;
			}
			if (images.size() < kTestBatchSize || labels.size() < kTestBatchSize) {
				return false;
			}

			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);
			float loss_ave = 0.0;
			int correct = 0;

			//int batch_tmp = kTestBatchSize;
			//if (batch_tmp > images.size())
			//{
			//	batch_tmp = images.size();
			//}

			//printf("%d -> lost:%d\n", images.size(), images.size() % kTestBatchSize);
			//for (int i = batch_tmp; i >= 2; i--)
			//{
			//	if (images.size() % i == 0) 
			//	{
			//		batch_tmp = i;
			//		break;
			//	}
			//}
			//printf("Please change:kTestBatchSize:%d -> %d\n", kTestBatchSize, batch_tmp);

			int testNum = images.size() / kTestBatchSize;
			if (testNum == 0)
			{
				throw error_exception("input size < test batch size");
			}
			for (size_t test = 0; test < testNum; ++test)
			{
				torch::Tensor batch_x;
				torch::Tensor batch_y;
				std::vector<int> index(kTestBatchSize);
				for (int k = 0; k < kTestBatchSize; k++)
				{
					index[k] = (kTestBatchSize * test + k) % images.size();
				}
				get_BATCH(images, labels, batch_x, batch_y, kTestBatchSize, index);

				torch::Tensor& data = batch_x.view({ kTestBatchSize,in_channels, in_H, in_W });
				torch::Tensor& targets = batch_y.view({ kTestBatchSize, out_channels, out_H, out_W });

				data = data.to(device);
				targets = targets.to(device);
				torch::Tensor output = model.get()->forward(data);
				targets = targets.reshape_as(output);

				torch::Tensor loss;
				if (classification)
				{
					loss = torch::nll_loss(output, targets.argmax(1));
				}
				else
				{
					loss = torch::mse_loss(output, targets);
				}
				AT_ASSERT(!std::isnan(loss.template item<float_t>()));

				loss_value = loss.template item<float_t>();
				loss_ave += loss_value;

				if (classification)
				{
					auto pred = output.argmax(1);
					correct += pred.eq(targets.argmax(1)).sum().template item<int64_t>();
				}
//				if (classification_one_hot_vector)
//				{
//#if 1
//					auto pred = output.argmax(1);
//					correct += pred.eq(targets.argmax(1)).sum().template item<int64_t>();
//#else
//#pragma omp parallel for
//					for (int k = 0; k < kTestBatchSize; k++)
//					{
//						correct += (vec_max_index(output[k]) == vec_max_index(targets[k])) ? 1 : 0;
//
//						{
//							//std::vector<tiny_dnn::tensor_t>& x =	toTensor_t(output[k], 1, 1, 1, out_data_size());
//							//std::vector<tiny_dnn::tensor_t>& y = toTensor_t(targets[k], 1, 1, 1, out_data_size());
//							//AT_ASSERT(vec_max_index(x[0][0])== vec_max_index(output[k]));
//							//AT_ASSERT(vec_max_index(y[0][0]) == vec_max_index(targets[k]));
//
//							//tiny_dnn::tensor_t& x = toTensor_t(output[k], 1, 1, out_data_size());
//							//tiny_dnn::tensor_t& y = toTensor_t(targets[k], 1, 1, out_data_size());
//							//AT_ASSERT(vec_max_index(x[0]) == vec_max_index(output[k]));
//							//AT_ASSERT(vec_max_index(y[0]) == vec_max_index(targets[k]));
//
//							//tiny_dnn::vec_t& x = toTensor_t(output[k], out_data_size());
//							//tiny_dnn::vec_t& y = toTensor_t(targets[k], out_data_size());
//							//AT_ASSERT(vec_max_index(x) == vec_max_index(output[k]));
//							//AT_ASSERT(vec_max_index(y) == vec_max_index(targets[k]));
//						}
//					}
//#endif
//				}
			}

			if (classification)
			{
				std::printf(" Accuracy: %.3f%% Loss: %.3f\n", 100.0*static_cast<float_t>(correct) / images.size(), loss_ave / testNum);
			}
			else
			{
				std::printf("Loss: %.3f\n", loss_ave / testNum);
			}
			return true;
		}

		/**
		* @param images             array of input data
		* @param labels				array of output data
		* @param kTestBatchSize     number of mini batch
		* assume
		*/
		bool test(
			tiny_dnn::tensor_t &images,
			tiny_dnn::tensor_t &labels,
			int kTestBatchSize
		)
		{
			std::vector<torch::Tensor> images_torch;
			std::vector<torch::Tensor> labels_torch;
			toTorchTensors(images, images_torch);
			toTorchTensors(labels, labels_torch);

			return test(images_torch, labels_torch, kTestBatchSize);
		}

		/**
		* @param images             array of input data
		* @param labels				array of lable(on-hot-vector) data
		* @param kTestBatchSize     number of mini batch
		* assume
		*/
		bool test(
			tiny_dnn::tensor_t &images,
			std::vector<tiny_dnn::label_t> &class_labels,
			int kTestBatchSize
		)
		{
			std::vector<tiny_dnn::vec_t> one_hot_vec;
			label2vec(class_labels, one_hot_vec);

			std::vector<torch::Tensor> images_torch;
			std::vector<torch::Tensor> labels_torch;
			toTorchTensors(images, images_torch);
			toTorchTensors(one_hot_vec, labels_torch);

			return test(images_torch, labels_torch, kTestBatchSize);
		}


		torch::Tensor fprop(torch::Tensor &in) {
			torch::NoGradGuard no_grad;
			model->eval();
			model.get()->train(false);
			return model.get()->forward(in);
		}

		/**
		 * executes forward-propagation and returns output
		 **/
		inline torch::Tensor predict(torch::Tensor& X)
		{
			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);
			torch::Tensor y =  model.get()->forward(X.to(device));

			return y;
		}
		/**
		 * executes forward-propagation and returns output
		 **/
		inline std::vector<tiny_dnn::tensor_t> predict(torch::Tensor& X, const int batch)
		{
			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);
			torch::Tensor y = model.get()->forward(X.to(device));

			std::vector<tiny_dnn::tensor_t> t;
			toTensor_t(y, t, batch, out_channels, out_H, out_W);
			return t;
		}
		/**
		* executes forward-propagation and returns output
		**/
		inline std::vector<tiny_dnn::vec_t> predict(std::vector<tiny_dnn::vec_t>& X, int batch = 1)
		{
			//printf("X.size()=%d\n", X.size()); fflush(stdout);
			std::vector<tiny_dnn::vec_t> out;

			int batch_n = X.size() / batch;

			if (X.size() < batch || batch == 1)
			{
				for (int i = 0; i < X.size(); i++)
				{
					out.emplace_back(predict(X[i]));
				}

				return out;
			}

			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);

			std::vector<torch::Tensor> n_batch_images(batch_n);
#pragma omp parallel for
			for (int i = 0; i < batch_n; i++)
			{
				torch::Tensor images_torch = toTorchTensors(X[i*batch]).view({ 1, in_channels, in_H, in_W }).to(device);
				auto batch_images = images_torch;
				for (int j = 1; j < batch; j++)
				{
					torch::Tensor images_torch = toTorchTensors(X[i*batch + j]).view({ 1, in_channels, in_H, in_W }).to(device);
					batch_images = torch::cat({ batch_images, images_torch }, 0);
				}
				n_batch_images[i] = batch_images;
			}

			for (int k = 0; k < batch_n; k++)
			{
				//cpp_torch::dump_dim("batch_images", batch_images);
				torch::Tensor y = model.get()->forward(n_batch_images[k]);
				y = y.view({ batch, out_channels, out_H, out_W });

				for (int i = 0; i < batch; i++)
				{
					//cpp_torch::dump_dim("torch::Tensor y", y);
					//std::cout << " " << out_data_size() << std::endl;
					tiny_dnn::vec_t& t = toTensor_t(y[i], out_data_size());

					out.emplace_back(t);
				}
			}

			int n = X.size() % batch;
			//printf("n=%d\n", n); fflush(stdout);

			if (n > 0 && X.size() > batch)
			{
				for (int i = X.size() - n; i < X.size(); i++)
				{
					out.emplace_back(predict(X[i]));
				}
			}
			//printf("out.size()=%d\n", out.size()); fflush(stdout);
			return out;
		}

		/**
		* executes forward-propagation and returns output
		**/
		inline tiny_dnn::vec_t predict(tiny_dnn::vec_t& X)
		{
			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);
			torch::Tensor images_torch = toTorchTensors(X).view({ 1, in_channels, in_H, in_W }).to(device);

			torch::Tensor y = model.get()->forward(images_torch);

			//cpp_torch::dump_dim("torch::Tensor y", y);
			//std::cout << " " << out_data_size() << std::endl;
			tiny_dnn::vec_t t = toTensor_t(y, out_data_size());
			return t;
		}
		/**
		* executes forward-propagation and returns output
		**/
		inline tiny_dnn::label_t predict_label(tiny_dnn::vec_t& X)
		{
			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);
			torch::Tensor images_torch = toTorchTensors(X).view({ 1, in_channels, in_H, in_W }).to(device);

			torch::Tensor y = model.get()->forward(images_torch);
			tiny_dnn::vec_t t = toTensor_t(y, out_data_size());
			return vec_max_index(t);
		}

		inline void label2vec(const std::vector<tiny_dnn::label_t>& labels, std::vector<tiny_dnn::vec_t>& vec)
		{
			const size_t outdim = out_data_size();
			vec.clear();
			vec.resize(labels.size());

			const size_t sz = labels.size();
#pragma omp parallel for
			for (int i = 0; i < sz; i++)
			{
				tiny_dnn::vec_t t(outdim, 0);
				t[labels[i]] = 1.0;
				vec[i] = t;
			}
		}

		inline void label2vec(const tiny_dnn::label_t& labels, tiny_dnn::vec_t& vec)
		{
			const size_t outdim = out_data_size();
			tiny_dnn::vec_t t(outdim, 0);
			t[labels] = 1.0;
			vec = t;
		}

		inline tiny_dnn::label_t vec_max_index(torch::Tensor &out) {
			return tiny_dnn::label_t(out.view({ out_data_size() }).argmax(0).template item<float_t>());
		}

		inline tiny_dnn::label_t vec_max_index(tiny_dnn::vec_t &out) {
			return tiny_dnn::label_t(max_index(out));
		}
		inline tiny_dnn::label_t vec_max_index(tiny_dnn::tensor_t &out) {
			return tiny_dnn::label_t(max_index(out[0]));
		}

		float_t get_loss(std::vector<tiny_dnn::vec_t> &in, std::vector<tiny_dnn::label_t> &t, int batchSize) {
			std::vector<tiny_dnn::vec_t> vec;
			label2vec(t, vec);

			return get_loss(in, vec, batchSize);
		}

		float_t get_loss( std::vector<tiny_dnn::vec_t> &in, std::vector<tiny_dnn::vec_t> &t, int BatchSize) {
			float_t sum_loss = float_t(0);

			std::vector<torch::Tensor> images;
			std::vector<torch::Tensor> labels;

			//printf("in:%d\n", in.size());
			toTorchTensors(in, images);
			toTorchTensors(t, labels);

			//int batch_tmp = BatchSize;
			//if (batch_tmp > in.size())
			//{
			//	batch_tmp = in.size();
			//}

			//printf("lost:%d\n", in.size() % BatchSize);
			//for (int i = batch_tmp; i >= 2; i--)
			//{
			//	if (images.size() % i == 0)
			//	{
			//		batch_tmp = i;
			//		break;
			//	}
			//}
			//printf("change:BatchSize:%d -> %d\n", BatchSize, batch_tmp);
			//BatchSize = batch_tmp;

			const int batchNum = (const int)((float)in.size() / BatchSize +0.5);
			if (batchNum == 0)
			{
				printf("input size:%d BatchSize:%d\n", in.size(), BatchSize);
				throw error_exception("input size < Batch Size");
			}

			std::vector< torch::Tensor> batch_x(batchNum);
			std::vector< torch::Tensor> batch_y(batchNum);

			//torch::NoGradGuard no_grad;
			//model->eval();
			model.get()->train(false);

			std::vector<float_t> loss_list(in.size(), 0.0);
//#pragma omp parallel for
			for (int i = 0; i < batchNum; i++) {

				//if (!pre_make_batch)
				{
					std::vector<int> index(BatchSize);
					for (int k = 0; k < BatchSize; k++)
					{
						index[k] = (i* BatchSize + k) % images.size();
					}
					get_BATCH(images, labels, batch_x[i], batch_y[i], BatchSize, index);

					batch_x[i] = batch_x[i].view({ BatchSize, in_channels, in_H, in_W });
					batch_y[i] = batch_y[i].view({ BatchSize, out_channels, out_H, out_W });
				}

				torch::Tensor& input = batch_x[i].to(device);
				torch::Tensor& targets = batch_y[i].to(device);

				torch::Tensor predicted = predict(input).to(device);


				//dump_dim(std::string("predicted"), predicted);
				//dump_dim(std::string("targets"), targets);

				torch::Tensor loss;
				if (classification)
				{
					loss = torch::nll_loss(predicted, targets.view_as(predicted).argmax(1));
				}
				else
				{
					loss = torch::mse_loss(predicted.view_as(targets), targets);
				}
				AT_ASSERT(!std::isnan(loss.template item<float_t>()));
				//dump_dim(std::string("loss"), loss);

				//std::cout << loss << std::endl;
				loss_list[i] = loss.template item<float_t>();
			}

			for (size_t i = 0; i < batchNum; i++) {
				sum_loss += loss_list[i];
			}
			return sum_loss/ BatchSize;
		}

		void set_tolerance(const float max_tol, const float min_tol, int div = 5)
		{
			if (div < 3) div = 3;
			Tolerance_Set.resize(div);

			for (int i = 0; i < div; i++)
			{
				Tolerance_Set[i] = (max_tol + i*(min_tol - max_tol) / (div - 1.0));
			}
		}
		std::vector<float_t>& get_tolerance()
		{
			return Tolerance_Set;
		}

		/*
		 * output vector  output[0..tolerance_set.size()-1]=num_success, output[tolerance_set.size()]=image of size, 
		 */
		std::vector<int> get_accuracy(tiny_dnn::tensor_t& images, tiny_dnn::tensor_t& labels, std::vector<float_t>& tolerance_set)
		{
			std::vector<int> result(tolerance_set.size()+1);

			if (images.size() == 0)
			{
				return result;
			}

			result[tolerance_set.size()] = images.size();
			for (int i = 0; i < images.size(); i++)
			{
				tiny_dnn::vec_t& predict_y = predict(images[i]);
				const tiny_dnn::vec_t& actual = labels[i];

				AT_ASSERT(predict_y.size() == actual.size());

				float sum = 0.0;
				for (int k = 0; k < predict_y.size(); k++)
				{
					sum += (predict_y[k] - actual[k])*(predict_y[k] - actual[k]);
				}
				sum /= predict_y.size();

				for (int j = 0; j < tolerance_set.size(); j++)
				{
					if (sum < tolerance_set[j])
					{
						result[j]++;
					}
				}
			}
			return result;
		}

		tiny_dnn::result  get_accuracy( tiny_dnn::tensor_t& images, tiny_dnn::tensor_t& labels, int batch = 1)
		{
			tiny_dnn::result result;

			if (images.size() == 0)
			{
				result.num_total = 1;
				return result;
			}

			const size_t sz = images.size();
#if 10
			std::vector< tiny_dnn::label_t> predicted_list(sz, 0);
			std::vector< tiny_dnn::label_t>actual_list(sz, 0);

			std::vector<tiny_dnn::vec_t>& n_predict_y = predict(images, batch);
#pragma omp parallel for
			for (int i = 0; i < sz; i++)
			{
				//tiny_dnn::vec_t& predict_y = predict(images[i]);
				tiny_dnn::vec_t& predict_y = n_predict_y[i];
				predicted_list[i] = vec_max_index(predict_y);
				actual_list[i] = vec_max_index(labels[i]);
			}
			for (int i = 0; i < sz; i++)
			{
				if (predicted_list[i] == actual_list[i]) result.num_success++;
				result.num_total++;
				result.confusion_matrix[predicted_list[i]][actual_list[i]]++;
			}

#else
			for (int i = 0; i < sz; i++)
			{
				tiny_dnn::vec_t& predict_y = predict(images[i]);
				const tiny_dnn::label_t predicted = vec_max_index(predict_y);
				const tiny_dnn::label_t actual = vec_max_index(labels[i]);

				if (predicted == actual) result.num_success++;
				result.num_total++;
				result.confusion_matrix[predicted][actual]++;
			}
#endif
			return result;
		}

		std::vector<int> test_tolerance(tiny_dnn::tensor_t& images, tiny_dnn::tensor_t& labels)
		{
			AT_ASSERT(Tolerance_Set.size() != 0);
			return get_accuracy(images, labels, Tolerance_Set);
		}

		// labels[#] = 0,1,..class-1 
		tiny_dnn::result  get_accuracy(tiny_dnn::tensor_t& images, std::vector <tiny_dnn::label_t>& labels)
		{
			std::vector<tiny_dnn::vec_t> vec;
			label2vec(labels, vec);	//on-hot-vector
			return get_accuracy(images, vec);
		}
		tiny_dnn::result  test(tiny_dnn::tensor_t& images, tiny_dnn::tensor_t& labels)
		{
			return get_accuracy(images, labels);
		}
		// labels[#] = on-hot-vector 
		tiny_dnn::result  test(tiny_dnn::tensor_t& images, std::vector <tiny_dnn::label_t>& labels)
		{
			return get_accuracy(images, labels);
		}

		inline int in_data_size() const
		{
			return in_channels * in_H*in_W;
		}
		inline int out_data_size() const
		{
			return out_channels * out_H*out_W;
		}
		inline void save(std::string& filename)
		{
			torch::save(model, filename);
		}
		inline void load(std::string& filename)
		{
			try
			{
				//CUDA information is also included in the data that has been learned and serialized by CUDA.
				// map_location = device 
				torch::load(model, filename/*, device*/);
				model.get()->to(device);
			}
			catch (c10::Error& err)
			{
				printf("load error[%s]\n", err.what());
			}
		}
	};


	void print_ConfusionMatrix(tiny_dnn::result& res)
	{
		//ConfusionMatrix
		std::cout << "ConfusionMatrix:" << std::endl;
		res.print_detail(std::cout);
		std::cout << res.num_success << "/" << res.num_total << std::endl;
		res.print_summary(std::cout);
		//printf("accuracy:%.3f%%\n", res.accuracy());
	}
	void print_ConfusionMatrix(std::vector<int>& res, std::vector<float_t>& tol)
	{
		//ConfusionMatrix
		std::cout << "ConfusionMatrix:" << std::endl;
		for (int i = 0; i < res.size()-1; i++)
		{
			printf("tolerance:%.4f %d / %d accuracy:%.3f%%\n", tol[i], res[i], res.back(),
				100.0*(float_t)res[i] / (float_t)res.back());
		}
	}

}
#endif
