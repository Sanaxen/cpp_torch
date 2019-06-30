#ifndef libtorch_UTILS
#define libtorch_UTILS

#include <random>
#include "../torch_util/utils.h"

inline void nop() {
	// do nothing
}

void label2vec(const std::vector<tiny_dnn::label_t>& labels, std::vector<tiny_dnn::vec_t>& vec, int max_label)
{
	vec.clear();
	vec.resize(labels.size());
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
	for (int i = 0; i < vec.size(); i++)
	{
		torch::Tensor& tensor = torch::tensor({ vec[i] });
		tensor_vect[i] = tensor;
	}
}
template <
	typename initial_vector>
	inline torch::Tensor toTorchTensors(initial_vector& vec)
{
		return torch::tensor({ vec });
}

inline std::vector<tiny_dnn::tensor_t> toTensor_t(torch::Tensor& x, int batch, int channel, int w, int h)
{
	std::vector<tiny_dnn::tensor_t> y;
	const int size = channel * w*h;
	torch::Tensor& xx = x.view({ batch, 1,1, size});

	for (int i = 0; i < batch; i++)
	{
		tiny_dnn::tensor_t t;
		tiny_dnn::vec_t v(size);
		for (int j = 0; j < size; j++)
		{
			v[j] = xx[i][0][0][j].template item<float>();
		}
		t.push_back(v);
		y.push_back(t);
	}
	return y;
}
inline tiny_dnn::tensor_t toTensor_t(torch::Tensor& x, int channel, int w, int h)
{
	const int size = channel * w*h;
	torch::Tensor& xx = x.view({ 1, 1,1, size });

	tiny_dnn::tensor_t t;
	tiny_dnn::vec_t v(size);
	for (int j = 0; j < size; j++)
	{
		v[j] = xx[0][0][0][j].template item<float>();
	}
	t.push_back(v);
	return t;
}
inline tiny_dnn::vec_t toTensor_t(torch::Tensor& x, int size)
{
	torch::Tensor& xx = x.view({ 1, 1,1, size });

	tiny_dnn::vec_t v(size);
	for (int j = 0; j < size; j++)
	{
		v[j] = xx[0][0][0][j].template item<float>();
	}
	return v;
}

inline int get_BATCH(const std::vector<torch::Tensor>& images, const std::vector<torch::Tensor>& labels, torch::Tensor& batch_images, torch::Tensor& batch_labels, const int batchSize, std::vector<int>& index)
{
	int batchNum = images.size() / batchSize;

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
public:
	int in_channels = 1;
	int in_W = 1;
	int in_H = 1;
	int out_channels = 1;
	int out_W = 1;
	int out_H = 1;

	torch::Device device;
	Model model;
	torch::optim::Optimizer* optimizer;
	float loss_value = 0.0;
	tiny_dnn::timer time_measurement;

	/**
	 * @param model_             model of neural networks
	 * @param optimizer_         optimizing algorithm for training
	 * @param device_            Device Type(kCPU, kCUDA)
	 * assume
	 */
	network_torch(Model& model_, torch::optim::Optimizer* optimizer_, torch::Device device_)
		:model(model_), optimizer(optimizer_), device(device_)
	{
		model.get()->to(device);
	}

	inline void input_dim(int c, int w, int h)
	{
		in_channels = c;
		in_W = w;
		in_H = h;
	}
	inline void out_dim(int c, int w, int h)
	{
		out_channels = c;
		out_W = w;
		out_H = h;
	}

	bool classification = false;
	bool batch_shuffle = true;
	bool pre_make_batch = false;

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


	inline void dump_dim(torch::Tensor& t)
	{
		printf("output dim:%d ", t.dim());
		for (int i = 0; i < t.sizes().size()-1; i++)
		{
			printf("%d x", t.sizes()[i]);
		}
		printf("%d\n", t.sizes()[t.sizes().size() - 1]);
	}
	/**
	 * @param optimizer          optimizing algorithm for training
     * @param images             array of input data
     * @param labels             array of labels output
	 * @param batch_size         number of samples per parameter update
	 * @param epoch              number of training epochs
	 * @param on_batch_enumerate callback for each mini-batch enumerate
	 * @param on_epoch_enumerate callback for each epoch
	 * assume
	 */
	bool fit(
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
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::uniform_int_distribution<> rand_index(0, (int)images.size() - 1);

		bool shuffle = batch_shuffle;
		const int batchNum = images.size() / kTrainBatchSize; ;

		std::vector< torch::Tensor> batch_x(batchNum);
		std::vector< torch::Tensor> batch_y(batchNum);

		if (pre_make_batch)
		{
			for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
			{
				std::vector<int> index(kTrainBatchSize);
				for (int k = 0; k < kTrainBatchSize; k++)
				{
					index[k] = rand_index(mt);
				}
				get_BATCH(images, labels, batch_x[batch_idx], batch_y[batch_idx], kTrainBatchSize, index);

				batch_x[batch_idx] = batch_x[batch_idx].view({ kTrainBatchSize, in_channels, in_W, in_H });
				batch_y[batch_idx] = batch_y[batch_idx].view({ kTrainBatchSize, out_channels, out_W, out_H });
			}
		}

		for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
		{
			model.get()->train();
			
			if (stop_training_) break;

			loss_value = 0.0;

			float loss_ave = 0.0;
			for (int batch_idx = 0; batch_idx < batchNum; batch_idx++)
			{
				if (!pre_make_batch)
				{
					std::vector<int> index(kTrainBatchSize);
					for (int k = 0; k < kTrainBatchSize; k++)
					{
						index[k] = rand_index(mt);
					}
					get_BATCH(images, labels, batch_x[batch_idx], batch_y[batch_idx], kTrainBatchSize, index);

					batch_x[batch_idx] = batch_x[batch_idx].view({ kTrainBatchSize, in_channels, in_W, in_H });
					batch_y[batch_idx] = batch_y[batch_idx].view({ kTrainBatchSize, out_channels, out_W, out_H });
				}
				torch::Tensor& data = batch_x[batch_idx];
				torch::Tensor& targets = batch_y[batch_idx];
				
				data = data.to(device);
				targets = targets.to(device);

				optimizer->zero_grad();
				auto output = model.get()->forward(data);
				targets = targets.reshape_as(output);

				//auto loss = torch::nll_loss(output, targets);
				auto loss = torch::mse_loss(output, targets);
				AT_ASSERT(!std::isnan(loss.template item<float>()));
				loss.backward();
				optimizer->step();

				loss_value = loss.template item<float>();
				loss_ave += loss_value;
				on_batch_enumerate();
			}			
			
			if (stop_training_) break;
			loss_value = loss_ave / batchNum;
			on_epoch_enumerate();
		}
		time_measurement.stop();
		return true;
	}

	/**
	 * @param optimizer          optimizing algorithm for training
	 * @param images             array of input data
	 * @param batch_size         number of samples per parameter update
	 * @param epoch              number of training epochs
	 * @param on_batch_enumerate callback for each mini-batch enumerate
	 * @param on_epoch_enumerate callback for each epoch
	 * assume
	 */
	bool fit(
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

		return fit(images_torch, labels_torch, kTrainBatchSize, kNumberOfEpochs, on_batch_enumerate, on_epoch_enumerate);
	}

	/**
	 * @param optimizer          optimizing algorithm for training
	 * @param images             array of input data
     * @param class_labels       array of label-id for each input data(0-origin)
	 * @param batch_size         number of samples per parameter update
	 * @param epoch              number of training epochs
	 * @param on_batch_enumerate callback for each mini-batch enumerate
	 * @param on_epoch_enumerate callback for each epoch
	 * assume
	 */
	bool train(
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

		return fit(images_torch, labels_torch, kTrainBatchSize, kNumberOfEpochs, on_batch_enumerate, on_epoch_enumerate);
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

		model.get()->train(false);
		float loss_ave = 0.0;
		int correct = 0;
		int testNum = images.size() / kTestBatchSize;
		for (size_t test = 0; test < testNum; ++test)
		{
			torch::Tensor batch_x;
			torch::Tensor batch_y;
			std::vector<int> index(kTestBatchSize);
			for (int k = 0; k < kTestBatchSize; k++)
			{
				index[k] = kTestBatchSize * test + k;
			}
			get_BATCH(images, labels, batch_x, batch_y, kTestBatchSize, index);

			torch::Tensor& data = batch_x.view({ kTestBatchSize,in_channels, in_W, in_H });
			torch::Tensor& targets = batch_y.view({ kTestBatchSize, out_channels, out_W, out_H });

			data = data.to(device);
			targets = targets.to(device);
			torch::Tensor output = model.get()->forward(data);
			output = output.reshape_as(targets);

			//auto loss = torch::nll_loss(output, targets);
			auto loss = torch::mse_loss(output, targets);
			AT_ASSERT(!std::isnan(loss.template item<float>()));

			loss_value = loss.template item<float>();
			loss_ave += loss_value;

			//if (test == 0)
			//{
			//	for (int i = 0; i < out_H; i++)
			//	{
			//		printf("%.3f ", output[0][0][0][i].template item<float>());
			//	}
			//	printf("\n");
			//	std::vector<tiny_dnn::tensor_t>& t = predict(data, kTestBatchSize);
			//	for (int i = 0; i < out_H; i++)
			//	{
			//		printf("%.3f ", t[0][0][i]);
			//	}
			//	printf("\n");
			//}

			if (classification)
			{
#pragma omp parallel for
				for (int k = 0; k < kTestBatchSize; k++)
				{
					correct += (vec_max_index(output[k]) == vec_max_index(targets[k])) ? 1 : 0;

					{ 
						//std::vector<tiny_dnn::tensor_t>& x =	toTensor_t(output[k], 1, 1, 1, out_data_size());
						//std::vector<tiny_dnn::tensor_t>& y = toTensor_t(targets[k], 1, 1, 1, out_data_size());
						//AT_ASSERT(vec_max_index(x[0][0])== vec_max_index(output[k]));
						//AT_ASSERT(vec_max_index(y[0][0]) == vec_max_index(targets[k]));
						
						//tiny_dnn::tensor_t& x = toTensor_t(output[k], 1, 1, out_data_size());
						//tiny_dnn::tensor_t& y = toTensor_t(targets[k], 1, 1, out_data_size());
						//AT_ASSERT(vec_max_index(x[0]) == vec_max_index(output[k]));
						//AT_ASSERT(vec_max_index(y[0]) == vec_max_index(targets[k]));
						
						//tiny_dnn::vec_t& x = toTensor_t(output[k], out_data_size());
						//tiny_dnn::vec_t& y = toTensor_t(targets[k], out_data_size());
						//AT_ASSERT(vec_max_index(x) == vec_max_index(output[k]));
						//AT_ASSERT(vec_max_index(y) == vec_max_index(targets[k]));
					}
				}
				//targets = targets.argmax(3);
				//output = output.argmax(3);
				//for (int k = 0; k < kTestBatchSize; k++)
				//{
				//	correct += ((int)output[k].template item<float>() == (int)targets[k].template item<float>()) ? 1 : 0;
				//}
			}
		}
		model.get()->train(true);

		if (classification)
		{
			std::printf(" Accuracy: %.3f Loss: %.3f\n", static_cast<double>(correct) / images.size(), loss_ave / testNum);
		}
		else
		{
			std::printf("Loss: %.3f\n", loss_ave / testNum);
		}
		return true;
	}

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


	/**
	 * executes forward-propagation and returns output
	 **/
	inline torch::Tensor predict(torch::Tensor& X)
	{
		return model.get()->forward(X);
	}
	/**
	 * executes forward-propagation and returns output
	 **/
	inline std::vector<tiny_dnn::tensor_t> predict(torch::Tensor& X, const int batch)
	{
		torch::Tensor y = model.get()->forward(X);
		std::vector<tiny_dnn::tensor_t> t;
		toTensor_t(y, t, batch, out_channels, out_W, out_H);
		return t;
	}

	void label2vec(const std::vector<tiny_dnn::label_t>& labels, std::vector<tiny_dnn::vec_t>& vec)
	{
		size_t outdim = out_data_size();
		vec.clear();
		vec.resize(labels.size());
		for (int i = 0; i < labels.size(); i++)
		{
			tiny_dnn::vec_t t(outdim, 0);
			t[labels[i]] = 1.0;
			vec[i] = t;
		}
	}

	tiny_dnn::label_t vec_max_index(torch::Tensor &out) {
		return tiny_dnn::label_t(out.view({ out_data_size() }).argmax(0).template item<float>());
	}

	tiny_dnn::label_t vec_max_index(tiny_dnn::vec_t &out) {
		return tiny_dnn::label_t(max_index(out));
	}
	tiny_dnn::label_t vec_max_index(tiny_dnn::tensor_t &out) {
		return tiny_dnn::label_t(max_index(out[0]));
	}

	inline int out_data_size()
	{
		return out_channels * out_W*out_H;
	}
	inline void save(std::string& filename)
	{
		torch::save(model, filename);
	}
	inline void load(std::string& filename)
	{
		torch::load(model, filename);
	}
};

#endif
