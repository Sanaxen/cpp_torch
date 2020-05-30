/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"

#define _LIBRARY_EXPORTS
#include "tiny_dnn2libtorch_dnn_scriptdll.h"

#define TEST		//cpp_torch
#define USE_CUDA
#define RNN_LAYERS_OPT


namespace rnn_dll_variables
{
#ifdef USE_CUDA
	int use_gpu = 1;
#else
	int use_gpu = 0;
#endif

	// The batch size for training.
	int64_t kTrainBatchSize = 0;

	// The batch size for training.
	int64_t kTestBatchSize = 0;

	// The number of epochs to train.
	int64_t kNumberOfEpochs = 0;

	cpp_torch::network_torch<cpp_torch::Net>* nn_ = nullptr;
	torch::Device device(torch::kCPU);

	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	std::string optimizer_name = "adam";
	float scale = 1.0;
	float learning_rate = 0.0001;
	bool test_mode = 0;
	int num_class = 0;
	bool batch_shuffle = false;

};
using namespace rnn_dll_variables;

// load MNIST dataset
extern "C" _LIBRARY_EXPORTS void read_mnist_dataset(const std::string &data_dir_path)
{
	std::vector<tiny_dnn::label_t> tr_labels;
	std::vector<tiny_dnn::label_t> te_labels;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels-idx1-ubyte",
		&tr_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images-idx3-ubyte",
		&train_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels-idx1-ubyte",
		&te_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images-idx3-ubyte",
		&test_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	cpp_torch::label2vec(tr_labels, train_labels, 10);
	cpp_torch::label2vec(te_labels, test_labels, 10);
}

void state_reset(std::string& rnn_type , cpp_torch::Net& model)
{
	//if (rnn_type == "lstm")
	//{
	//	for (int i = 0; i < model.get()->lstm.size(); i++)
	//	{
	//		model.get()->lstm[i].get()->initial_state();
	//	}
	//}
	//if (rnn_type == "gru")
	//{
	//	for (int i = 0; i < model.get()->gru.size(); i++)
	//	{
	//		model.get()->gru[i].get()->reset();
	//	}
	//}
	//if (rnn_type == "rnn")
	//{
	//	for (int i = 0; i < model.get()->rnn.size(); i++)
	//	{
	//		model.get()->rnn[i].get()->reset();
	//	}
	//}
}


extern "C" _LIBRARY_EXPORTS void torch_getData(const char* filename, std::vector<tiny_dnn::vec_t>& data)
{
	CSVReader csv(filename, ',', false);
	data = csv.toTensor();
}
void scaling_data(std::vector<tiny_dnn::vec_t>& data, std::vector<tiny_dnn::vec_t>& dataset)
{
	data.clear();
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < dataset.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < dataset[0].size(); k++)
		{
			y.push_back(dataset[i][k]/ scale);
		}
		data.push_back(y);
	}
}

cpp_torch::network_torch<cpp_torch::Net>* toNet(void* nn)
{
	return (cpp_torch::network_torch<cpp_torch::Net>*)nn;
}

extern "C" _LIBRARY_EXPORTS void send_train_images(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(train_images, data);
}
extern "C" _LIBRARY_EXPORTS void send_train_labels(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(train_labels, data);
}
extern "C" _LIBRARY_EXPORTS void send_test_images(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(test_images, data);
}
extern "C" _LIBRARY_EXPORTS void send_test_labels(std::vector<tiny_dnn::vec_t>& data)
{
	scaling_data(test_labels, data);
}

extern "C" _LIBRARY_EXPORTS void get_train_images(std::vector<tiny_dnn::vec_t>& data)
{
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < train_images.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < train_images[0].size(); k++)
		{
			y.push_back(train_images[i][k] * scale);
		}
		data.push_back(y);
	}
}
extern "C" _LIBRARY_EXPORTS void get_train_labels(std::vector<tiny_dnn::vec_t>& data)
{
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < train_labels.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < train_labels[0].size(); k++)
		{
			y.push_back(train_labels[i][k] * scale);
		}
		data.push_back(y);
	}
}

extern "C" _LIBRARY_EXPORTS void get_test_images(std::vector<tiny_dnn::vec_t>& data)
{
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < test_images.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < test_images[0].size(); k++)
		{
			y.push_back(test_images[i][k] * scale);
		}
		data.push_back(y);
	}
}
extern "C" _LIBRARY_EXPORTS void get_test_labels(std::vector<tiny_dnn::vec_t>& data)
{
	std::vector<tiny_dnn::vec_t> yy;
	for (int i = 0; i < test_labels.size(); i++)
	{
		tiny_dnn::vec_t y;
		for (int k = 0; k < test_labels[0].size(); k++)
		{
			y.push_back(test_labels[i][k] * scale);
		}
		data.push_back(y);
	}
}


extern "C" _LIBRARY_EXPORTS int getBatchSize()
{
	return kTrainBatchSize;
}
extern "C" _LIBRARY_EXPORTS int getNumberOfEpochs()
{
	return kNumberOfEpochs;
}
extern "C" _LIBRARY_EXPORTS float getScale()
{
	return scale;
}

extern "C" _LIBRARY_EXPORTS float torch_get_loss(std::vector<tiny_dnn::vec_t>& train_images_,  std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	float loss = nn_->get_loss(train_images_, train_labels_, batch);
	return loss;
}
extern "C" _LIBRARY_EXPORTS float torch_get_Loss(int batch)
{
	float loss = nn_->get_loss(train_images, train_labels, batch);
	return loss;
}
extern "C" _LIBRARY_EXPORTS float torch_get_loss_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	float loss = toNet(nn)->get_loss(train_images_, train_labels_, batch);
	return loss;
}
extern  _LIBRARY_EXPORTS tiny_dnn::result torch_get_accuracy_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch)
{
	tiny_dnn::result result;
	if (train_images_.size())
	{
		result.num_total = 1;
		return result;
	}

	result = toNet(nn)->get_accuracy(train_images_, train_labels_);

	//ConfusionMatrix
	std::cout << "ConfusionMatrix:" << std::endl;
	result.print_detail(std::cout);
	std::cout << result.num_success << "/" << result.num_total << std::endl;
	printf("accuracy:%.3f%%\n", result.accuracy());

	return result;
}


extern "C" _LIBRARY_EXPORTS void* torch_model(std::string& ptfile)
{
	cpp_torch::Net bakup_model;
	cpp_torch::network_torch<cpp_torch::Net>* nn2 = new cpp_torch::network_torch<cpp_torch::Net>(bakup_model, device);
	*nn2 = *nn_;

	nn2->load(ptfile);

	return (void*)nn2;
}


extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x)
{
	for (auto& xx : x) xx /= scale;
	tiny_dnn::vec_t y = nn_->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		for (auto& yy : y) yy *= scale;
	}
	return y;
}
extern _LIBRARY_EXPORTS tiny_dnn::vec_t torch_model_predict(const void* nn, tiny_dnn::vec_t x)
{
	cpp_torch::network_torch<cpp_torch::Net>* nn2
		= (cpp_torch::network_torch<cpp_torch::Net>*)(nn);

	for (auto& xx : x) xx /= scale;

	tiny_dnn::vec_t y = nn2->predict(x);
	if (num_class >= 2)
	{
	}
	else
	{
		for (auto& yy : y) yy *= scale;
	}
	return y;
}

extern "C" _LIBRARY_EXPORTS void torch_delete_model()
{
	if (nn_ == nullptr) return;
	delete nn_;

	nn_ = nullptr;
}
extern "C" _LIBRARY_EXPORTS void torch_delete_load_model(void* n)
{
	if (n == nullptr) return;
	delete n;

	n = nullptr;
}


extern "C" _LIBRARY_EXPORTS void* torch_getNet()
{
	return(void*)(nn_);
}

extern "C" _LIBRARY_EXPORTS void* torch_getDevice()
{
	return(void*)(&device);
}
extern "C" _LIBRARY_EXPORTS void* torch_setDevice(const char* device_name)
{
	torch::DeviceType device_type;

	if (std::string(device_name) == std::string("gpu") || std::string(device_name) == std::string("cuda"))
	{
		if (torch::cuda::is_available()) {
			std::cout << "CUDA available! Training on GPU." << std::endl;
			device_type = torch::kCUDA;
		}
		else
		{
			std::cout << "CUDA no available -> Training on CPU." << std::endl;
			device_type = torch::kCPU;
		}
	}
	else
	{
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	device = torch::Device(device_type);
}

extern "C" _LIBRARY_EXPORTS void* torch_progress_display(size_t length)
{
	cpp_torch::progress_display* disp = new cpp_torch::progress_display(length);
	return (void*)disp;
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_restart(void* disp_, size_t length)
{
	cpp_torch::progress_display* disp = (cpp_torch::progress_display*)disp_;
	disp->restart(length);
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_count(void* disp_, int count)
{
	cpp_torch::progress_display& disp = *(cpp_torch::progress_display*)disp_;
	disp += count;
}
extern "C" _LIBRARY_EXPORTS void torch_progress_display_delete(void* disp_)
{
	cpp_torch::progress_display* disp = (cpp_torch::progress_display*)disp_;
	delete disp;
}

extern "C" _LIBRARY_EXPORTS void torch_stop_ongoing_training()
{
	nn_->stop_ongoing_training();
}

extern "C" _LIBRARY_EXPORTS void torch_save_nn(void* nn, const char* name)
{
	toNet(nn)->save(std::string(name));
}

extern "C" _LIBRARY_EXPORTS void torch_save(const char* name)
{
	nn_->save(std::string(name));
}
extern "C" _LIBRARY_EXPORTS void torch_load(const char* name)
{
	nn_->load(std::string(name));
}
extern "C" _LIBRARY_EXPORTS void* torch_load_new(const char* name)
{
	void* nn2 = torch_model(std::string(name));
	return nn2;
}



extern "C" _LIBRARY_EXPORTS int torch_train_custom(
	std::string define_layers_file_name,
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
)
{
	char buf[4096];
	FILE* fp = fopen(define_layers_file_name.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}

	char* p = NULL;
	int nparam = 0;
	while (fgets(buf, 4096, fp) != NULL)
	{
		if (buf[0] == '\n') continue;
		if (strstr(buf, "device="))
		{
			char dev[32];
			sscanf(buf, "device=%s\n", dev);
			if (strcmp(dev, "cpu") == 0) torch_setDevice("cpu");
			if (strcmp(dev, "gpu") == 0) torch_setDevice("gpu");
			if (strcmp(dev, "cuda") == 0) torch_setDevice("gpu");
			continue;
		}
		if (strstr(buf, "scale="))
		{
			sscanf(buf, "scale=%f", &scale);
			continue;
		}
		if (strstr(buf, "num_class="))
		{
			sscanf(buf, "num_class=%d", &num_class);
			continue;
		}
	}
	fclose(fp);

	if (train_images_.size() > 0 && train_labels_.size() > 0)
	{
		send_train_images(train_images_);

		auto scale_bak = scale;
		if (num_class >= 2)
		{
			scale = 1.0;
		}
		send_train_labels(train_labels_);
		scale = scale_bak;
	}

	kNumberOfEpochs = n_train_epochs;
	kTrainBatchSize = n_minibatch;

	cpp_torch::Net model;

	model.get()->device = device;
	
	fp = fopen(define_layers_file_name.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}

	p = NULL;
	while (fgets(buf, 4096, fp) != NULL)
	{
		if (buf[0] == '\n') continue;
		if (strstr(buf, "epoch="))
		{
			sscanf(buf, "epoch=%d", &kNumberOfEpochs);
			continue;
		}
		if (strstr(buf, "train_batch="))
		{
			sscanf(buf, "train_batch=%d", &kTrainBatchSize);
			continue;
		}
		if (strstr(buf, "test_batch="))
		{
			sscanf(buf, "test_batch=%d", &kTestBatchSize);
			continue;
		}
		if (strstr(buf, "learning_rate="))
		{
			sscanf(buf, "learning_rate=%f", &learning_rate);
			continue;
		}
		if (strstr(buf, "optimizer="))
		{
			char solv[32];
			sscanf(buf, "optimizer=%s\n", solv);
			optimizer_name = std::string(solv);
			continue;
		}
		//


		if (strstr(buf, "input"))
		{
			int input_c, input_w, input_h;
			nparam = sscanf(buf, "input c=%d w=%d h=%d", &input_c, &input_w, &input_h);
			if (nparam != 3)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (input_c <= 0 && input_w <= 0 && input_h <= 0)
			{
				input_c = 1;
				input_w = 1;
				input_h = train_images.size();
			}
			model.get()->setInput(input_c, input_w, input_h);
			continue;
		}
		char* token = NULL;
		if (strstr(buf, "fc"))
		{
			int unit = -1, bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "unit=")) != NULL)
				{
					nparam = sscanf(token, "unit=%d", &unit);
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					nparam = sscanf(token, "bias=%d", &bias);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (unit <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (bias == -1) bias = 1;
			model.get()->add_fc(unit, (bias != 0) ? true : false);
			continue;
		}
		/*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		*/
		if (strstr(buf, "conv2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 1, 1 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			bool bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
						dilation[1] = dilation[0];
					}
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					int b;
					nparam = sscanf(token, "bias=%d", &b);
					if (b) bias = true;
					else bias = false;
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			else
			{
				model.get()->add_conv2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ dilation[0],dilation[1] },
					bias
				);
			}
			continue;
		}

		/**
		* constructing transposed convolution layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param out_padding  [in] out_padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		if (strstr(buf, "conv_transpose2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 1, 1 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> out_padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			bool bias = true;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "out_padding=")) != NULL)
				{
					nparam = sscanf(token, "out_padding=(%d,%d)", &out_padding[0], &out_padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "out_padding=%d", &out_padding[0]);
						out_padding[1] = out_padding[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
						dilation[1] = dilation[0];
					}
				}
				if ((token = strstr(buf, "bias=")) != NULL)
				{
					int b;
					nparam = sscanf(token, "bias=%d", &b);
					if (b) bias = true;
					else bias = false;
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_conv_transpose2d(
					kernel_size[0],
					stride[0],
					padding[0],
					out_padding[0],
					dilation[0], bias);
			}
			else
			{
				model.get()->add_conv_transpose2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ out_padding[0],out_padding[1] },
					{ dilation[0],dilation[1] }, bias);
			}
			continue;
		}
		/* constructing max pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		**/
		if (strstr(buf, "maxpool2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 0, 0 };
			std::vector<int> padding = { 0, 0 };
			std::vector<int> dilation = { 1, 1 };
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
					}
				}
				if ((token = strstr(buf, "dilation=")) != NULL)
				{
					nparam = sscanf(token, "dilation=(%d,%d)", &dilation[0], &dilation[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "dilation=%d", &dilation[0]);
					}
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[0] <= 0 )
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}

			if (stride[0] <= 0 || stride[1] <= 0)
			{
				stride[0] = kernel_size[0];
				stride[1] = kernel_size[1];
			}

			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_maxpool2d(
					kernel_size[0],
					stride[0],
					padding[0],
					dilation[0]
				);
			}
			else
			{
				model.get()->add_maxpool2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] },
					{ dilation[0],dilation[1] }
				);
			}
			continue;
		}
		/**
		* constructing average pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		**/
		if (strstr(buf, "avgpool2d"))
		{
			int input_channels = -1;
			int output_channels = -1;
			std::vector<int> kernel_size = { -1, -1 };
			std::vector<int> stride = { 0, 0 };
			std::vector<int> padding = { 0, 0 };
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "input_channels=")) != NULL)
				{
					nparam = sscanf(token, "input_channels=%d", &input_channels);
				}
				if ((token = strstr(buf, "output_channels=")) != NULL)
				{
					nparam = sscanf(token, "output_channels=%d", &output_channels);
				}
				if ((token = strstr(buf, "kernel_size=")) != NULL)
				{
					nparam = sscanf(token, "kernel_size=(%d,%d)", &kernel_size[0], &kernel_size[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "kernel_size=%d", &kernel_size[0]);
						kernel_size[1] = kernel_size[0];
					}
				}
				if ((token = strstr(buf, "stride=")) != NULL)
				{
					nparam = sscanf(token, "stride=(%d,%d)", &stride[0], &stride[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "stride=%d", &stride[0]);
						stride[1] = stride[0];
					}
				}
				if ((token = strstr(buf, "padding=")) != NULL)
				{
					nparam = sscanf(token, "padding=(%d,%d)", &padding[0], &padding[1]);
					if (nparam != 2)
					{
						nparam = sscanf(token, "padding=%d", &padding[0]);
						padding[1] = padding[0];
					}
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (kernel_size[1] <= 0)
			{
				kernel_size[1] = kernel_size[0];
			}
			if (stride[1] <= 0)
			{
				stride[1] = stride[0];
			}
			if (padding[1] <= 0)
			{
				padding[1] = padding[0];
			}
			if (kernel_size[0] <= 0 )
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (stride[0] <= 0 || stride[1] <= 0)
			{
				stride[0] = kernel_size[0];
				stride[1] = kernel_size[1];
			}
			if (input_channels <= 0 || output_channels <= 0)
			{
				model.get()->add_avgpool2d(
					kernel_size[0],
					stride[0],
					padding[0]
				);
			}
			else
			{
				model.get()->add_avgpool2d_(input_channels, output_channels,
					{ kernel_size[0],kernel_size[1] },
					{ stride[0],stride[1] },
					{ padding[0],padding[1] }
				);
			}
			continue;
		}
		
		if (strstr(buf, "conv_drop"))
		{
			float rate = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rate=")) != NULL)
				{
					nparam = sscanf(token, "rate=%f", &rate);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (rate <= 0 )
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_conv_drop(rate);
			continue;
		}

		if (strstr(buf, "dropout"))
		{
			float rate = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rate=")) != NULL)
				{
					nparam = sscanf(token, "rate=%f", &rate);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (rate <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_dropout(rate);
			continue;
		}
		if (strstr(buf, "pixel_shuffle"))
		{
			float upscale_factor = -1;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "upscale_factor=")) != NULL)
				{
					nparam = sscanf(token, "upscale_factor=%f", &upscale_factor);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (upscale_factor <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_pixel_shuffle(upscale_factor);
			continue;
		}
		
		/**
		* constructing Batch Normalization layer
		* @param momentum        [in] momentum in the computation of the exponential
		* @param eos             [in] The epsilon value added for numerical stability.
		**/
		if (strstr(buf, "batchnorml2d"))
		{
			float momentum = 0.1;
			float eps = 1e-5;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "momentum=")) != NULL)
				{
					nparam = sscanf(token, "momentum=%f", &momentum);
				}
				if ((token = strstr(buf, "eps=")) != NULL)
				{
					nparam = sscanf(token, "eps=%f", &eps);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (eps <= 0 || momentum <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_bn2d(momentum, eps);
			continue;
		}

		/**
		* compute multi-layer long-short-term-memory (RNN)
		**/
		/**
		* @param rnn_type        [in] "rnn", "lstm", "gru"
		* @param sequence_length [in] number of sequence of length
		* @param hidden_size     [in] number of hidden of size
		* @param num_layers      [in] The number of recurrent layers (cells) to use.
		* @param dropout         [in] If non-zero, adds dropout with the given probability to the output of each RNN layer, except the final layer.
		**/
		if (strstr(buf, "recurrent"))
		{
			char rnn_type[32] = { '\0' };
			int sequence_length = 1;
			int hidden_size = 32;
			int num_layers = 1;
			float dropout = 0;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "rnn_type=")) != NULL)
				{
					nparam = sscanf(token, "rnn_type=%s", rnn_type);
				}
				if ((token = strstr(buf, "sequence_length=")) != NULL)
				{
					nparam = sscanf(token, "sequence_length=%d", &sequence_length);
				}
				if ((token = strstr(buf, "hidden_size=")) != NULL)
				{
					nparam = sscanf(token, "hidden_size=%d", &hidden_size);
				}
				if ((token = strstr(buf, "num_layers=")) != NULL)
				{
					nparam = sscanf(token, "num_layers=%d", &num_layers);
				}
				if ((token = strstr(buf, "dropout=")) != NULL)
				{
					nparam = sscanf(token, "dropout=%f", &dropout);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			if (sequence_length <= 0 || hidden_size <= 0 || num_layers <= 0)
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			if (std::string(rnn_type) != std::string("rnn") && std::string(rnn_type) != std::string("gru") && std::string(rnn_type) != std::string("lstm"))
			{
				printf("ERROR:%s", buf);
				return -1;
			}
			model.get()->add_recurrent(std::string(rnn_type), sequence_length, hidden_size, num_layers, dropout);
			continue;
		}
		if (strstr(buf, "tanh"))
		{
			model.get()->add_Tanh();
			continue;
		}
		if (strstr(buf, "leaky_relu"))
		{
			float negative_slope = 0.01;
			do {
				fgets(buf, 4096, fp);
				if ((token = strstr(buf, "slope=")) != NULL)
				{
					nparam = sscanf(token, "slope=%f", &negative_slope);
				}
				p = buf;
				while (isspace(*p))p++;
			} while (*p != '\0');

			model.get()->add_LeakyReLU(negative_slope);
			continue;
		}
		if (strstr(buf, "relu"))
		{
			model.get()->add_ReLU();
			continue;
		}
		if (strstr(buf, "selu"))
		{
			model.get()->add_SELU();
			continue;
		}
		if (strstr(buf, "sigmoid"))
		{
			model.get()->add_Sigmoid();
			continue;
		}
		if (strstr(buf, "log_softmax"))
		{
			model.get()->add_LogSoftmax(1);
			continue;
		}
		if (strstr(buf, "softmax"))
		{
			model.get()->add_Softmax(1);
			continue;
		}
	}
	fclose(fp);

	if (model.get()->layer.size() == 0)
	{
		return -2;
	}
	size_t outsize = 
		model.get()->layer[model.get()->layer.size() - 1].out_[0] *
		model.get()->layer[model.get()->layer.size() - 1].out_[1] *
		model.get()->layer[model.get()->layer.size() - 1].out_[2];

	if (outsize != train_labels[0].size())
	{
		printf("output size miss match %d != %d\n", outsize, train_labels[0].size());
		return -3;
	}
	//xavier
	for (auto w : model.get()->fc)
	{
		torch::nn::init::xavier_uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
		//auto &x = w->all_weights();
		//for (auto xx : x)
		//{
		//	torch::nn::init::xavier_uniform_(xx, torch::nn::init::calculate_gain(torch::kReLU));
		//}
	}
#if 0
	//uniform
	for (auto w : model.get()->fc)
	{
		torch::nn::init::uniform_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
	}
	//gaussian
	for (auto w : model.get()->fc)
	{
		torch::nn::init::normal_(w->weight, torch::nn::init::calculate_gain(torch::kReLU));
	}
	for (auto w : model.get()->lstm)
	{
	}
#endif

	nn_ = new cpp_torch::network_torch<cpp_torch::Net>(model, device);

	nn_->input_dim(1, 1, train_images[0].size());
	nn_->output_dim(1, 1, train_labels[0].size());
	nn_->classification = (num_class > 2);
	nn_->batch_shuffle = batch_shuffle;
	printf("num_class=%d\n", num_class);

	torch::optim::Optimizer* optimizer = nullptr;

	float moment = 0;
	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(learning_rate).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(learning_rate));

	if (optimizer_name == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (optimizer_name == "adam")
	{
		optimizer = &optimizerAdam;
		std::cout << optimizer_name << std::endl;
	}

	if (!test_mode)
	{
		// train
		std::cout << "start training" << std::endl;
		nn_->fit(optimizer, train_images, train_labels, n_minibatch,
			n_train_epochs, on_enumerate_minibatch,
			on_enumerate_epoch);
		std::cout << "end training." << std::endl;
	}
}



_LIBRARY_EXPORTS int torch_train_init()
{
	torch::manual_seed(1);
	return 0;
}

