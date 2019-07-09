#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../../torch_util/libtorch_utils.h"
#include "../../torch_util/libtorch.hpp"

//#define TEST1
#define TEST2

#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 3;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

#define TEST

namespace cpp_torch {
	enum class LayerType : int16_t {
		INITIAL = 0,
		FC = 1,
		CONV2D = 2,
		CONV_DROP = 3,
		MAXPOOL2D = 4,
		DROPOUT = 5,
		RULE = 100
	};

	class LayerInOut
	{
	public:
		int id = -1;
		LayerType type = LayerType::INITIAL;
		std::string name = "";
		int inC;
		int inW;
		int inH;

		int outC;
		int outW;
		int outH;

		int w_kernel_size = 0;
		int w_padding = 0;
		int w_stride = 0;
		int w_dilation = 0;

		int h_kernel_size = 0;
		int h_padding = 0;
		int h_stride = 0;
		int h_dilation = 0;
		bool bias = true;

		float dropout_rate = 0.0;

	};
}

#ifndef TEST
struct NetImpl : torch::nn::Module {
	NetImpl()
		: conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
		conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
		fc1(320, 50),
		fc2(50, 10) {
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv2_drop", conv2_drop);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	torch::Tensor forward(torch::Tensor x) {
		
		//cpp_torch::dump_dim(std::string("xx"), conv1->forward(x));
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		x = torch::relu(
			torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));

		//cpp_torch::dump_dim(std::string("xx"),x);
		x = x.view({ -1, 320 });
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		x = fc2->forward(x);
		//return x;
		return torch::log_softmax(x, /*dim=*/1);
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::FeatureDropout conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};
TORCH_MODULE(Net); // creates module holder for NetImpl
#else
struct NetImpl : torch::nn::Module {
	NetImpl(): 
		conv2d(1, nullptr),	fc(1, nullptr), conv_drop(1, nullptr)
	{
		fc.clear();
		conv2d.clear();
		conv_drop.clear();
	}

	std::vector<cpp_torch::LayerInOut> layer;
	void setInput(int channels, int w, int h)
	{
		cpp_torch::LayerInOut inout;
		inout.name = "input";
		inout.type = cpp_torch::LayerType::INITIAL;
		inout.id = 0;
		inout.inC = channels;
		inout.inH = h;
		inout.inW = w;
		inout.outC = channels;
		inout.outH = h;
		inout.outW = w;
		layer.emplace_back(inout);
	}


	void add_fc(int out, bool bias=true)
	{
		int id = fc.size();
		cpp_torch::LayerInOut inout;
		inout.name = "fc";
		inout.type = cpp_torch::LayerType::FC;
		inout.id = id;

		int in;
		const int i = layer.size();
		in = layer[i - 1].inC*layer[i - 1].inW*layer[i - 1].inH;

		fc.emplace_back(register_module("fc" + std::to_string(id), torch::nn::Linear(torch::nn::LinearOptions(in, out).with_bias(bias))));
		inout.outC = 1;
		inout.outW = 1;
		inout.outH = out;
		layer.emplace_back(inout);
	}

	void add_conv2d_(int input_channels, int output_channels, int w_kernel_size=1, int h_kernel_size=1, int w_padding = 0, int h_padding = 0, int w_stride = 1, int h_stride = 1, int w_dilation = 1, int h_dilation = 1, bool bias = true)
	{
		int id = conv2d.size();
		cpp_torch::LayerInOut inout;
		inout.name = "conv2d";
		inout.type = cpp_torch::LayerType::CONV2D;
		inout.id = id;

		const int i = layer.size();
		inout.inC = input_channels;
		inout.inW = layer[i - 1].outW;
		inout.inH = layer[i - 1].outH;

		auto& l = register_module("conv2d" + std::to_string(id), torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, { h_kernel_size, w_kernel_size }).with_bias(bias).padding({ h_padding,w_padding }).stride({ h_stride,w_stride }).dilation({ h_dilation, w_dilation })));
		conv2d.emplace_back(l);

		inout.outC = output_channels;
		inout.outW = (int)floor((double)(inout.inW + 2 * w_padding - w_dilation * (w_kernel_size - 1) - 1) / (double)w_stride + 1);
		inout.outH = (int)floor((double)(inout.inH + 2 * h_padding - h_dilation * (h_kernel_size - 1) - 1) / (double)h_stride + 1);
		
		inout.bias = bias;
		inout.w_dilation = w_dilation;
		inout.w_kernel_size = w_kernel_size;
		inout.w_padding = w_padding;
		inout.w_stride = w_stride;

		inout.h_dilation = h_dilation;
		inout.h_kernel_size = h_kernel_size;
		inout.h_padding = h_padding;
		inout.h_stride = h_stride;
		layer.emplace_back(inout);
	}
	void add_conv2d(int input_channels, int output_channels, int kernel_size, int padding = 0, int stride = 1, int dilation = 1, bool bias = true)
	{
		add_conv2d_(input_channels, output_channels, kernel_size, kernel_size, padding, padding, stride, stride, dilation, dilation, bias);
	}

	void add_maxpool2d_(int input_channels, int output_channels, int w_kernel_size = 1, int h_kernel_size = 1, int w_padding = 0, int h_padding = 0, int w_stride = 1, int h_stride = 1, int w_dilation = 1, int h_dilation = 1, bool bias = true)
	{
		int id = maxpool2d_count++;
		cpp_torch::LayerInOut inout;
		inout.name = "maxpool2d";
		inout.type = cpp_torch::LayerType::MAXPOOL2D;
		inout.id = id;

		const int i = layer.size();
		inout.inC = input_channels;
		inout.inW = layer[i - 1].outW;
		inout.inH = layer[i - 1].outH;

		inout.outC = output_channels;
		inout.outW = (int)floor((double)(inout.inW + 2 * w_padding - w_dilation * (w_kernel_size - 1) - 1) / (double)w_stride + 1);
		inout.outH = (int)floor((double)(inout.inH + 2 * h_padding - h_dilation * (h_kernel_size - 1) - 1) / (double)h_stride + 1);

		inout.bias = bias;
		inout.w_dilation = w_dilation;
		inout.w_kernel_size = w_kernel_size;
		inout.w_padding = w_padding;
		inout.w_stride = w_stride;

		inout.h_dilation = h_dilation;
		inout.h_kernel_size = h_kernel_size;
		inout.h_padding = h_padding;
		inout.h_stride = h_stride;
		layer.emplace_back(inout);
	}
	void add_maxpool2d(int kernel_size, int padding = 0, int stride = 1, int dilation = 1, bool bias = true)
	{
		const int i = layer.size();
		add_maxpool2d_(layer[i - 1].outC, layer[i - 1].outC, kernel_size, kernel_size, padding, padding, stride, stride, dilation, dilation, bias);
	}

	void add_conv_drop(float rate)
	{
		int id = conv_drop.size();
		cpp_torch::LayerInOut inout;
		inout.name = "conv_drop";
		inout.type = cpp_torch::LayerType::CONV_DROP;
		inout.id = id;

		const int i = layer.size();
		inout.inC = layer[i - 1].outC;
		inout.inW = layer[i - 1].outW;
		inout.inH = layer[i - 1].outH;
		conv_drop.emplace_back(register_module("conv_drop" + std::to_string(id), torch::nn::FeatureDropout(torch::nn::FeatureDropout(rate))));
		inout.outC = inout.inC;
		inout.outW = inout.inW;
		inout.outH = inout.inH;
		layer.emplace_back(inout);
	}

	void add_dropout(float rate)
	{
		cpp_torch::LayerInOut inout;
		inout.name = "dropout";
		inout.type = cpp_torch::LayerType::DROPOUT;
		inout.id = dropout_count++;
		inout.dropout_rate = rate;

		const int i = layer.size();
		inout.inC = layer[i - 1].outC;
		inout.inW = layer[i - 1].outW;
		inout.inH = layer[i - 1].outH;
		inout.outC = inout.inC;
		inout.outW = inout.inW;
		inout.outH = inout.inH;

		layer.emplace_back(inout);
	}

	void add_relu()
	{
		cpp_torch::LayerInOut inout;
		inout.name = "relu";
		inout.type = cpp_torch::LayerType::RULE;
		inout.id = relu_count++;
		const int i = layer.size();
		inout.inC = layer[i - 1].outC;
		inout.inW = layer[i - 1].outW;
		inout.inH = layer[i - 1].outH;
		inout.outC = inout.inC;
		inout.outW = inout.inW;
		inout.outH = inout.inH;

		layer.push_back(inout);
	}
	
	torch::Tensor forward(torch::Tensor x) {

		for (int i = 1; i < layer.size(); i++)
		{
			if (layer[i].type == cpp_torch::LayerType::FC)
			{
				const int in = layer[i - 1].outC*layer[i - 1].outW*layer[i - 1].outH;
				x = x.view({ -1, in });
				x = fc[layer[i].id]->forward(x);
				//cpp_torch::dump_dim("fc=>" + layer[i].name, x);
				//printf("%d %d %d\n", layer[i].outC, layer[i].outH, layer[i].outW);
				continue;
			}
			if (layer[i].type == cpp_torch::LayerType::CONV2D)
			{
				x = x.view({ -1, layer[i - 1].outC, layer[i - 1].outH, layer[i - 1].outW });
				x = conv2d[layer[i].id]->forward(x);
				//cpp_torch::dump_dim("conv2d=>" + layer[i].name, x);
				//printf("%d %d %d\n", layer[i].outC, layer[i].outH, layer[i].outW);
				continue;
			}
			if (layer[i].type == cpp_torch::LayerType::MAXPOOL2D)
			{
				x = x.view({ -1, layer[i - 1].outC, layer[i - 1].outH, layer[i - 1].outW });
				x = torch::max_pool2d(x,
				{ layer[i].h_kernel_size, layer[i].w_kernel_size },
				{ layer[i].h_stride, layer[i].w_stride },
				{ layer[i].h_padding, layer[i].w_padding },
				{ layer[i].h_dilation, layer[i].w_dilation });
				//cpp_torch::dump_dim("maxpool2d=>" + layer[i].name, x);
				//printf("%d %d %d\n", layer[i].outC, layer[i].outH, layer[i].outW);

				continue;
			}
			if (layer[i].type == cpp_torch::LayerType::CONV_DROP)
			{
				const int in = layer[i - 1].outC*layer[i - 1].outW*layer[i - 1].outH;
				x = conv_drop[layer[i].id]->forward(x);
				//cpp_torch::dump_dim("conv_drop=>" + layer[i].name, x);
				//printf("%d %d %d\n", layer[i].outC, layer[i].outH, layer[i].outW);
				continue;
			}
			if (layer[i].type == cpp_torch::LayerType::DROPOUT)
			{
				const int in = layer[i - 1].outC*layer[i - 1].outW*layer[i - 1].outH;
				x = x.view({ -1, in });
				x = torch::dropout(x, layer[i].dropout_rate, is_training());
				continue;
			}
			if (layer[i].type == cpp_torch::LayerType::RULE)
			{
				x = torch::relu(x);
				continue;
			}
		}
		return torch::log_softmax(x, /*dim=*/1);
		//return x;
	}
	
	int relu_count = 0;
	int maxpool2d_count = 0;
	int dropout_count = 0;
	
	std::vector<torch::nn::Conv2d> conv2d;
	std::vector<torch::nn::Linear> fc;
	std::vector<torch::nn::FeatureDropout> conv_drop;
};
TORCH_MODULE(Net); // creates module holder for NetImpl
#endif


// load MNIST dataset
std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

void read_mnist_dataset(const std::string &data_dir_path)
{
	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels-idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images-idx3-ubyte",
		&train_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);

	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels-idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images-idx3-ubyte",
		&test_images, 0.0, 1.0, 0, 0, 0.1307, 0.3081);
}

void learning_and_test_mnist_dataset(torch::Device device)
{

	Net model;

#ifdef TEST
	model.get()->setInput(1, 28, 28);
	model.get()->add_conv2d(1, 10, 5);
	model.get()->add_maxpool2d(2);
	model.get()->add_relu();

	model.get()->add_conv2d(10, 20, 5);
	model.get()->add_conv_drop(0.5);
	model.get()->add_maxpool2d(2);
	model.get()->add_relu();
	model.get()->add_fc(50);
	model.get()->add_relu();
	model.get()->add_dropout(0.5);
	model.get()->add_fc(10);
#endif

	//torch::optim::SGD optimizer(
	//	model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

	cpp_torch::network_torch<Net> nn(model, device);
	nn.input_dim(1, 28, 28);
	nn.out_dim(1, 1, 10);
	nn.classification = true;

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	auto optimizer =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(0.01));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += kTrainBatchSize;
	};

	// train
	nn.train(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	float_t loss = nn.get_loss(train_images, train_labels);
	printf("loss:%f\n", loss);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	cpp_torch::print_ConfusionMatrix(res);

	nn.test(test_images, test_labels, kTrainBatchSize);

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


	read_mnist_dataset(std::string(kDataRoot));

	learning_and_test_mnist_dataset(device);
}
