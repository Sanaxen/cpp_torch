#ifndef __libtorch_sequential_layer_model_H
#define __libtorch_sequential_layer_model_H
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

// Tensor -> { C, H, W}
namespace cpp_torch
{
	enum class LayerType : int16_t {
		INITIAL = 0,
		FC = 1,
		CONV2D = 2,
		CONV_DROP = 3,
		MAXPOOL2D = 4,
		DROPOUT = 5,
		BATCHNORMAL = 6,

		ReLU = 100,
		LeakyReLU = 101,
		SELU = 102,
		Sigmoid = 103,
		Tanh = 104,
		Softmax = 105,
		LogSoftmax = 106
	};

	class LayerInOut
	{
	public:
		int id = -1;
		LayerType type = LayerType::INITIAL;
		std::string name = "";

		std::vector<int> in_;
		std::vector<int> out_;

		std::vector<int> kernel_size;
		std::vector<int> padding = { 0,0 };
		std::vector<int> stride = { 1,1 };
		std::vector<int> dilation = { 1,1 };
		bool bias = true;

		int dim = 1;	//softmax, logsoftmax
		float dropout_rate = 0.0;
	};

	struct NetImpl : torch::nn::Module {
		NetImpl() :
			fc(1, nullptr), conv2d(1, nullptr),
			conv_drop(1, nullptr), bn(1, nullptr)
		{
			fc.clear();
			conv2d.clear();
			conv_drop.clear();
			bn.clear();
		}

		std::vector<cpp_torch::LayerInOut> layer;
		void setInput(int channels, int w, int h)
		{
			cpp_torch::LayerInOut inout;
			inout.name = "input";
			inout.type = cpp_torch::LayerType::INITIAL;
			inout.id = 0;
			inout.in_ = { channels, h, w };
			inout.out_ = { channels, h, w };
			layer.push_back(inout);

			std::cout << "input {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}


		/**
		* compute fully-connected(matmul) operation
		**/
		/**
		* @param out [in] number of elements of the output
		* @param bias [in] whether to include additional bias to the layer
		**/
		void add_fc(int out, bool bias = true)
		{
			int id = fc.size();
			cpp_torch::LayerInOut inout;
			inout.name = "fc";
			inout.type = cpp_torch::LayerType::FC;
			inout.id = id;

			int in;
			const int i = layer.size();
			in = layer[i - 1].out_[0] * layer[i - 1].out_[1] * layer[i - 1].out_[2];

			std::cout << layer[i - 1].out_ << std::endl;
			inout.in_ = layer[i - 1].out_;
			fc.emplace_back(register_module("fc" + std::to_string(id), torch::nn::Linear(torch::nn::LinearOptions(in, out).with_bias(bias))));
			inout.out_ = { 1,1, out };
			layer.emplace_back(inout);

			std::cout << "fc {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* 2D convolution layer
		*
		* take input as two-dimensional *image* and applying filtering operation.
		**/
		/**
		* constructing convolutional layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param padding      [in] padding size
		* @param stride       [in] stride size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		void add_conv2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> stride = { 1,1 }, std::vector<int> dilation = { 1,1 }, bool bias = true)
		{
			int id = conv2d.size();
			cpp_torch::LayerInOut inout;
			inout.name = "conv2d";
			inout.type = cpp_torch::LayerType::CONV2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = { input_channels,layer[i - 1].out_[1], layer[i - 1].out_[2] };

			auto& l = register_module("conv2d" + std::to_string(id), torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, { kernel_size[0],kernel_size[1] }).with_bias(bias).padding({ padding[0],padding[1] }).stride({ stride[0],stride[1] }).dilation({ dilation[0],dilation[1] })));
			conv2d.emplace_back(l);

			inout.out_ = {
				output_channels,
				(int)floor((double)(inout.in_[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (double)stride[0] + 1),
				(int)floor((double)(inout.in_[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (double)stride[1] + 1)
			};

			inout.bias = bias;
			inout.dilation = dilation;
			inout.kernel_size = kernel_size;
			inout.padding = padding;
			inout.stride = stride;
			layer.emplace_back(inout);

			std::cout << "conv {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}
		/**
		* constructing convolutional layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param padding         [in] padding size
		* @param stride       [in] stride size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		void add_conv2d(int input_channels, int output_channels, int kernel_size, int padding = 0, int stride = 1, int dilation = 1, bool bias = true)
		{
			add_conv2d_(input_channels, output_channels, { kernel_size, kernel_size }, { padding, padding }, { stride, stride }, { dilation, dilation }, bias);
		}

		/**
		* applies max-pooling operaton to the spatial data
		**/
		/**
		* constructing max pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param padding      [in] padding size
		* @param stride       [in] stride size
		* @param dilation     [in] dilation
		**/
		void add_maxpool2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> stride = { 1,1 }, std::vector<int> dilation = { 1,1 })
		{
			int id = maxpool2d_count++;
			cpp_torch::LayerInOut inout;
			inout.name = "maxpool2d";
			inout.type = cpp_torch::LayerType::MAXPOOL2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = { input_channels, layer[i - 1].out_[1], layer[i - 1].out_[2] };

			inout.out_ = {
				output_channels,
				(int)floor((double)(inout.in_[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (double)stride[0] + 1),
				(int)floor((double)(inout.in_[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (double)stride[1] + 1)
			};

			inout.dilation = dilation;
			inout.kernel_size = kernel_size;
			inout.padding = padding;
			inout.stride = stride;

			layer.emplace_back(inout);

			std::cout << "maxpool {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}
		/**
		* constructing max pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param padding         [in] padding size
		* @param stride       [in] specify the horizontal interval at which to apply the filters to the input
		* @param dilation     [in] specify the horizontal interval to control the spacing between the kernel points
		**/
		void add_maxpool2d(int kernel_size, int padding, int stride = 1, int dilation = 1)
		{
			const int i = layer.size();
			add_maxpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { padding,padding }, { stride, stride }, { dilation, dilation });
		}
		/**
		* constructing max pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		**/
		void add_maxpool2d(int kernel_size)
		{
			const int i = layer.size();
			add_maxpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { 0, 0 }, { kernel_size, kernel_size }, { 1, 1 });
		}

		/**
		* constructing FeatureDropout 2D layer
		*
		* @param rate     [in] probability of an element to be zero-ed.
		**/
		/*
		* Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j]).
		* Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
		* Usually the input comes from nn.Conv2d modules.
		* As described in the paper Efficient Object Localization Using Convolutional Networks , if adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then i.i.d. dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease.
		*/
		void add_conv_drop(float rate)
		{
			int id = conv_drop.size();
			cpp_torch::LayerInOut inout;
			inout.name = "conv_drop";
			inout.type = cpp_torch::LayerType::CONV_DROP;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = layer[i - 1].out_;
			conv_drop.emplace_back(register_module("conv_drop" + std::to_string(id), torch::nn::FeatureDropout(torch::nn::FeatureDropout(rate))));
			inout.out_ = inout.in_;
			layer.emplace_back(inout);

			std::cout << "conv_drop {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* constructing Dropout layer
		*
		* @param rate     [in] probability of an element to be zero-ed.
		**/
		/*
		* During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
		* Each channel will be zeroed out independently on every forward call.
		* This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .
		* Furthermore, the outputs are scaled by a factor of \frac{1}{1-p}
		* during training. This means that during evaluation the module simply computes an identity function.
		*/
		void add_dropout(float rate)
		{
			cpp_torch::LayerInOut inout;
			inout.name = "dropout";
			inout.type = cpp_torch::LayerType::DROPOUT;
			inout.id = dropout_count++;
			inout.dropout_rate = rate;

			const int i = layer.size();
			inout.in_ = layer[i - 1].out_;
			inout.out_ = inout.in_;

			layer.emplace_back(inout);

			std::cout << "drop {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* constructing Batch Normalization layer
		* @param momentum        [in] momentum in the computation of the exponential
		* @param eos             [in] The epsilon value added for numerical stability.
		**/
		void add_bn(float momentum=0.1, float eps= 1e-5)
		{
			int id = bn.size();
			cpp_torch::LayerInOut inout;
			inout.name = "batchnorml";
			inout.type = cpp_torch::LayerType::BATCHNORMAL;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = layer[i - 1].out_;
			bn.emplace_back(register_module("bn" + std::to_string(id), torch::nn::BatchNorm(torch::nn::BatchNormOptions(inout.in_[0]).eps(eps).momentum(momentum))));

			inout.out_ = inout.in_;
			//printf("out %d %d %d ->", inout.outC, inout.outH, inout.outW);
			layer.emplace_back(inout);


			std::cout << "bn {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

#define ACTIVATION_LAYER( id_name ) void add_##id_name()	\
		{												\
		cpp_torch::LayerInOut inout;					\
		inout.name = #id_name;							\
		inout.type = cpp_torch::LayerType::##id_name;	\
		inout.id = activation_count++;					\
		const int i = layer.size();						\
		inout.in_ = layer[i - 1].out_;					\
		inout.out_ = inout.in_;							\
		layer.push_back(inout);							\
		}

#define ACTIVATION_LAYER1( id_name ) void add_##id_name(const int d)	\
		{												\
		cpp_torch::LayerInOut inout;					\
		inout.name = #id_name;							\
		inout.type = cpp_torch::LayerType::##id_name;	\
		inout.id = activation_count++;					\
		const int i = layer.size();						\
		inout.in_ = layer[i - 1].out_;					\
		inout.out_ = inout.in_;							\
		inout.dim = d;									\
		layer.push_back(inout);							\
		}

		ACTIVATION_LAYER(ReLU)
		ACTIVATION_LAYER(LeakyReLU)
		ACTIVATION_LAYER(SELU)
		ACTIVATION_LAYER(Sigmoid)
		ACTIVATION_LAYER(Tanh)
		ACTIVATION_LAYER1(Softmax)
		ACTIVATION_LAYER1(LogSoftmax)

		int debug_dmp = 0;
		torch::Tensor forward(torch::Tensor x)
		{

			for (int i = 1; i < layer.size(); i++)
			{
				if (debug_dmp)cpp_torch::dump_dim(std::string(""), x);
				if (layer[i].type == cpp_torch::LayerType::FC)
				{
					const int in = layer[i - 1].out_[0]*layer[i - 1].out_[1]*layer[i - 1].out_[2];
					x = x.view({ -1, in });
					x = fc[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(std::string("fc"), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::CONV2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = conv2d[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(std::string("conv"), x);
					continue;
				}

				if (layer[i].type == cpp_torch::LayerType::MAXPOOL2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = torch::max_pool2d(x,
							{layer[i].kernel_size[0],layer[i].kernel_size[1] },
							{layer[i].stride[0], layer[i].stride[1]},
							{layer[i].padding[0], layer[i].padding[1]},
							{layer[i].dilation[0], layer[i].dilation[1]}
					);

					if (debug_dmp)cpp_torch::dump_dim(std::string("maxpool"), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::CONV_DROP)
				{
					const int in = layer[i - 1].out_[0]*layer[i - 1].out_[1]*layer[i - 1].out_[2];
					x = conv_drop[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(std::string("conv_drop"), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::DROPOUT)
				{
					const int in = layer[i - 1].out_[0] * layer[i - 1].out_[1] * layer[i - 1].out_[2];
					x = x.view({ -1, in });
					x = torch::dropout(x, layer[i].dropout_rate, is_training());
					if (debug_dmp)cpp_torch::dump_dim(std::string("drop"), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::BATCHNORMAL)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = bn[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(std::string("bn"), x);
					continue;
				}


				//Activation
				if (layer[i].type >= cpp_torch::LayerType::ReLU)
				{
					if (debug_dmp)cpp_torch::dump_dim(layer[i].name, x);

					switch (layer[i].type)
					{
					case cpp_torch::LayerType::ReLU:
						x = torch::relu(x); break;
					case cpp_torch::LayerType::LeakyReLU:
						x = torch::leaky_relu(x); break;
					case cpp_torch::LayerType::SELU:
						x = torch::selu(x); break;
					case cpp_torch::LayerType::Sigmoid:
						x = torch::sigmoid(x); break;
					case cpp_torch::LayerType::Tanh:
						x = torch::tanh(x); break;
					case cpp_torch::LayerType::Softmax:
						x = torch::softmax(x, layer[i].dim); break;
					case cpp_torch::LayerType::LogSoftmax:
						x = torch::log_softmax(x, layer[i].dim); break;
					default:
						break;
						/* empty */
					}
				}
			}
			debug_dmp = 0;
			return x;
		}

		int activation_count = 0;
		int maxpool2d_count = 0;
		int dropout_count = 0;

		std::vector<torch::nn::Conv2d> conv2d;
		std::vector<torch::nn::Linear> fc;
		std::vector<torch::nn::FeatureDropout> conv_drop;
		std::vector<torch::nn::BatchNorm> bn;
	};
	TORCH_MODULE(Net); // creates module holder for NetImpl
}
#endif
