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
		AVGPOOL2D = 5,
		DROPOUT = 6,
		BATCHNORMAL = 7,
		RNN = 8,
		LSTM = 9,
		GRU = 10,

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
		bool count_include_pad = true;
		bool ceil_mode = false;
		bool bias = true;

		int rnn_seqence_length = 1;
		int rnn_sequence_single_size = 1;

		int dim = 1;	//softmax, logsoftmax
		float_t dropout_rate = 0.0;
	};

	struct NetImpl : torch::nn::Module {
		NetImpl() :
			fc(1, nullptr), conv2d(1, nullptr),
			conv_drop(1, nullptr), bn(1, nullptr),
			lstm(1, nullptr),gru(1, nullptr), rnn(1, nullptr),
			device(torch::kCPU)
		{
			fc.clear();	conv2d.clear();	
			conv_drop.clear();	bn.clear();
			lstm.clear();gru.clear();rnn.clear();
		}

		int activation_count = 0;
		int maxpool2d_count = 0;
		int avgpool2d_count = 0;
		int dropout_count = 0;

		std::vector<torch::nn::Conv2d> conv2d;
		std::vector<torch::nn::Linear> fc;
		std::vector<torch::nn::FeatureDropout> conv_drop;
		std::vector<torch::nn::BatchNorm> bn;
		std::vector<torch::nn::LSTM> lstm;
		std::vector<torch::nn::GRU> gru;
		std::vector<torch::nn::RNN> rnn;


		torch::Device device;
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
		* applies average pooling operaton to the spatial data
		**/
		/**
		* constructing average pooling 2D layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size  [in] window(kernel) size of convolution
		* @param padding      [in] padding size
		* @param stride       [in] stride size
		**/
		void add_avgpool2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> stride = { 1,1 }, bool ceil_mode=false, bool count_include_pad = true)
		{
			int id = avgpool2d_count++;
			cpp_torch::LayerInOut inout;
			inout.name = "avgpool2d";
			inout.type = cpp_torch::LayerType::AVGPOOL2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = { input_channels, layer[i - 1].out_[1], layer[i - 1].out_[2] };

			inout.out_ = {
				output_channels,
				(int)floor((double)(inout.in_[1] + 2 * padding[0] - kernel_size[0]) / (double)stride[0] + 1),
				(int)floor((double)(inout.in_[2] + 2 * padding[1] - kernel_size[1]) / (double)stride[1] + 1)
			};

			inout.ceil_mode = ceil_mode;
			inout.count_include_pad = count_include_pad;
			inout.kernel_size = kernel_size;
			inout.padding = padding;
			inout.stride = stride;

			layer.emplace_back(inout);

			std::cout << "avgpool {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* constructing average  pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param padding         [in] padding size
		* @param stride       [in] specify the horizontal interval at which to apply the filters to the input
		**/
		void add_avgpool2d(int kernel_size, int padding, int stride = 1, int dilation = 1)
		{
			const int i = layer.size();
			add_avgpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { padding,padding }, { stride, stride });
		}
		/**
		* constructing average  pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		**/
		void add_avgpool2d(int kernel_size)
		{
			const int i = layer.size();
			add_avgpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { 0, 0 }, { kernel_size, kernel_size });
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
		void add_conv_drop(float_t rate)
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
		void add_dropout(float_t rate)
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
		void add_bn(float_t momentum=0.1, float_t eps= 1e-5)
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
		void add_recurrent(std::string& rnn_type, int sequence_length, int hidden_size, int num_layers = 1, float dropout = 0.0, const std::string& activatin="ReLU")
		{
			cpp_torch::LayerInOut inout;
			inout.name = rnn_type;
			
			if (rnn_type == "rnn")inout.type = cpp_torch::LayerType::RNN;
			if (rnn_type == "lstm")inout.type = cpp_torch::LayerType::LSTM;
			if (rnn_type == "gru")inout.type = cpp_torch::LayerType::GRU;

			int in;
			const int i = layer.size();
			in = layer[i - 1].out_[0] * layer[i - 1].out_[1] * layer[i - 1].out_[2];

			inout.rnn_seqence_length = sequence_length;
			inout.rnn_sequence_single_size = in / sequence_length;

			std::cout << layer[i - 1].out_ << std::endl;
			inout.in_ = { 1,sequence_length, inout.rnn_sequence_single_size };
			
			int id = -1;
			if (rnn_type == "rnn")
			{
				id = rnn.size();
				auto opt = torch::nn::RNNOptions(inout.rnn_sequence_single_size, hidden_size);
				opt = opt.batch_first(true);
				if (activatin == "ReLU") opt.activation(torch::nn::RNNActivation::ReLU);
				if (activatin == "TanH") opt.activation(torch::nn::RNNActivation::Tanh);
				opt = opt.layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				rnn.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::RNN(opt)));
			}
			if (rnn_type == "lstm")
			{
				id = lstm.size();
				auto opt = torch::nn::LSTMOptions(inout.rnn_sequence_single_size, hidden_size);
				opt = opt.batch_first(true);
				opt = opt.layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				lstm.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::LSTM(opt)));
			}
			if (rnn_type == "gru")
			{ 
				id = gru.size();
				auto opt = torch::nn::GRUOptions(inout.rnn_sequence_single_size, hidden_size);
				opt = opt.batch_first(true);
				opt = opt.layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				gru.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::GRU(opt)));
			}
			if (id == -1)
			{
				throw error_exception("recurrent type error");
			}
			inout.id = id;
			inout.out_ = { 1,sequence_length, hidden_size };
			layer.emplace_back(inout);

			std::cout << rnn_type << "{" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
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
			const int batch = x.sizes()[0];

			for (int i = 1; i < layer.size(); i++)
			{
				if (debug_dmp)cpp_torch::dump_dim(std::string("IN"), x);
				if (layer[i].type == cpp_torch::LayerType::FC)
				{
					const int in = layer[i - 1].out_[0]*layer[i - 1].out_[1]*layer[i - 1].out_[2];
					x = x.view({ batch, -1 });
					x = fc[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(fc[layer[i].id]->name(), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::CONV2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = conv2d[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(conv2d[layer[i].id]->name(), x);
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

					if (debug_dmp)cpp_torch::dump_dim(layer[i].name, x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::AVGPOOL2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = torch::avg_pool2d(x,
						{ layer[i].kernel_size[0],layer[i].kernel_size[1] },
						{ layer[i].stride[0], layer[i].stride[1] },
						{ layer[i].padding[0], layer[i].padding[1] },
						layer[i].ceil_mode, layer[i].count_include_pad);
					if (debug_dmp)cpp_torch::dump_dim(layer[i].name, x);
					continue;
				}

				if (layer[i].type == cpp_torch::LayerType::CONV_DROP)
				{
					const int in = layer[i - 1].out_[0]*layer[i - 1].out_[1]*layer[i - 1].out_[2];
					x = conv_drop[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(conv_drop[layer[i].id]->name(), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::DROPOUT)
				{
					const int in = layer[i - 1].out_[0] * layer[i - 1].out_[1] * layer[i - 1].out_[2];
					x = x.view({ -1, in });
					x = torch::dropout(x, layer[i].dropout_rate, is_training());
					if (debug_dmp)cpp_torch::dump_dim(layer[i].name, x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::BATCHNORMAL)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = bn[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(bn[layer[i].id]->name(), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::LSTM ||
					layer[i].type == cpp_torch::LayerType::GRU  ||
					layer[i].type == cpp_torch::LayerType::RNN
					)
				{
					const int in = layer[i - 1].out_[0] * layer[i - 1].out_[1] * layer[i - 1].out_[2];
					x = x.view({ -1, layer[i].rnn_seqence_length, layer[i].rnn_sequence_single_size });
					//dump_dim("X", x);
					if (layer[i].type == cpp_torch::LayerType::LSTM)
					{
						x = lstm[layer[i].id]->forward(x).output;
					}else
					if (layer[i].type == cpp_torch::LayerType::GRU)
					{
						x = gru[layer[i].id]->forward(x).output;
					}else
					if (layer[i].type == cpp_torch::LayerType::RNN)
					{
						x = rnn[layer[i].id]->forward(x).output;
					}
					x = x.view({ batch,  layer[i].rnn_seqence_length, -1 });
					//dump_dim("X", x);

					if (layer[i].type == cpp_torch::LayerType::LSTM)
					{
						if (debug_dmp)cpp_torch::dump_dim(lstm[layer[i].id]->name(), x);
					}
					else
					if (layer[i].type == cpp_torch::LayerType::GRU)
					{
						if (debug_dmp)cpp_torch::dump_dim(gru[layer[i].id]->name(), x);
					}
					else
					if (layer[i].type == cpp_torch::LayerType::RNN)
					{
						if (debug_dmp)cpp_torch::dump_dim(rnn[layer[i].id]->name(), x);
					}
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
	};
	TORCH_MODULE(Net); // creates module holder for NetImpl
}
#endif
