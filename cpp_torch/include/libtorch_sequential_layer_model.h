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
		BATCHNORMAL2D = 7,
		CONV_TRANSPOSE2D = 8,
		PIXEL_SHUFFLE = 9,
		RNN = 90,
		LSTM = 91,
		GRU = 92,

		ReLU = 100,				// ReLU(x)
		ReLU_ = 101,			// ReLU(x,inplace=True) 
		LeakyReLU = 102,
		LeakyReLU_ = 103,		// LeakyReLU(x,inplace=True) 
		SELU = 104,
		Sigmoid = 105,
		Tanh = 106,
		Softmax = 107,
		LogSoftmax = 108,
		Squeeze = 109
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
		std::vector<int> out_padding = { 0,0 };
		std::vector<int> stride = { 1,1 };
		std::vector<int> dilation = { 1,1 };
		bool count_include_pad = true;
		bool ceil_mode = false;
		bool bias = true;

		int rnn_hidden_size = 0;
		int rnn_seqence_length = 1;
		int rnn_sequence_single_size = 1;

		int dim = 1;	//softmax, logsoftmax
		float_t dropout_rate = 0.0;
		float_t negative_slope = 0.01;	//LeakyReLU(negative_slope=0.01)
		int upscale_factor = 1;	//pixel_shuffle
	};

	struct NetImpl : torch::nn::Module {
		NetImpl() :
			fc(1, nullptr), conv2d(1, nullptr),
			conv_drop(1, nullptr), bn2d(1, nullptr),
			lstm(1, nullptr),gru(1, nullptr), rnn(1, nullptr),
			conv_transpose2d(1, nullptr),
			device(torch::kCPU)
		{
			fc.clear();	conv2d.clear();	
			conv_drop.clear();	bn2d.clear();
			lstm.clear();gru.clear();rnn.clear(),
			conv_transpose2d.clear();
		}

		int activation_count = 0;
		int maxpool2d_count = 0;
		int avgpool2d_count = 0;
		int dropout_count = 0;
		int pixel_shuffle_count = 0;

		int pycode_dump_only = 0;
		FILE* pycode_dump = NULL;
		std::vector<torch::nn::Conv2d> conv2d;
		std::vector<torch::nn::ConvTranspose2d> conv_transpose2d;
		std::vector<torch::nn::Linear> fc;
		std::vector<torch::nn::Dropout2d> conv_drop;
		std::vector<torch::nn::BatchNorm2d> bn2d;
		std::vector<torch::nn::LSTM> lstm;
		std::vector<torch::nn::GRU> gru;
		std::vector<torch::nn::RNN> rnn;


		torch::Device device;
		std::vector<cpp_torch::LayerInOut> layer;
		void setInput(int channels, int w, int h)
		{
			pycode_dump = fopen("torch_pycode.dmp", "w");
			cpp_torch::LayerInOut inout;
			inout.name = "input";
			inout.type = cpp_torch::LayerType::INITIAL;
			inout.id = 0;
			inout.in_ = { channels, h, w };
			inout.out_ = { channels, h, w };
			layer.push_back(inout);

			if (pycode_dump)
			{
				fprintf(pycode_dump, "#input %d\n#", inout.in_.size());
				for (int i = 0; i < inout.in_.size(); i++)
				{
					fprintf(pycode_dump, "%d", inout.in_[i]);
					if (i < inout.in_.size() - 1)fprintf(pycode_dump, ",");
					else fprintf(pycode_dump, "\n");
				}
				fprintf(pycode_dump, "#output %d\n#", inout.out_.size());
				for (int i = 0; i < inout.out_.size(); i++)
				{
					fprintf(pycode_dump, "%d", inout.out_[i]);
					if (i < inout.out_.size() - 1)fprintf(pycode_dump, ",");
					else fprintf(pycode_dump, "\n");
				}
				fprintf(pycode_dump, "class Net(nn.Module):\n");
				fprintf(pycode_dump, "    def __init__(self, ngpu):\n");
				fprintf(pycode_dump, "        super(Net, self).__init__()\n");
				fprintf(pycode_dump, "        self.ngpu = ngpu\n");
				fprintf(pycode_dump, "        self.main = nn.Sequential(\n");
			}
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

			//std::cout << layer[i - 1].out_ << std::endl;
			inout.in_ = layer[i - 1].out_;
			fc.emplace_back(register_module("fc" + std::to_string(id), torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(bias))));
			inout.out_ = { 1,1, out };
			layer.emplace_back(inout);

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.Linear(%d,%d,bias=%s),\n", in, out, bias ? "True" : "False");
			}
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
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		void add_conv2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> stride = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> dilation = { 1,1 }, bool bias = true)
		{
			int id = conv2d.size();
			cpp_torch::LayerInOut inout;
			inout.name = "conv2d";
			inout.type = cpp_torch::LayerType::CONV2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = { input_channels,layer[i - 1].out_[1], layer[i - 1].out_[2] };

			auto& l = register_module("conv2d" + std::to_string(id), torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, { kernel_size[0],kernel_size[1] }).bias(bias).padding({ padding[0],padding[1] }).stride({ stride[0],stride[1] }).dilation({ dilation[0],dilation[1] })));
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

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.Conv2d(%d,%d,%d,stride=%d,padding=%d,dilation=%d,bias=%s),\n", input_channels, output_channels, kernel_size[0], stride[0], padding[0], dilation[0], bias ? "True" : "False");
			}
			std::cout << "conv {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}
		/**
		* constructing convolutional layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding         [in] padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		void add_conv2d(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1, bool bias = true)
		{
			add_conv2d_(input_channels, output_channels, { kernel_size, kernel_size }, { stride, stride }, { padding, padding }, { dilation, dilation }, bias);
		}

		/**
		*  2D transposed convolution layer
		*
		* take input as two-dimensional *image* and applying filtering operation.
		**/
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
		void add_conv_transpose2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> stride = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> out_padding = { 0,0 }, std::vector<int> dilation = { 1,1 }, bool bias = true)
		{
			int id = conv_transpose2d.size();
			cpp_torch::LayerInOut inout;
			inout.name = "conv_transpose2d";
			inout.type = cpp_torch::LayerType::CONV_TRANSPOSE2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = { input_channels,layer[i - 1].out_[1], layer[i - 1].out_[2] };

			auto& l = register_module("conv_transpose2d" + std::to_string(id), torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(input_channels, output_channels, { kernel_size[0],kernel_size[1] }).bias(bias).padding({ padding[0],padding[1] }).stride({ stride[0],stride[1] }).dilation({ dilation[0],dilation[1] })));
			conv_transpose2d.emplace_back(l);

			inout.out_ = {
				output_channels,
				(inout.in_[1] - 1)*stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + out_padding[0] + 1,
				(inout.in_[2] - 1)*stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + out_padding[1] + 1
			};

			inout.bias = bias;
			inout.dilation = dilation;
			inout.kernel_size = kernel_size;
			inout.padding = padding;
			inout.out_padding = out_padding;
			inout.stride = stride;
			layer.emplace_back(inout);

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.ConvTranspose2d(%d,%d,%d,stride=%d,padding=%d,dilation=%d,bias=%s),\n", input_channels, output_channels, kernel_size[0], stride[0], padding[0], dilation[0], bias ? "True" : "False");
			}
			std::cout << "conv_transpose {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}
		/**
		* constructing convolutional layer
		*
		* @param input_channels  [in] input image channels (grayscale=1, rgb=3)
		* @param output_channels [in] output image channels
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param stride       [in] stride size
		* @param padding         [in] padding size
		* @param out_padding  [in] out_padding size
		* @param dilation     [in] dilation
		* @param bias         [in] whether to add a bias vector to the filter
		**/
		void add_conv_transpose2d(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0, int out_padding = 0, int dilation = 1, bool bias = true)
		{
			add_conv_transpose2d_(input_channels, output_channels, { kernel_size, kernel_size }, { stride, stride }, { padding, padding }, { out_padding, out_padding }, { dilation, dilation }, bias);
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
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		* @param dilation     [in] dilation
		**/
		void add_maxpool2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> stride = { 1,1 }, std::vector<int> padding = { 0,0 }, std::vector<int> dilation = { 1,1 })
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

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.MaxPool2d(%d,stride=%d,padding=%d,dilation=%d),\n", kernel_size[0], stride[0], padding[0], dilation[0]);
			}
			std::cout << "maxpool {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}
		/**
		* constructing max pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param stride       [in] specify the horizontal interval at which to apply the filters to the input
		* @param padding         [in] padding size
		* @param dilation     [in] specify the horizontal interval to control the spacing between the kernel points
		**/
		void add_maxpool2d(int kernel_size, int stride, int padding = 0, int dilation = 1)
		{
			const int i = layer.size();
			add_maxpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { stride, stride }, { padding,padding }, { dilation, dilation });
		}
		/**
		* constructing max pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		**/
		void add_maxpool2d(int kernel_size)
		{
			const int i = layer.size();
			add_maxpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { kernel_size, kernel_size }, { 0, 0 }, { 1, 1 });
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
		* @param stride       [in] stride size
		* @param padding      [in] padding size
		**/
		void add_avgpool2d_(int input_channels, int output_channels, std::vector<int> kernel_size = { 1,1 }, std::vector<int> stride = { 1,1 }, std::vector<int> padding = { 0,0 }, bool ceil_mode=false, bool count_include_pad = true)
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

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.AvgPool2d(%d,stride=%d,padding=%d),\n", kernel_size[0], stride[0], padding[0]);
			}
			std::cout << "avgpool {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* constructing average  pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		* @param stride       [in] specify the horizontal interval at which to apply the filters to the input
		* @param padding         [in] padding size
		**/
		void add_avgpool2d(int kernel_size, int stride = 1, int padding = 0, int dilation = 1)
		{
			const int i = layer.size();
			add_avgpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { stride, stride }, { padding,padding });
		}
		/**
		* constructing average  pooling 2D layer
		*
		* @param kernel_size     [in] window(kernel) size of convolution
		**/
		void add_avgpool2d(int kernel_size)
		{
			const int i = layer.size();
			add_avgpool2d_(layer[i - 1].out_[0], layer[i - 1].out_[0], { kernel_size, kernel_size }, { kernel_size, kernel_size }, { 0, 0 } );
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
			conv_drop.emplace_back(register_module("conv_drop" + std::to_string(id), torch::nn::Dropout2d(torch::nn::Dropout2d(rate))));
			inout.out_ = inout.in_;
			layer.emplace_back(inout);

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.Dropout2d(%f),\n", rate);
			}
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

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.Dropout(%f),\n", rate);
			}
			std::cout << "drop {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		//PIXEL_SHUFFLE
		/*
		 * This is useful for implementing efficient sub-pixel convolution with a stride of 1/r .
		 * Look at the paper: Real-Time Single Image and Video Super-Resolution 
		 * Using an Efficient Sub-Pixel Convolutional Neural Network by Shi et. al (2016) for more details.
		 *
		 * @param upscale_factor     [in] factor to increase spatial resolution by.
		 */
		void add_pixel_shuffle(int upscale_factor)
		{
			cpp_torch::LayerInOut inout;
			inout.name = "pixel_shuffle";
			inout.type = cpp_torch::LayerType::PIXEL_SHUFFLE;
			inout.id = pixel_shuffle_count++;
			inout.upscale_factor = upscale_factor;

			const int i = layer.size();
			inout.in_ = layer[i - 1].out_;
			inout.out_ = inout.in_;
			float r = sqrt(inout.in_[0]);
			inout.out_[0] = 1;
			inout.out_[1] = inout.in_[1] * r;
			inout.out_[2] = inout.in_[2] * r;

			layer.emplace_back(inout);

			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.PixelShuffle(%d),\n", upscale_factor);
			}
			std::cout << "pixel_shuffle {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
		}

		/**
		* constructing Batch Normalization layer
		* @param momentum        [in] momentum in the computation of the exponential
		* @param eos             [in] The epsilon value added for numerical stability.
		**/
		void add_bn2d(float_t momentum=0.1, float_t eps= 1e-5)
		{
			int id = bn2d.size();
			cpp_torch::LayerInOut inout;
			inout.name = "batchnorml2d";
			inout.type = cpp_torch::LayerType::BATCHNORMAL2D;
			inout.id = id;

			const int i = layer.size();
			inout.in_ = layer[i - 1].out_;
			bn2d.emplace_back(register_module("bn2d" + std::to_string(id), torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(inout.in_[0]).eps(eps).momentum(momentum))));

			inout.out_ = inout.in_;
			//printf("out %d %d %d ->", inout.outC, inout.outH, inout.outW);
			layer.emplace_back(inout);


			if (pycode_dump)
			{
				fprintf(pycode_dump, "            nn.BatchNorm2d(%d,eps=%f,momentum=%f),\n", inout.in_[0], eps, momentum);
			}
			std::cout << "bn2d {" << inout.in_ << "}->{" << inout.out_ << "}" << std::endl;
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

			inout.rnn_hidden_size = hidden_size;
			inout.rnn_seqence_length = sequence_length;
			inout.rnn_sequence_single_size = in / sequence_length;

			std::cout << layer[i - 1].out_ << std::endl;
			inout.in_ = { 1,sequence_length, inout.rnn_sequence_single_size };
			
			int id = -1;
			if (rnn_type == "rnn")
			{
				id = rnn.size();
				auto opt = torch::nn::RNNOptions(inout.rnn_sequence_single_size, inout.rnn_hidden_size);
				opt = opt.batch_first(true);

				if (activatin == "ReLU") opt = opt.nonlinearity(torch::kReLU);
				if (activatin == "TanH") opt = opt.nonlinearity(torch::kTanh);
				opt = opt.num_layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				rnn.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::RNN(opt)));
				
				if (pycode_dump)
				{
					string ac = "";
					if (activatin == "ReLU") ac = "rule";
					if (activatin == "TanH") ac = "tanh";

					fprintf(pycode_dump, "            nn.RNN(%d,%d,num_layers=%d,dropout=%f,nonlinearity='%s', batch_first=True),\n", inout.rnn_sequence_single_size, inout.rnn_hidden_size, num_layers, dropout, ac.c_str());
				}
			}
			if (rnn_type == "lstm")
			{
				id = lstm.size();
				auto opt = torch::nn::LSTMOptions(inout.rnn_sequence_single_size, inout.rnn_hidden_size);
				opt = opt.batch_first(true);
				opt = opt.num_layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				lstm.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::LSTM(opt)));
				if (pycode_dump)
				{
					fprintf(pycode_dump, "            nn.LSTM(%d,%d,num_layers=%d,dropout=%f, batch_first=True),\n", inout.rnn_sequence_single_size, inout.rnn_hidden_size, num_layers, dropout);
				}
			}
			if (rnn_type == "gru")
			{ 
				id = gru.size();
				auto opt = torch::nn::GRUOptions(inout.rnn_sequence_single_size, inout.rnn_hidden_size);
				opt = opt.batch_first(true);
				opt = opt.num_layers(num_layers);
				if (dropout > 0.0) opt = opt.dropout(dropout);

				gru.emplace_back(register_module(rnn_type + std::to_string(id), torch::nn::GRU(opt)));
				if (pycode_dump)
				{
					fprintf(pycode_dump, "            nn.GRU(%d,%d,num_layers=%d,dropout=%f, batch_first=True),\n", inout.rnn_sequence_single_size, inout.rnn_hidden_size, num_layers, dropout);
				}
			}
			if (id == -1)
			{
				throw error_exception("recurrent type error");
			}
			inout.id = id;
			//inout.out_ = { 1,sequence_length, hidden_size };
			inout.out_ = { 1, num_layers, inout.rnn_hidden_size };
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
		if (pycode_dump)								\
		{												\
			fprintf(pycode_dump, "            nn.%s(),\n", inout.name.c_str());\
		}												\
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
		if (pycode_dump)								\
		{												\
			fprintf(pycode_dump, "            nn.%s(dim=%d),\n", inout.name.c_str(),inout.dim);\
		}												\
		}

#define ACTIVATION_LAYER2( id_name ) void add_##id_name(float_t negative_slope)	\
		{												\
		cpp_torch::LayerInOut inout;					\
		inout.name = #id_name;							\
		inout.type = cpp_torch::LayerType::##id_name;	\
		inout.id = activation_count++;					\
		const int i = layer.size();						\
		inout.in_ = layer[i - 1].out_;					\
		inout.out_ = inout.in_;							\
		inout.negative_slope = negative_slope;			\
		layer.push_back(inout);							\
		if (pycode_dump)								\
		{												\
			fprintf(pycode_dump, "            nn.%s(%f,inplace=True),\n", inout.name.c_str(),negative_slope);\
		}												\
		}

		ACTIVATION_LAYER(ReLU)
		ACTIVATION_LAYER(ReLU_)
		ACTIVATION_LAYER2(LeakyReLU)
		ACTIVATION_LAYER2(LeakyReLU_)
		ACTIVATION_LAYER(SELU)
		ACTIVATION_LAYER(Sigmoid)
		ACTIVATION_LAYER(Tanh)
		ACTIVATION_LAYER1(Softmax)
		ACTIVATION_LAYER1(LogSoftmax)
		ACTIVATION_LAYER(Squeeze)

		int debug_dmp = 0;
		torch::Tensor forward(torch::Tensor x)
		{
			if (pycode_dump)
			{
				fprintf(pycode_dump, "        )\n\n");
				fprintf(pycode_dump, "    def forward(self, input):\n");
				fprintf(pycode_dump, "        if input.is_cuda and self.ngpu > 1:\n");
				fprintf(pycode_dump, "            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))\n");
				fprintf(pycode_dump, "        else:\n");
				fprintf(pycode_dump, "            output = self.main(input)\n");
				fprintf(pycode_dump, "        return output\n");
				fprintf(pycode_dump, "        \n\n");
				fclose(pycode_dump);
				if (pycode_dump_only) exit(0);
			}
			pycode_dump = NULL;

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
				if (layer[i].type == cpp_torch::LayerType::CONV_TRANSPOSE2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = conv_transpose2d[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(conv_transpose2d[layer[i].id]->name(), x);
					
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
				if (layer[i].type == cpp_torch::LayerType::BATCHNORMAL2D)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = bn2d[layer[i].id]->forward(x);
					if (debug_dmp)cpp_torch::dump_dim(bn2d[layer[i].id]->name(), x);
					continue;
				}
				if (layer[i].type == cpp_torch::LayerType::PIXEL_SHUFFLE)
				{
					x = x.view({ -1, layer[i - 1].out_[0], layer[i - 1].out_[1], layer[i - 1].out_[2] });
					x = torch::pixel_shuffle(x, layer[i].upscale_factor);
					if (debug_dmp)cpp_torch::dump_dim(bn2d[layer[i].id]->name(), x);
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
						//x = std::get<0>(lstm[layer[i].id]->forward(x));
						x = std::get<0>(std::get<1>(lstm[layer[i].id]->forward(x)));
					}else
					if (layer[i].type == cpp_torch::LayerType::GRU)
					{
						x = std::get<0>(gru[layer[i].id]->forward(x));
					}else
					if (layer[i].type == cpp_torch::LayerType::RNN)
					{
						x = std::get<0>(rnn[layer[i].id]->forward(x));
					}
					//x = x.view({ batch,  layer[i].rnn_seqence_length, -1 });
					//x = x.view({ batch,  -1, layer[i].rnn_hidden_size });
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
					case cpp_torch::LayerType::Squeeze:
						x = x.squeeze(); break;
					case cpp_torch::LayerType::ReLU_:
						x = torch::relu_(x); break;
					case cpp_torch::LayerType::ReLU:
						x = torch::relu(x); break;
					case cpp_torch::LayerType::LeakyReLU_:
						x = torch::leaky_relu_(x, layer[i].negative_slope); break;
					case cpp_torch::LayerType::LeakyReLU:
						x = torch::leaky_relu(x, layer[i].negative_slope); break;
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
