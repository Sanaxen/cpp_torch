/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"
#include "../test/include/seqence_data.h"

#define TEST		//cpp_torch
#define USE_CUDA

#define ZERO_TOL 0.0001

const char* kDataRoot = "./data";

// The batch size for training.
int64_t kTrainBatchSize = 32;

// The batch size for testing.
int64_t kTestBatchSize = 10;

// The number of epochs to train.
int64_t kNumberOfEpochs = 2000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

std::string solver_name = std::string ("Adam");

int explanatory_variable = 0;
int sequence_length = 21;
int out_sequence_length = 1;
int hidden_size = 64;
int x_dim = 1;
int y_dim = 3;
int n_rnn = 1;
int normalize = 1;
float test_size = 0.3f;
std::string input = "sample.csv";

#ifdef USE_CUDA
int use_gpu = 1;
#else
int use_gpu = 0;
#endif


void comannd_line_opt(int argc, char** argv)
{
	for (int i = 1; i < argc; i++)
	{
		if (std::string(argv[i]) == "--x_dim")
		{
			x_dim = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--y_dim")
		{
			y_dim = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--hidden_size")
		{
			hidden_size = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--sequence_length")
		{
			sequence_length = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--out_sequence_length")
		{
			out_sequence_length = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--kNumberOfEpochs")
		{
			kNumberOfEpochs = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--kTestBatchSize")
		{
			kTestBatchSize = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--kTrainBatchSize")
		{
			kTrainBatchSize = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--hidden_size")
		{
			hidden_size = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--hidden_size")
		{
			hidden_size = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--n_rnn")
		{
			n_rnn = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--normalize")
		{
			normalize = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--test")
		{
			test_size = atof(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--input")
		{
			input = std::string(argv[i + 1]);
			kDataRoot = ".";
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--use_gpu")
		{
			use_gpu = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--solver")
		{
			solver_name = std::string(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--explanatory_variable"|| std::string(argv[i]) == "--ex_var")
		{
			explanatory_variable = atoi(argv[i + 1]);
			i++;
			continue;
		}
		
	}
}

#define RNN_LAYERS_OPT
struct NetImpl : torch::nn::Module {
	NetImpl()
		: 
		lstm({ nullptr }),
		lstm2({ nullptr }),
		lstm3({ nullptr }),
		fc1(sequence_length*hidden_size, 128),
		fc2(128, 128),
		fc3(128, 50),
		fc4(50, y_dim*out_sequence_length)
	{
#ifdef RNN_LAYERS_OPT
		auto opt = torch::nn::LSTMOptions(y_dim, hidden_size).layers(n_rnn);
		opt = opt.batch_first(true);
		lstm = torch::nn::LSTM(opt);
		lstm.get()->options.batch_first(true);
#else
		//Results are incorrect(?)
		auto opt = torch::nn::LSTMOptions(y_dim, hidden_size);
		opt = opt.batch_first(true);

		lstm = torch::nn::LSTM(opt);
		lstm.get()->options.batch_first(true);

		auto opt2 = torch::nn::LSTMOptions(hidden_size, hidden_size);
		opt2 = opt2.batch_first(true);

		lstm2 = torch::nn::LSTM(opt2);
		lstm2.get()->options.batch_first(true);

		auto opt3 = torch::nn::LSTMOptions(hidden_size, hidden_size);
		opt3 = opt3.batch_first(true);

		lstm3 = torch::nn::LSTM(opt3);
		lstm3.get()->options.batch_first(true);
#endif

		fc1.get()->options.with_bias(false);
		fc2.get()->options.with_bias(false);
		fc3.get()->options.with_bias(false);
		fc4.get()->options.with_bias(false);

#ifdef RNN_LAYERS_OPT
		register_module("lstm", lstm);
#else
		register_module("lstm", lstm);
		register_module("lstm2", lstm);
		register_module("lstm3", lstm);
#endif
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("fc3", fc3);
		register_module("fc4", fc4);
	}

	torch::Tensor forward(torch::Tensor x) {

		int batch = x.sizes()[0];
		//cpp_torch::dump_dim("X", x);
		x = x.view({ batch, -1, y_dim});
		//cpp_torch::dump_dim("X", x);
#ifdef RNN_LAYERS_OPT
		x = lstm->forward(x).output;
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(x);
#else
		x = lstm->forward(x).output;
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(x);
		x = lstm2->forward(x).output;
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(x);
		x = lstm3->forward(x).output;
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(x);
#endif

		//cpp_torch::dump_dim("X", x);
		x = x.view({ batch,  -1});
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(fc1(x));
		//cpp_torch::dump_dim("X", x);
		x = torch::tanh(fc2(x));
		x = torch::tanh(fc3(x));
		x = torch::tanh(fc4(x));

		return x;
	}

	torch::nn::LSTM lstm;
	torch::nn::LSTM lstm2;
	torch::nn::LSTM lstm3;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
	torch::nn::Linear fc4;
};
TORCH_MODULE(Net); // creates module holder for NetImpl

std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;


void read_rnn_dataset(cpp_torch::test::SeqenceData& seqence_data, const std::string &data_dir_path, const std::string& filename = "sample.csv", int normalize_type = 1, float testsize = 0.3)
{
	seqence_data.normalize_type = normalize_type;
	seqence_data.testsize = testsize;

	seqence_data.Initialize(data_dir_path, filename);

	seqence_data.get_train_data(train_images, train_labels);
	seqence_data.get_test_data(test_images, test_labels);
}

void learning_and_test_rnn_dataset(cpp_torch::test::SeqenceData& seqence_data, torch::Device device)
{

#ifndef TEST
	Net model;
#endif

#ifdef TEST
	cpp_torch::Net model;

	model.get()->device = device;
	model.get()->setInput(1, 1, (explanatory_variable) ? x_dim : 0 *sequence_length);

#ifdef RNN_LAYERS_OPT
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size, n_rnn);
#else
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size);
	model.get()->add_Tanh();
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size);
	model.get()->add_Tanh();
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size);
#endif
	model.get()->add_Tanh();
	model.get()->add_fc(128);
	model.get()->add_Tanh();
	model.get()->add_fc(128);
	model.get()->add_Tanh();
	model.get()->add_fc(y_dim*out_sequence_length);
#endif


#ifndef TEST
	cpp_torch::network_torch<Net> nn(model, device);
#else
	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);
#endif

	nn.input_dim(1, 1, (y_dim + (explanatory_variable)?x_dim:0)*sequence_length);
	nn.output_dim(1, 1, y_dim*out_sequence_length);
	nn.classification = false;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;


	torch::optim::Optimizer* optimizer = nullptr;

	const std::string& solver_name = "SGD";

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(0.0001));

	if (solver_name == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (solver_name == "Adam")
	{
		optimizer = &optimizerAdam;
	}

	FILE* lossfp = fopen("loss.dat", "w");
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (epoch % kLogInterval == 0)
		{
			float loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
			std::cout << "loss :" << loss << std::endl;
			fprintf(lossfp, "%f\n", loss);
			fflush(lossfp);

			seqence_data.sequence_test(nn);
			if (loss < ZERO_TOL)
			{
				nn.stop_ongoing_training();
			}
		}
		++epoch;

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

	nn.set_tolerance(0.01, 0.0001, 5);

	// train
	nn.fit(optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	fclose(lossfp);

	float_t loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
	printf("loss:%f\n", loss);
	std::vector<int>& res = nn.test_tolerance(test_images, test_labels);
	{
		cpp_torch::textColor color("YELLOW");
		cpp_torch::print_ConfusionMatrix(res, nn.get_tolerance());
	}

	nn.test(test_images, test_labels, kTestBatchSize);


}


int main(int argc, char** argv)
{

	comannd_line_opt(argc, argv);

	torch::manual_seed(1);

	torch::DeviceType device_type;

	if (use_gpu)
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
	torch::Device device(device_type);

	cpp_torch::test::SeqenceData seqence_data;
	seqence_data.x_dim = x_dim;
	seqence_data.y_dim = y_dim;
	seqence_data.sequence_length = sequence_length;
	seqence_data.out_sequence_length = out_sequence_length;
	seqence_data.n_minibatch = kTrainBatchSize;

	if (explanatory_variable)
	{
		// true to add x to the explanatory variable
		seqence_data.add_explanatory_variable = true;
	}
	read_rnn_dataset(seqence_data, std::string(kDataRoot), input, normalize, test_size);

	learning_and_test_rnn_dataset(seqence_data, device);
}
