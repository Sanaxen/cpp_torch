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
int64_t kLogInterval = 10;

float stop_min_loss = 0;
std::string solver_name = std::string("Adam");

int explanatory_variable = 0;
int sequence_length = 21;
int out_sequence_length = 1;
int hidden_size = 64;
int x_dim = 1;
int y_dim = 3;
int n_rnn = 1;
int normalize = 1;
float test_size = 0.3f;
float lr = 0.0001;
float moment = 0.5;
int nfc = -1;
int fc_hidden_size = 128;
int classification = 0;
int prophecy = 0;
std::string input = "/sample.csv";

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
		if (std::string(argv[i]) == "--explanatory_variable" || std::string(argv[i]) == "--ex_var")
		{
			explanatory_variable = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--lr")
		{
			lr = atof(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--moment")
		{
			moment = atof(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--nfc")
		{
			nfc = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--fc_hidden_size")
		{
			fc_hidden_size = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--classification")
		{
			classification = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--prophecy")
		{
			prophecy = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--kLogInterval")
		{
			kLogInterval = atoi(argv[i + 1]);
			i++;
			continue;
		}
		if (std::string(argv[i]) == "--stop_min_loss")
		{
			stop_min_loss = atof(argv[i + 1]);
			i++;
			continue;
		}
	}
	printf("--x_dim					:%d\n", x_dim);
	printf("--y_dim					:%d\n", y_dim);
	printf("--hidden_size			:%d\n", hidden_size);
	printf("--sequence_length		:%d\n", sequence_length);
	printf("--out_sequence_length	:%d\n", out_sequence_length);
	printf("--kNumberOfEpochs		:%d\n", kNumberOfEpochs);
	printf("--kTestBatchSize		:%d\n", kTestBatchSize);
	printf("--kTrainBatchSize		:%d\n", kTrainBatchSize);
	printf("--n_rnn					:%d\n", n_rnn);
	printf("--normalize				:%d\n", normalize);
	printf("--input					:%s\n", input.c_str());
	printf("--use_gpu				:%d\n", use_gpu);
	printf("--solver				:%s\n", solver_name.c_str());
	printf("--explanatory_variable	:%d\n", explanatory_variable);
	printf("--lr					:%f\n", lr);
	printf("--moment				:%f\n", moment);
	printf("--nfc					:%d\n", nfc);
	printf("--fc_hidden_size		:%d\n", fc_hidden_size);
	printf("--classification		:%d\n", classification);
	printf("--prophecy				:%d\n", prophecy);
	printf("--kLogInterval			:%d\n", kLogInterval);
	printf("--stop_min_loss			:%f\n", stop_min_loss);
}

#define RNN_LAYERS_OPT

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
	cpp_torch::Net model;

	model.get()->device = device;
	
	const int z_dim = y_dim + ((explanatory_variable != 0) ? x_dim : 0);
	model.get()->setInput(1, 1, z_dim*sequence_length);

	model.get()->add_fc(z_dim*sequence_length);
	model.get()->add_Tanh();

	if (n_rnn <= 0) n_rnn = 1;
#ifdef RNN_LAYERS_OPT
	model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size, n_rnn, 0.1);
#else
	for (int i = 0; i < n_rnn; i++)
	{
		model.get()->add_recurrent(std::string("lstm"), sequence_length, hidden_size);
		model.get()->add_Tanh();
	}
#endif
	model.get()->add_Tanh();
	if (nfc == -1)
	{
		model.get()->add_fc(fc_hidden_size);
		model.get()->add_Tanh();
		model.get()->add_fc(fc_hidden_size);
		model.get()->add_Tanh();
	}
	else
	{
		size_t sz = fc_hidden_size;
		if (sz < (y_dim*out_sequence_length) * 10)
		{
			sz = (y_dim*out_sequence_length) * 10;
		}
		for (int i = 0; i < nfc; i++) {
			model.get()->add_fc(sz, false);
			model.get()->add_Tanh();
		}
		
		sz = (y_dim*out_sequence_length) * 10;
		model.get()->add_fc(sz, false);
		model.get()->add_Tanh();
	}
	model.get()->add_fc(y_dim*out_sequence_length);
	if (classification > 1)
	{
		model.get()->add_LogSoftmax(1);
	}

	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);

	nn.input_dim(1, 1, z_dim*sequence_length);
	nn.output_dim(1, 1, y_dim*out_sequence_length);
	nn.classification = false;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	//tiny_dnn::progress_display disp(train_images.size());
	cpp_torch::progress_display disp(train_images.size());

	tiny_dnn::timer t;


	torch::optim::Optimizer* optimizer = nullptr;

	auto optimizerSGD = torch::optim::SGD(
		model.get()->parameters(), torch::optim::SGDOptions(lr).momentum(moment));

	auto optimizerAdam =
		torch::optim::Adam(model.get()->parameters(),
			torch::optim::AdamOptions(lr));

	if (solver_name == "SGD")
	{
		optimizer = &optimizerSGD;
	}
	if (solver_name == "Adam")
	{
		optimizer = &optimizerAdam;
	}

	float min_loss = 9999999.0;

	FILE* lossfp = fopen("loss.dat", "w");
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (epoch % kLogInterval == 0)
		{
			float loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
			std::cout << "loss :" << loss << " min_loss :" << min_loss << std::endl;
			fprintf(lossfp, "%f\n", loss);
			fflush(lossfp);

			seqence_data.sequence_test(nn);
			if (loss < ZERO_TOL || loss < stop_min_loss)
			{
				nn.stop_ongoing_training();
			}
			if (loss < min_loss)
			{
				min_loss = loss;
				nn.save(std::string("best_model.pt"));
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
	//printf("loss:%f\n", loss);
	std::vector<int>& res = nn.test_tolerance(test_images, test_labels);
	{
		cpp_torch::textColor color("YELLOW");
		cpp_torch::print_ConfusionMatrix(res, nn.get_tolerance());
	}

	nn.test(test_images, test_labels, kTestBatchSize);
	{
		cpp_torch::Net bakup_model;
		cpp_torch::network_torch<cpp_torch::Net> nn2(bakup_model, device);
		nn2 = nn;

		nn2.load(std::string("best_model.pt"));
		printf("best_model(loss):%f\n", min_loss);
		seqence_data.sequence_test(nn2);

		nn2.test(test_images, test_labels, kTestBatchSize);

		tiny_dnn::result res2 = nn2.test(test_images, test_labels);
		std::vector<int>& res = nn2.test_tolerance(test_images, test_labels);
		{
			cpp_torch::textColor color("BLUE");
			cpp_torch::print_ConfusionMatrix(res, nn2.get_tolerance());
		}
	}
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
	seqence_data.prophecy = prophecy;

	if (explanatory_variable)
	{
		// true to add x to the explanatory variable
		seqence_data.add_explanatory_variable = true;
	}
	read_rnn_dataset(seqence_data, std::string(kDataRoot), input, normalize, test_size);

	learning_and_test_rnn_dataset(seqence_data, device);
}
