/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"
#include "third_party/word/tokenizer.h"
#include "third_party/word/word_embed.h"

//#define USE_CUDA

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 10;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const int sequence_length = 32;
const int out_sequence_length = 32;
const int hidden_size = 400;
//const int h_size = 400;
const int vocab_size = 10000;
const int embed_size = 200;

int dim_sequence;
int dim_out_sequence;

WordEmbed *load_data(const std::string filename, const int vocab_size, bool tokenize, bool addEOS) {


	std::ifstream reading_file(filename, std::ios::in);

	std::string reading_line_buffer;


	vector<string> sequences;
	while (!reading_file.eof()) {
		// read by line
		std::getline(reading_file, reading_line_buffer);

		sequences.push_back(reading_line_buffer);
	}

	WordEmbed *wd = new WordEmbed(vocab_size);

	wd->addSentences(sequences, tokenize, addEOS);

	return wd;
}

#define TEST

struct EncoderRNNImpl : torch::nn::Module {
	EncoderRNNImpl()
		: 
	embed_ja(torch::nn::Linear(embed_size, vocab_size)),
	embed_en(torch::nn::Linear(embed_size, vocab_size)),
	lstm_ja({ nullptr }),
	lstm_en({ nullptr })
	{
		auto opt = torch::nn::LSTMOptions(hidden_size, embed_size);
		opt = opt.batch_first(true);

		lstm_ja = torch::nn::LSTM(opt);
		lstm_ja.get()->options.batch_first(true);
		register_module("embed_ja", embed_ja);

		opt = torch::nn::LSTMOptions(hidden_size, embed_size);
		opt = opt.batch_first(true);

		lstm_en = torch::nn::LSTM(opt);
		lstm_en.get()->options.batch_first(true);
		register_module("embed_en", embed_en);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = embed_ja->forward(x);
		x = torch::tanh(x);
		auto y = lstm_ja->forward(x, {});
		x = std::get<0>(y);
		auto z = lstm_en->forward(x, std::get<1>(y));
		x = embed_en->forward(std::get<0>(z));
		return x;
	}

	torch::nn::Linear embed_ja;
	torch::nn::Linear embed_en;
	torch::nn::LSTM lstm_ja;
	torch::nn::LSTM lstm_en;
};
TORCH_MODULE(EncoderRNN); // creates module holder for NetImpl


// load MNIST dataset
std::vector<tiny_dnn::vec_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;


void read_seq2seq_dataset(const std::string &data_dir_path)
{
	WordEmbed* wd_ja = load_data(data_dir_path + "/tanaka_corpus_j_10000", vocab_size, false, true);
	WordEmbed* wd_en = load_data(data_dir_path + "/tanaka_corpus_e_10000", vocab_size, false, true);
	
	wd_ja->paddingAll(vocab_size);
	wd_en->paddingAll(vocab_size);
	std::vector<std::vector<int>> seqs_ids_ja = wd_ja->getSequencesIds();
	std::vector<std::vector<int>> seqs_ids_en = wd_en->getSequencesIds();

	if (seqs_ids_ja.size() != seqs_ids_en.size()) {
		std::cout << "no match seq numbers:" << "ja:" << seqs_ids_ja.size() << " en:" << seqs_ids_en.size() << endl;
		exit(1);
	}
	seqs_ids_ja.resize(10);
	seqs_ids_en.resize(10);

	std::cout << "ja word_count:" << wd_ja->getWordCount() << std::endl;
	std::cout << "en word_count:" << wd_en->getWordCount() << std::endl;

	dim_sequence = wd_ja->getWordCount();
	dim_out_sequence = wd_en->getWordCount();
	int train_size = seqs_ids_en.size()*0.8;
	int test_size = seqs_ids_en.size() - train_size;

	for (int i = 0; i < train_size; i++)
	{
		std::vector<float> dd;
		for (int j = 0; j < seqs_ids_en[i].size(); j++)
		{
			std::vector<float> d((wd_en->getWordCount() + 5),0);
			d[seqs_ids_en[i][j]] = 1.0;

			dd.resize(dd.size() + d.size());
			dd.insert(dd.end(), d.begin(), d.end()); // 連結
		}
		train_images.push_back(dd);
	}

	for (int i = 0; i < train_size; i++)
	{
		std::vector<float> dd;
		for (int j = 0; j < seqs_ids_ja[i].size(); j++)
		{
			std::vector<float> d((wd_ja->getWordCount() + 5), 0);
			d[seqs_ids_ja[i][j]] = 1.0;

			dd.resize(dd.size() + d.size());
			dd.insert(dd.end(), d.begin(), d.end()); // 連結
		}
		train_labels.push_back(dd);
	}
	for (int i = train_size; i < seqs_ids_en.size(); i++)
	{
		std::vector<float> dd;
		for (int j = 0; j < seqs_ids_en[i].size(); j++)
		{
			std::vector<float> d((wd_en->getWordCount() + 5), 0);
			d[seqs_ids_en[i][j]] = 1.0;

			dd.resize(dd.size() + d.size());
			dd.insert(dd.end(), d.begin(), d.end()); // 連結
		}
		test_images.push_back(dd);
	}

	for (int i = train_size; i < seqs_ids_en.size(); i++)
	{
		std::vector<float> dd;
		for (int j = 0; j < seqs_ids_ja[i].size(); j++)
		{
			std::vector<float> d((wd_ja->getWordCount() + 5), 0);
			d[seqs_ids_ja[i][j]] = 1.0;

			dd.resize(dd.size() + d.size());
			dd.insert(dd.end(), d.begin(), d.end()); // 連結
		}
		test_labels.push_back(dd);
	}
}

void learning_and_test_seq2seq_dataset(torch::Device device)
{
	EncoderRNN model;
	//cpp_torch::Net model;
//
//	model.get()->setInput(1, 1, train_images[0].size());
//	model.get()->add_fc(train_images[0].size());
//	model.get()->add_Tanh();
//	model.get()->add_fc(train_images[0].size());
//	model.get()->add_Tanh();
//	model.get()->add_fc(train_images[0].size());
//	model.get()->add_Tanh();
//	model.get()->add_fc(train_labels[0].size());
//
//
//#ifndef TEST
	cpp_torch::network_torch<EncoderRNN> nn(model, device);
//#else
//	cpp_torch::network_torch<cpp_torch::Net> nn(model, device);
//#endif

	nn.input_dim(1, dim_sequence, sequence_length*hidden_size);
	nn.output_dim(1, dim_out_sequence, out_sequence_length*hidden_size);
	//nn.classification = true;
	nn.batch_shuffle = false;

	std::cout << "start training" << std::endl;

	cpp_torch::progress_display disp(train_images.size());
	tiny_dnn::timer t;


	torch::optim::SGD optimizer(
		model.get()->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
	//auto optimizer =
	//	torch::optim::Adam(model.get()->parameters(),
	//		torch::optim::AdamOptions(0.01));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << kNumberOfEpochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (epoch % kLogInterval == 0)
		{
			tiny_dnn::result res = nn.test(test_images, test_labels);
			std::cout << res.num_success << "/" << res.num_total << std::endl;
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

	//progress.start();
	// train
	nn.fit(&optimizer, train_images, train_labels, kTrainBatchSize,
		kNumberOfEpochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	float_t loss = nn.get_loss(train_images, train_labels, kTestBatchSize);
	printf("loss:%f\n", loss);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	{
		cpp_torch::textColor color("YELLOW");
		cpp_torch::print_ConfusionMatrix(res);
	}

	nn.test(test_images, test_labels, kTestBatchSize);

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
	std::string dir = std::string(kDataRoot) + std::string("/");

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

	read_seq2seq_dataset(std::string(kDataRoot));

	learning_and_test_seq2seq_dataset(device);
}
