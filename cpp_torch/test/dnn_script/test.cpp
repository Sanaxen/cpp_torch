#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <chrono>

namespace tiny_dnn {
	typedef float float_t;
	typedef size_t label_t;
	typedef std::vector<float_t> vec_t;
	typedef std::vector<vec_t> tensor_t;

	class timer {
	public:
		timer() : t1(std::chrono::high_resolution_clock::now()) {}
		float_t elapsed() {
			return std::chrono::duration_cast<std::chrono::duration<float_t>>(
				std::chrono::high_resolution_clock::now() - t1)
				.count();
		}
		void restart() { t1 = std::chrono::high_resolution_clock::now(); }
		void start() { t1 = std::chrono::high_resolution_clock::now(); }
		void stop() { t2 = std::chrono::high_resolution_clock::now(); }
		float_t total() {
			stop();
			return std::chrono::duration_cast<std::chrono::duration<float_t>>(t2 - t1)
				.count();
		}
		~timer() {}

	private:
		std::chrono::high_resolution_clock::time_point t1, t2;
	};
};
#include "utils.h"
#include "tiny_dnn2libtorch_dnn_scriptdll.h"
#include <iostream>

std::vector<tiny_dnn::vec_t> train_labels_, test_labels_;
std::vector<tiny_dnn::vec_t> train_images_, test_images_;
bool stop_ongoing_training_flag = false;

void test_test(void* nn)
{
	void* nn2 = torch_load_new("best_model.pt");

	float scale = getScale();

	std::vector<tiny_dnn::vec_t> predict;

	predict.resize(test_images_.size());
#pragma omp parallel for
	for (int i = 0; i < test_images_.size(); i++)
	{
		predict[i] = torch_model_predict(nn2, test_images_[i]);
	}
	tiny_dnn::result result = torch_get_accuracy_nn(nn2, test_images_, test_labels_, getBatchSize());

	result.print_summary(std::cout);

	torch_delete_load_model(nn2);
}
void test_train(void* nn)
{
	void* nn2 = torch_load_new("best_model.pt");

	float scale = getScale();
	printf("scale=%f\n", scale);

	std::vector<tiny_dnn::vec_t> predict;

	predict.resize(train_images_.size());
#pragma omp parallel for
	for (int i = 0; i < train_images_.size(); i++)
	{
		predict[i] = torch_model_predict(nn2, train_images_[i]);
	}

	tiny_dnn::result result = torch_get_accuracy_nn(nn2, train_images_, train_labels_, getBatchSize());

	result.print_summary(std::cout);

	torch_delete_load_model(nn2);
}


int main()
{
	bool MNIST = true;
	bool isTrain = true;
	
	if (MNIST)
	{
		read_mnist_dataset("./data");
	}

	if (isTrain)
	{
		if (MNIST)
		{
			get_train_images(train_images_);
			get_train_labels(train_labels_);
		}
		else
		{
			torch_getData("train_images_tr.csv", train_images_);
			torch_getData("train_labels_tr.csv", train_labels_);
		}
	}
	else
	{
		if (MNIST)
		{
			get_train_images(test_images_);
			get_train_labels(test_labels_);
		}
		else
		{
			torch_getData("test_images_ts.csv", test_images_);
			torch_getData("test_labels_ts.csv", test_labels_);
		}
	}

	torch_train_init();
	torch_setDevice("cpu");


	int epoch = 1;
	int batch = 1;
	void* disp = NULL;
	tiny_dnn::timer t;
	float min_loss = std::numeric_limits<float>::max();

	FILE* fp_error_loss2 = NULL;

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << getNumberOfEpochs() << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;

		if (stop_ongoing_training_flag)
		{
			torch_stop_ongoing_training();
		}

		if (epoch % 10 == 0)
		{
			float loss = torch_get_Loss(getBatchSize());
			std::cout << "loss :" << loss << " min_loss :" << min_loss << std::endl;

			if (!fp_error_loss2 && min_loss < std::numeric_limits<float>::max())
			{
				fp_error_loss2 = fopen("error_loss.dat", "w");
			}
			if (fp_error_loss2)
			{
				fprintf(fp_error_loss2, "%.10f %.10f 0\n", loss, min_loss);
				fflush(fp_error_loss2);
			}
			if (loss < min_loss)
			{
				min_loss = loss;
				torch_save("best_model.pt");
			}
			test_train(torch_getNet());
		}
		++epoch;

		if (epoch <= getNumberOfEpochs())
		{
			torch_progress_display_restart(disp, train_images_.size());
		}
		t.restart();
		batch = 1;
	};

	auto on_enumerate_minibatch = [&]() {
		if (disp == NULL)
		{
			disp = torch_progress_display(train_images_.size());
		}
		//printf("              ");
		//printf("\r%d/%d %.2f\r", getBatchSize()*batch, train_images_.size(), (double)(getBatchSize()*batch)/(double)train_images_.size());
		torch_progress_display_count(disp, getBatchSize());
		batch++;
	};

	int n_epoch = 10;
	int minbatch = 64;
	if (1)
	{
		std::string define_layers_file_name = "layer.txt";
		torch_train_custom(
			define_layers_file_name,
			train_images_,
			train_labels_,
			minbatch,
			n_epoch,
			on_enumerate_minibatch, on_enumerate_epoch);
	}
	else
	{
		test_test(torch_getNet());
	}
	torch_delete_model();
	return 0;
}