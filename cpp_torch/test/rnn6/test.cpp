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
#include "tiny_dnn2libtorch_dll.h"
#include <iostream>

std::vector<tiny_dnn::vec_t> train_labels_, test_labels_;
std::vector<tiny_dnn::vec_t> train_images_, test_images_;
FILE* fp_error_loss2 = NULL;
bool stop_ongoing_training_flag = false;

void test_(void* nn)
{
	void* nn2 = torch_load_new("best_model.pt");

	int y_dim = getYdim();
	float scale = getScale();

	std::vector<tiny_dnn::vec_t> predict;

	predict.resize(train_images_.size());
#pragma omp parallel for
	for (int i = 0; i < train_images_.size() - getSequence_length() - train_labels_[0].size(); i++)
	{
		tiny_dnn::vec_t next_y = torch_model_predict(nn2, train_images_[i]);
		//output sequence_length 
		for (int j = 0; j < next_y.size(); j++)
		{
			tiny_dnn::vec_t yy(y_dim);
			tiny_dnn::vec_t y(y_dim);
			for (int k = 0; k < y_dim; k++)
			{
				yy[k] = next_y[y_dim*j + k];
			}
			predict[i + getSequence_length() + j] = yy;
		}
	}

	FILE* fp = fopen("test.dat", "w");
	float t = 0;
	float dt = 1.0;
	for (int i = 0; i < getSequence_length(); i++)
	{
		fprintf(fp, "%f", t);
		for (int k = 0; k < y_dim; k++)
		{
			fprintf(fp, " %f NaN", train_images_[i][k]);
		}
		fprintf(fp, "\n");
		t += dt;
	}
	fclose(fp);

	fp = fopen("predict1.dat", "w");
	for (int i = getSequence_length(); i < train_images_.size() - getSequence_length() - train_labels_[0].size(); i++)
	{
		fprintf(fp, "%f", t);
		for (int k = 0; k < y_dim; k++)
		{
			fprintf(fp, " %f %f", train_images_[i][k], predict[i][k]);
		}
		fprintf(fp, "\n");
		t += dt;
	}
	fclose(fp);

	fp = fopen("predict2.dat", "w");
	int sz = train_images_.size();

	for (int i = train_images_.size() - getSequence_length() - train_labels_[0].size(); i < sz - 1; i++)
	{
		fprintf(fp, "%f", t);
		for (int k = 0; k < y_dim; k++)
		{
			fprintf(fp, " %f %f", train_images_[i][k], predict[i][k]);
		}
		fprintf(fp, "\n");
		t += dt;
	}
	fclose(fp);

	fp = fopen("prophecy.dat", "w");
	fprintf(fp, "\n");
	fclose(fp);

	torch_delete_load_model(nn2);
}

int main()
{
	bool isTrain = true;
	if (isTrain)
	{
		torch_read_train_params();
		torch_getData("train_images_tr.csv", train_images_);
		torch_getData("train_labels_tr.csv", train_labels_);
	}
	else
	{
		torch_read_test_params();
		torch_getData("train_images_ts.csv", train_images_);
		torch_getData("train_labels_ts.csv", train_labels_);
	}

	
	torch_train_init();
	torch_setDevice("cpu");


	int epoch = 1;
	void* disp = torch_progress_display(train_images_.size());
	tiny_dnn::timer t;
	float min_loss = std::numeric_limits<float>::max();


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
				fprintf(fp_error_loss2, "%.10f %.10f %.4f\n", loss, min_loss, getTolerance());
				fflush(fp_error_loss2);
			}
			if (loss < min_loss)
			{
				min_loss = loss;
				torch_save("best_model.pt");
			}
			test_(torch_getNet());
		}
		++epoch;

		if (epoch <= getNumberOfEpochs())
		{
			torch_progress_display_restart(disp, train_images_.size());
		}
		t.restart();
	};

	int batch = 1;
	auto on_enumerate_minibatch = [&]() {
		if (batch == 1 && epoch == 1)
		{
			torch_progress_display_restart(disp, train_images_.size());
		}
		torch_progress_display_count(disp, getBatchSize());
		batch++;
	};

	int n_epoch = 30;
	int minbatch = 128;
	if (1)
	{
		char regression[10] = { '\0' };
		torch_train_fc(
			train_images_,
			train_labels_,
			minbatch,
			n_epoch,
			regression,
			on_enumerate_minibatch, on_enumerate_epoch);
		//torch_train(
		//	train_images_,
		//	train_labels_,
		//	minbatch,
		//	n_epoch,
		//	on_enumerate_minibatch, on_enumerate_epoch);
	}
	else
	{
		send_train_images(train_images_);
		send_train_labels(train_labels_);

		Train(
			minbatch,
			n_epoch,
			on_enumerate_minibatch, on_enumerate_epoch);
	}
	if (!isTrain)
	{
		test_(torch_getNet());
	}
	torch_delete_model();
	return 0;
}