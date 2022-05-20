/**
MIT License

Copyright (c) 2019 Sanaxen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
**/
/**
Experimental DLL

By using this DLL, the code created by tiny-dnn can be run on 
pytorch (libtorch) / C ++, so it will run on GPU (CUDA).

train-> torch_train
predict-> torch_predict
get_loss-> torch_get_loss
etc ..
**/
#pragma once
#include <functional>

#ifdef _LIBRARY_EXPORTS
#define _LIBRARY_EXPORTS __declspec(dllexport)
#else
#define _LIBRARY_EXPORTS __declspec(dllimport)
#endif

extern "C" _LIBRARY_EXPORTS int cuda_is_available();
extern "C" _LIBRARY_EXPORTS void  set_sampling(float rate);
extern "C" _LIBRARY_EXPORTS void  reset_sampling();

extern "C" _LIBRARY_EXPORTS void read_mnist_dataset(const std::string &data_dir_path);

extern "C" _LIBRARY_EXPORTS int getBatchSize();
extern "C" _LIBRARY_EXPORTS int getNumberOfEpochs();
extern "C" _LIBRARY_EXPORTS int getSequence_length();
extern "C" _LIBRARY_EXPORTS int getYdim();
extern "C" _LIBRARY_EXPORTS int getXdim();
extern "C" _LIBRARY_EXPORTS float getScale();
extern "C" _LIBRARY_EXPORTS float getTolerance();

extern "C" _LIBRARY_EXPORTS int torch_train_init(void);
extern "C" _LIBRARY_EXPORTS int torch_train_init_seed(int seed);
extern "C" _LIBRARY_EXPORTS void* torch_getDevice();
extern "C" _LIBRARY_EXPORTS void torch_setDevice(const char* device_name);
extern "C" _LIBRARY_EXPORTS void torch_setDeviceIndex(const int id);

extern "C" _LIBRARY_EXPORTS void torch_getData(const char* filename, std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void send_train_images(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void send_train_labels(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void get_train_images(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void get_train_labels(std::vector<tiny_dnn::vec_t>& data);

extern "C" _LIBRARY_EXPORTS void send_test_images(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void send_test_labels(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void get_test_images(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void get_test_labels(std::vector<tiny_dnn::vec_t>& data);

extern "C" _LIBRARY_EXPORTS void torch_params(
	int n_train_epochs_,
	int n_minibatch_,
	int input_size_,

	int n_layers_,
	float dropout_,
	int n_hidden_size_,
	int fc_hidden_size_,
	float learning_rate_,

	float clip_gradients_,
	int use_cnn_,
	int use_add_bn_,
	int use_cnn_add_bn_,
	int residual_,
	int padding_prm_,

	int classification_,
	char* weight_init_type_,
	char* activation_fnc_,
	int early_stopping_,
	char* opt_type_,
	bool batch_shuffle_,
	int shuffle_seed_,
	bool L1_loss_,
	int test_mode_
);

extern "C" _LIBRARY_EXPORTS void torch_read_params(bool train);
extern "C" _LIBRARY_EXPORTS void torch_read_train_params();
extern "C" _LIBRARY_EXPORTS void torch_read_test_params();

extern "C" _LIBRARY_EXPORTS void torch_train_post_fc(
	std::vector<tiny_dnn::vec_t>&train_images_,
	std::vector<tiny_dnn::vec_t>&train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
);

extern "C" _LIBRARY_EXPORTS void torch_train_fc(
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
);

extern "C" _LIBRARY_EXPORTS void torch_train(
	std::vector<tiny_dnn::vec_t>& train_images,
	std::vector<tiny_dnn::vec_t>& train_labels,
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_batch_enumerate,
	std::function <void(void)> on_enumerate_minibatch
);
extern "C" _LIBRARY_EXPORTS void Train(
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
);

extern "C" _LIBRARY_EXPORTS int torch_train_custom(
	std::string define_layers_file_name,
	std::vector<tiny_dnn::vec_t>& train_images_,
	std::vector<tiny_dnn::vec_t>& train_labels_,
	int n_minibatch,
	int n_train_epochs,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
);

extern "C" _LIBRARY_EXPORTS void* torch_getNet();
extern "C" _LIBRARY_EXPORTS void torch_delete_model();
extern "C" _LIBRARY_EXPORTS void torch_delete_load_model(void* n);

extern "C" _LIBRARY_EXPORTS float torch_get_loss(std::vector<tiny_dnn::vec_t>& train_images, std::vector<tiny_dnn::vec_t>& train_labels, int batch);
extern "C" _LIBRARY_EXPORTS float torch_get_Loss(int batch);
extern "C" _LIBRARY_EXPORTS float torch_get_loss_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch);
extern "C" _LIBRARY_EXPORTS float torch_get_train_loss();

extern  _LIBRARY_EXPORTS tiny_dnn::result torch_get_accuracy_nn(void* nn, std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_, int batch);


extern "C" _LIBRARY_EXPORTS void* torch_progress_display(size_t length);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_restart(void* disp_, size_t length);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_count(void* disp_, int count);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_delete(void* disp_);


extern "C" _LIBRARY_EXPORTS void torch_stop_ongoing_training();

extern "C" _LIBRARY_EXPORTS void torch_save(const char* name);
extern "C" _LIBRARY_EXPORTS void torch_load(const char* name);
extern "C" _LIBRARY_EXPORTS void torch_save_nn(void* nn, const char* name);
extern "C" _LIBRARY_EXPORTS void* torch_load_new(const char* name);

extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_model_predict(const void* nn, tiny_dnn::vec_t x);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x);
extern	_LIBRARY_EXPORTS std::vector<tiny_dnn::vec_t> torch_model_predict_batch(const void* nn, std::vector<tiny_dnn::vec_t>& x, int batch);
extern  _LIBRARY_EXPORTS std::vector<tiny_dnn::vec_t> torch_predict_batch(std::vector<tiny_dnn::vec_t>& x, int bacth);

extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_post_predict(tiny_dnn::vec_t x);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_invpost_predict(tiny_dnn::vec_t x);

extern "C" _LIBRARY_EXPORTS void torch_stop_ongoing_training();
extern "C" _LIBRARY_EXPORTS void state_reset(std::string& rnn_type, void* nn);




