#pragma once
#include <functional>

#ifdef _LIBRARY_EXPORTS
#define _LIBRARY_EXPORTS __declspec(dllexport)
#else
#define _LIBRARY_EXPORTS __declspec(dllimport)
#endif

extern "C" _LIBRARY_EXPORTS int getBatchSize();
extern "C" _LIBRARY_EXPORTS int getNumberOfEpochs();
extern "C" _LIBRARY_EXPORTS int getSequence_length();
extern "C" _LIBRARY_EXPORTS int getYdim();
extern "C" _LIBRARY_EXPORTS int getXdim();
extern "C" _LIBRARY_EXPORTS float getScale();
extern "C" _LIBRARY_EXPORTS float getTolerance();

extern "C" _LIBRARY_EXPORTS int torch_train_init(void);
extern "C" _LIBRARY_EXPORTS void* getDevice();
extern "C" _LIBRARY_EXPORTS void* setDevice(const char* device_name);

extern "C" _LIBRARY_EXPORTS void getData(const char* filename, std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void send_train_images(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void send_train_labels(std::vector<tiny_dnn::vec_t>& data);
extern "C" _LIBRARY_EXPORTS void read_params();

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

extern "C" _LIBRARY_EXPORTS void* getNet();
extern "C" _LIBRARY_EXPORTS void delete_model();

extern "C" _LIBRARY_EXPORTS float get_loss(std::vector<tiny_dnn::vec_t>& train_images, std::vector<tiny_dnn::vec_t>& train_labels);
extern "C" _LIBRARY_EXPORTS float get_Loss();

extern "C" _LIBRARY_EXPORTS void* torch_progress_display(size_t length);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_restart(void* disp_, size_t length);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_count(void* disp_, int count);
extern "C" _LIBRARY_EXPORTS void torch_progress_display_delete(void* disp_);


extern "C" _LIBRARY_EXPORTS void torch_stop_ongoing_training();

extern "C" _LIBRARY_EXPORTS void save(const char* name);
extern "C" _LIBRARY_EXPORTS void* load(const char* name);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_model_predict(const void* nn, tiny_dnn::vec_t x);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x);




