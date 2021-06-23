/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#if 0
#include "cpp_torch.h"
#else
#include <torch/csrc/api/include/torch/nn/init.h>
#include <torch/enum.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
#endif

#define USE_LIBTORCH_190
#include "libtorch_link_libs.hpp"

#include <torch/script.h> // One-stop header.

#define USE_CUDA

auto main() -> int {


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


	torch::jit::script::Module module = torch::jit::load("data/script_module.pt");
	
	std::cout << "start" << '\n';
	// LibTorch‘¤‚Å—pˆÓ‚µ‚½Tensor‚ð“ü—Í‚µ‚ÄŽÀs

	at::TensorOptions options(at::kFloat);

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({ 1, 96 }).to(device));
	std::cout << "Create a vector of inputs OK" << '\n';

	// Execute the model and turn its output into a tensor.
	at::Tensor output = module.forward(inputs).toTensor();
	std::cout << "output.tensor OK" << '\n';

	std::cout << "ok" << '\n';
	return 0;
}
