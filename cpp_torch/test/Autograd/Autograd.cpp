/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"


#define USE_CUDA

#define NUM	1000
torch::Tensor X()
{
	tiny_dnn::vec_t s;
	for (int i = 0; i < NUM; i++)
	{
		s.push_back(2.0*M_PI*i / NUM);
	}
	return cpp_torch::toTorchTensors(s);
}

torch::Tensor Sin(torch::Tensor x)
{
	return torch::sin(x);
}
torch::Tensor Cos(torch::Tensor x)
{
	return torch::cos(x);
}

/*
torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
torch::Tensor b = torch::randn({2, 2});
auto c = a + b;
c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
*/
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

	torch::Tensor& x = X().to(device);
	//torch::autograd::GradMode::set_enabled(true);
	//requires_grad=True(Specify the target of automatic differentiation with True)
	if (!x.requires_grad())
	{
		x.set_requires_grad(true);
	}

	// Build calculation graph
	torch::Tensor& sinx = Sin(x).to(device);
	torch::Tensor& y = sinx +x*x;

	// dy/dx
	torch::Tensor& cosx = Cos(x).to(device);
	torch::Tensor& yy = cosx + 2*x;

	//calculate the slope of y
	y.backward();

	// d y/dx
	torch::Tensor& dy_dx = x.grad().to(device);

	x.print();
	x.set_requires_grad(false);


	cpp_torch::dump_dim("y", y);
	FILE* fp = fopen("output.csv", "w");
	for (int i = 0; i < NUM; i++)
	{
		//fprintf(fp, "%f,%f,%f\n", 
		//	x[i].template item<float_t>(),
		//	y[i].template item<float_t>(),
		//	yy[i].template item<float_t>()
		//);
		fprintf(fp, "%f,%f,%f,%f\n", 
			x[i].template item<float_t>(),
			y[i].template item<float_t>(),
			dy_dx[i].template item<float_t>(),
			yy[i].template item<float_t>()
			);
	}
	fclose(fp);

}
