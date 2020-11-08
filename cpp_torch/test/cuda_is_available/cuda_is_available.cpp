/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "cpp_torch.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


#define USE_CUDA

auto main() -> int {

	std::string msg = "";
	if (torch::cuda::is_available()) {
		msg = "CUDA available!";
		std::cout << msg << std::endl;
		return 0;
	}
	else
	{
		msg = "ERROR:is not available\n";
		msg += "Calculations using the GPU are not possible with the hardware you are currently using.\n";
		msg += "Please update the GPU driver or install CUDA.";
		std::cout << msg << std::endl;

		FILE* fp = fopen("cuda_is_available.log", "w");
		if (fp)
		{
			fprintf(fp, "%s", msg.c_str());
			fclose(fp);
			return 0;
		}
	}

	return -1;
}
