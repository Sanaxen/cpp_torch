/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
//#include <torch/torch.h>

//#include <cstddef>
//#include <cstdio>
//#include <iostream>
#include <string>
#include <vector>

//#include "libtorch_utils.h"
//#include "libtorch_sequential_layer_model.h"
//#include "util/Progress.hpp"
#include "util/download_data_set.h"
//#include "libtorch_link_libs.hpp"




auto main() -> int {

	//Decompression of zip (gz) by directory structure is not supported
	//cpp_torch::file_uncompress("./data/test.zip", false);

	std::string url = "";
	std::vector<std::string> files = {"test.tar.gz"	};
	std::string dir = std::string("./data") + std::string("/");

	cpp_torch::url_download_dataSet(url, files, dir);

}
