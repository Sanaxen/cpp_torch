/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
#ifndef _CPP_TORCH_H

#define _CPP_TORCH_H


#include <torch/csrc/api/include/torch/nn/init.h>
#include <torch/enum.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "config.h"

#include "libtorch_utils.h"
#include "libtorch_sequential_layer_model.h"
#include "util/Progress.hpp"
#include "util/download_data_set.h"
#include "csvreader.h"

#ifdef USE_IMAGE_UTIL
#include "util/Image.hpp"
#endif

#ifdef USE_IMAGE_UTIL
#include "util/opencv_util.h"
#include "util/opencv_link_libs.h"
#endif

#ifdef USE_TORCHVISION_0100
#define USE_TORCHVISION
#ifdef USE_TORCHVISION
#include "../../libtorch/torchvision/include/torchvision/vision.h"
#include "../../libtorch/torchvision/include/torchvision/models/resnet.h"
#endif
#endif

#include "libtorch_link_libs.hpp"

#endif
