#ifndef libtorch_LIBS
#define libtorch_LIBS
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#ifdef USE_LIBTORCH_120
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "torch.lib")
#endif

#ifdef USE_LIBTORCH_110
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "foxi_dummy.lib")
#pragma comment(lib, "foxi_loader.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "onnx.lib")
#pragma comment(lib, "onnxifi_dummy.lib")
#pragma comment(lib, "onnxifi_loader.lib")
#pragma comment(lib, "onnx_proto.lib")
#pragma comment(lib, "torch.lib")
#endif

#endif
