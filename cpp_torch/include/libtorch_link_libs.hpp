#ifndef libtorch_LIBS
#define libtorch_LIBS
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifdef USE_TORCHVISION_0100
#pragma comment(lib, "../torchvision/lib/torchvision.lib")
#endif

#ifdef USE_LIBTORCH_190
#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10d.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "Caffe2_perfkernels_avx.lib")
#pragma comment(lib, "Caffe2_perfkernels_avx2.lib")
#pragma comment(lib, "Caffe2_perfkernels_avx512.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "fbjni.lib")
#pragma comment(lib, "kineto.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "mkldnn.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "pytorch_jni.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "XNNPACK.lib")
#endif

#ifdef USE_LIBTORCH_181
#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10d.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "fbjni.lib")
#pragma comment(lib, "gloo.lib")
#pragma comment(lib, "gloo_cuda.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "mkldnn.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "pytorch_jni.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "XNNPACK.lib")
#endif

#ifdef USE_LIBTORCH_171
#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10d.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "gloo.lib")
#pragma comment(lib, "gloo_cuda.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "mkldnn.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#endif

#ifdef USE_LIBTORCH_160
#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "mkldnn.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#endif

#ifdef USE_LIBTORCH_150
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
#pragma comment(lib, "mkldnn.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#endif

#ifdef USE_LIBTORCH_140
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "torch.lib")
#endif

#ifdef USE_LIBTORCH_130
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
