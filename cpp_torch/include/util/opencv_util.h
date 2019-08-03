#ifndef _OPENCV_UTIL_H
#define _OPENCV_UTIL_H

#define _CRT_SECURE_NO_WARNINGS 1  

#ifdef USE_OPENCV_UTIL
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <filesystem>

#include <time.h> 
using namespace cv;
using namespace std;

namespace cpp_torch
{
	namespace cvutil
	{
		cv::Mat tensorToMat(const torch::Tensor &tensor, const int scale = 1.0)
		{
			auto sizes = tensor.sizes();
			//printf("%d %d %d\n", sizes[0], sizes[1], sizes[2]);
			torch::Tensor out_tensor = tensor.detach();
			out_tensor.to(torch::kCPU);
			out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
			out_tensor = out_tensor.mul(scale).clamp(0, scale).to(torch::kU8);
			out_tensor = out_tensor.to(torch::kCPU);

			cv::Mat resultImg(sizes[1], sizes[2], CV_8UC3);
			std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
			cv::cvtColor(resultImg, resultImg, cv::COLOR_BGR2RGB);

			//cv::imshow("", resultImg);
			//cv::waitKey();
			return resultImg;
		}

		cv::Mat TensorToImageFile(torch::Tensor image_tensor, const std::string& filename, const int scale = 1.0)
		{
			cv::Mat& cv_mat = tensorToMat(image_tensor, scale);
			cv::imwrite(filename, cv_mat);
			return cv_mat;
		}

		cv::Mat ImageWrite(const torch::Tensor batch_tensor, int M, int N, const std::string& image_file_name)
		{
			printf("%d\n", batch_tensor.sizes()[0]);

			int l = batch_tensor.sizes()[0];
			int k = 0;
			cv::Mat im1;
			cv::Mat im2;
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					if (k >= l) break;

					cv::Mat	im = tensorToMat(batch_tensor[k], 255.0);
					if (j == 0) im1 = im;
					else  cv::hconcat(im1, im, im1);
					k += 1;
				}
				if (i == 0) im2 = im1;
				else cv::vconcat(im2, im1, im2);
				if (k >= l) break;
			}
			//cv::cvtColor(im2, im2, cv::COLOR_BGR2RGB);
			cv::imwrite(image_file_name, im2);

			return im2;
		} // main

		cv::Mat  ImageWrite(const std::string& path, int M, int N, const std::string& image_file_name)
		{
			std::vector<string> flist;
			std::tr2::sys::path p(path);

			for_each(std::tr2::sys::directory_iterator(p),
				std::tr2::sys::directory_iterator(),
				[&flist](const std::tr2::sys::path& p) {
				if (std::tr2::sys::is_regular_file(p)) {
					string s = p.string();
					if (
						strstr(s.c_str(), ".bmp") == NULL && strstr(s.c_str(), ".BMP") == NULL &&
						strstr(s.c_str(), ".jpg") == NULL && strstr(s.c_str(), ".JPG") == NULL &&
						strstr(s.c_str(), ".png") == NULL && strstr(s.c_str(), ".PNG") == NULL
						)
					{
						/* skipp*/
					}
					else
					{
						flist.push_back(s);
					}
				}
			});

			printf("%d\n", flist.size());

			int l = flist.size();
			int k = 0;
			cv::Mat im1;
			cv::Mat im2;
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					if (k >= l) break;

					cv::Mat	im = imread(flist[k]);
					if (j == 0) im1 = im;
					else  cv::hconcat(im1, im, im1);
					k += 1;
				}
				if (i == 0) im2 = im1;
				else cv::vconcat(im2, im1, im2);
				if (k >= l) break;
			}
			//cv::cvtColor(im2, im2, cv::COLOR_BGR2RGB);
			cv::imwrite(image_file_name, im2);

			return im2;
		} // main

	}
}
#endif

#endif
