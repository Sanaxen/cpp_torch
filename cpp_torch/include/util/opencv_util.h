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

#ifndef _HAS_CXX17
#error  CXX17
#endif

#include <time.h> 
using namespace cv;
using namespace std;

namespace cpp_torch
{
	namespace cvutil
	{
		cv::Mat ImgeTocvMat(const Image* img)
		{
			cv::Mat cvimg(img->height, img->width, CV_8UC3);

			const size_t sz = img->height*img->width;
#pragma omp parallel for
			for (int k = 0; k < sz; k++)
			{
				const int i = k / img->width;
				const int j = k % img->width;
				cvimg.at<cv::Vec3b>(i, j)[0] = img->data[k].b;//Blue
				cvimg.at<cv::Vec3b>(i, j)[1] = img->data[k].g; //Green
				cvimg.at<cv::Vec3b>(i, j)[2] = img->data[k].r; //Red	
			}
			return cvimg;
		}
		Image cvMatToImage(const cv::Mat& cvimg)
		{
			Image im;
			im.height = cvimg.rows;
			im.width = cvimg.cols;
			im.data.resize(im.height*im.width);

			const size_t sz = im.height*im.width;
#pragma omp parallel for
			for (int k = 0; k < sz; k++)
			{
				const int i = k / im.width;
				const int j = k % im.width;
				im.data[k].b = cvimg.at<cv::Vec3b>(i, j)[0];//Blue
				im.data[k].g = cvimg.at<cv::Vec3b>(i, j)[1]; //Green
				im.data[k].r = cvimg.at<cv::Vec3b>(i, j)[2]; //Red	
			}
			return im;
		}

		void resize(cv::Mat& cvimg, int h, int w)
		{
			cv:resize(cvimg, cvimg, cv::Size(), (double)(w) / cvimg.cols, (double)(h) / cvimg.rows);
		}
		void resize(cv::Mat& cvimg, int padding)
		{
			cv:resize(cvimg, cvimg, cv::Size(), (double)(cvimg.cols + padding) / cvimg.cols, (double)(cvimg.rows + padding) / cvimg.cols);
		}
		cv::Mat padding_img(cv::Mat& target_mat, int padding)
		{

			cv::Mat convert_mat, work_mat;
			work_mat = cv::Mat::zeros(cv::Size(target_mat.cols + padding, target_mat.rows + padding), CV_8UC3);
			convert_mat = target_mat.clone();

			cv::Mat Roi1(work_mat, cv::Rect((target_mat.cols + padding - convert_mat.cols) / 2.0, (target_mat.rows + padding - convert_mat.rows) / 2.0,
				convert_mat.cols, convert_mat.rows));
			convert_mat.copyTo(Roi1);

			return work_mat.clone();
		}

		cv::Mat tensorToMat(const torch::Tensor &tensor, const int scale = 1.0)
		{
			const auto sizes = tensor.sizes();
			//printf("%d %d %d\n", sizes[0], sizes[1], sizes[2]);
			if (sizes.size() > 3)
			{
				printf("%d %d %d\n", sizes[0], sizes[1], sizes[2]);
				throw error_exception("tensorToMat input size error.");
			}
			torch::Tensor out_tensor = tensor.detach();
			out_tensor.to(torch::kCPU);
			out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
			out_tensor = out_tensor.mul(scale).clamp(0, 255).to(torch::kU8);
			out_tensor = out_tensor.to(torch::kCPU);

			cv::Mat resultImg(sizes[1], sizes[2], CV_8UC3);

			if (tensor.device().type() == torch::kCPU)
			{
#pragma omp parallel for
				for (int y = 0; y < sizes[2]; ++y)
				{
					for (int x = 0; x < sizes[1]; ++x)
					{
						for (int c = 0; c < sizes[0]; ++c)
						{
							resultImg.at< cv::Vec3b>(y, x)[c] = out_tensor[y][x][c].template item<float_t>();
						}
					}
				}
			}
			else
			{
				std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
			}
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

		cv::Mat ImageWrite(const torch::Tensor batch_tensor, int M, int N, const std::string& image_file_name, int padding = 0)
		{
			//printf("%d\n", batch_tensor.sizes()[0]);

			int l = batch_tensor.sizes()[0];
			int k = 0;
			cv::Mat im1;
			cv::Mat im2;
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					if (k >= l) break;

					cv::Mat	im = tensorToMat(batch_tensor[k], 1.0);
					if (padding)
					{
						im = padding_img(im, padding);
					}
					if (j == 0) im1 = im;
					else  cv::hconcat(im1, im, im1);
					k += 1;
				}
				if (i == 0) im2 = im1;
				else cv::vconcat(im2, im1, im2);
				if (k >= l) break;
			}
			if (padding)
			{
				im2 = padding_img(im2, padding);
			}
			//cv::cvtColor(im2, im2, cv::COLOR_BGR2RGB);
			cv::imwrite(image_file_name, im2);

			return im2;
		} // main

		cv::Mat  ImageWrite(const std::string& path, int M, int N, const std::string& image_file_name, int padding = 0)
		{
			std::vector<string> flist;
			std::filesystem::path p(path);

			for_each(std::filesystem::directory_iterator(p),
				std::filesystem::directory_iterator(),
				[&flist](const std::filesystem::path& p) {
				if (std::filesystem::is_regular_file(p)) {
					string s = p.string();
					if (
						strstr(s.c_str(), ".bmp") == NULL && strstr(s.c_str(), ".BMP") == NULL &&
						strstr(s.c_str(), ".jpg") == NULL && strstr(s.c_str(), ".JPG") == NULL &&
						strstr(s.c_str(), ".jpeg") == NULL && strstr(s.c_str(), ".JPEG") == NULL &&
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
					if (padding)
					{
						im = padding_img(im, padding);
					}
					if (j == 0) im1 = im;
					else  cv::hconcat(im1, im, im1);
					k += 1;
				}
				if (i == 0) im2 = im1;
				else cv::vconcat(im2, im1, im2);
				if (k >= l) break;
			}
			if (padding)
			{
				im2 = padding_img(im2, padding);
			}
			//cv::cvtColor(im2, im2, cv::COLOR_BGR2RGB);
			cv::imwrite(image_file_name, im2);

			return im2;
		} // main

	}
}
#endif

#endif
