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
		int ImageWrite(const std::string& path, const std::string& image_file_name)
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
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
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

			return 0;
		} // main

	}
}
#endif

#endif
