#define _CRT_SECURE_NO_WARNINGS 1  

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


int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "image_concat.exe <image_dir> <M>x<N> output_image\n");
		return -1;
	}
	std::vector<string> flist;
	std::tr2::sys::path p(argv[1]);

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

	int M = 8;
	int N = 8;
	sscanf(argv[2], "%dx%d", &M, &N);
	printf("%d %dx%d\n", flist.size(), M, N);

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
	if (argc >= 4)
	{
		cv::imwrite(argv[3], im2);
	}
	else
	{
		cv::imwrite("image_array.png", im2);
	}
	return 0;
} // main
