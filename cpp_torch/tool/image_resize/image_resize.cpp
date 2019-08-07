#define _CRT_SECURE_NO_WARNINGS 1  

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <filesystem>

#include <string.h>
#include <time.h> 
using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "image_resize.exe <image_dir> <W>x<H> output_dir\n");
		return -1;
	}
	std::vector<std::string> flist;
	std::tr2::sys::path p(argv[1]);

	for_each(std::tr2::sys::directory_iterator(p),
		std::tr2::sys::directory_iterator(),
		[&flist](const std::tr2::sys::path& p) {
		if (std::tr2::sys::is_regular_file(p)) {
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

	int W = 0;
	int H = 0;
	sscanf(argv[2], "%dx%d", &W, &H);
	printf("%d %dx%d\n", flist.size(), W, H);

	int l = flist.size();
	int k = 0;
	cv::Mat im1;
	cv::Mat im2;
	for (int i = 0; i < l; i++)
	{
		cv::Mat	im = imread(flist[i]);
		printf("%dx%d ->", im.cols, im.rows);
		if (im.cols == 0 || im.rows == 0) continue;
		cv:resize(im, im, cv::Size(), (double)(W) / im.cols, (double)(H) / im.rows);
		
		char drive[_MAX_DRIVE];	// ドライブ名
		char dir[_MAX_DIR];		// ディレクトリ名
		char fname[_MAX_FNAME];	// ファイル名
		char ext[_MAX_EXT];		// 拡張子

		_splitpath(flist[i].c_str(), drive, dir, fname, ext);

		std::string outfile_name = std::string(argv[3]) + std::string("\\") + fname + std::string(ext);

		printf("%s\n", outfile_name.c_str());
		cv::imwrite(outfile_name, im);
	}
	return 0;
} // main
