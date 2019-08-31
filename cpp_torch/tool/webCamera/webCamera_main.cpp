#define _CRT_SECURE_NO_WARNINGS 1  

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

#include <time.h> 
using namespace cv;
using namespace std;

const char* params =
"{ help           | false | print usage         }"
"{ camera_device  | 0     | camera device number}"
"{ frame_rate     | 40    | frame_rate  }"
"{ opencl         | false | enable OpenCL }"
"{ camera_width   | 640   | camera device width  }"
"{ camera_height  | 480   | camera device height }"
"{ video_file     |       | video file name }";
;

const char* about = "This sample uses web camera\n";

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, params);

	if (parser.get<bool>("help"))
	{
		cout << about << endl;
		parser.printMessage();
		return 0;
	}


	VideoCapture cap;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, parser.get<double>("camera_width"));
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, parser.get<double>("camera_height"));

	printf("%d x %d\n", parser.get<int>("camera_width"), parser.get<int>("camera_height"));
	printf("%d x %d\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("%f x %f\n", cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));


	int cameraDevice = parser.get<int>("camera_device");
	printf("cameraDevice=%d\n", cameraDevice);
	cap = VideoCapture(cameraDevice);
	if (!cap.isOpened())
	{
		cout << "Couldn't find camera: " << cameraDevice << endl;
		return -1;
	}
	if (parser.get<string>("video_file") != "")
	{
		if (cap.isOpened()) cap.release();
		cap.open(parser.get<string>("video_file"));

		if (!cap.isOpened())
		{
			cout << "Couldn't open video file: " << parser.get<string>("video_file") << endl;
			return -1;
		}
	}


	float camera_x = (int)parser.get<double>("camera_width");
	float camera_y = (int)parser.get<double>("camera_height");
	double frame_rate = cap.get(cv::CAP_PROP_FPS);
	if (frame_rate < 1.0) frame_rate = parser.get<double>("frame_rate");
	printf("frame rate:%f\n", frame_rate);
	
	__time64_t t1;
	_time64(&t1);
	struct tm t2;
	_localtime64_s(&t2, &t1);

	char tmpName[256];
	sprintf(tmpName, "%04d%02d%02d%02d%02d%02d.avi", t2.tm_year + 1900, t2.tm_mon + 1, t2.tm_mday, t2.tm_hour, t2.tm_min, t2.tm_sec);
	printf("output:%s\n", tmpName);

	cv::VideoWriter writer(tmpName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frame_rate, cv::Size(camera_x, camera_y));

	for (;;)
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera/video or read image

		if (frame.empty())
		{
			waitKey();
			break;
		}

		cv::Mat resizeframe;
		cv::resize(frame, resizeframe, Size(), camera_x / frame.cols, camera_y / frame.rows);
		frame = resizeframe.clone();

		if (frame.empty())
		{
			waitKey();
			break;
		}

		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);


		imshow("video", frame);

		if (writer.isOpened())
		{
			writer << frame;
		}

		if (waitKey(1) >= 0) break;
	}
	if (writer.isOpened())
	{
		writer.release();
	}

	return 0;
} // main
