#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325) "
                    "with ResNet-10 architecture to detect faces on camera/video/image.\n"
                    "More information about the training is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt\n"
                    ".caffemodel model's file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel\n"
                    ".prototxt file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt\n";

const char* params
    = "{ help           | false | print usage          }"
      "{ proto          |       | model configuration (deploy.prototxt) }"
      "{ model          |       | model weights (res10_300x300_ssd_iter_140000.caffemodel) }"
      "{ camera_device  | 0     | camera device number }"
	  "{ camera_width   | 640   | camera device width  }"
	  "{ camera_height  | 480   | camera device height }"
	  "{ video          |       | video or image for detection }"
      "{ opencl         | false | enable OpenCL }"
	  "{ view           | true  | enable viewe }"
	  "{ min_size       | 10    | min image size }"
	  "{ min_confidence | 0.5   | min confidence       }";

std::vector<std::string> getImageFiles(const std::string& dir)
{
	std::vector<std::string> flist;
	std::tr2::sys::path p(dir);

	for_each(std::tr2::sys::directory_iterator(p),
		std::tr2::sys::directory_iterator(),
		[&flist](const std::tr2::sys::path& p) {
		if (std::tr2::sys::is_regular_file(p)) {
			string s = p.string();
			if (
				strstr(s.c_str(), ".bmp") == NULL && strstr(s.c_str(), ".BMP") == NULL &&
				strstr(s.c_str(), ".jpg") == NULL && strstr(s.c_str(), ".JPG") == NULL &&
				strstr(s.c_str(), ".jpeg") == NULL && strstr(s.c_str(), ".JPEG") == NULL &&
				strstr(s.c_str(), ".png") == NULL && strstr(s.c_str(), ".PNG") == NULL &&
				strstr(s.c_str(), ".avi") == NULL && strstr(s.c_str(), ".AVI") == NULL &&
				strstr(s.c_str(), ".mp4") == NULL && strstr(s.c_str(), ".MP4") == NULL 
				)
			{
				// skipp
			}
			else
			{
				flist.push_back(s);
			}
		}
		else if (std::tr2::sys::is_directory(p))
		{
			//cout << p.string() << endl;
			//getchar();
			std::vector<std::string>& ls = getImageFiles(p.string());
			flist.insert(flist.end(), ls.begin(), ls.end());
		}
	});

	return flist;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

    if (parser.get<bool>("opencl"))
    {
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    }
	std::vector<std::string> imagefiles;

    VideoCapture cap;
	if (parser.get<String>("video").empty())
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }
		imagefiles.push_back("camera_device");
    }
    else
    {
        cap.open(parser.get<String>("video"));
        if(!cap.isOpened())
        {
			imagefiles = getImageFiles(parser.get<String>("video"));
			if (imagefiles.size() == 0)
			{
				imagefiles.push_back(parser.get<String>("video"));
				cout << "image or video or directory: " << parser.get<String>("video") << endl;
				//return -1;
			}
		}
		else
		{
			cap.release();
			imagefiles.push_back(parser.get<String>("video"));
		}
    }
	if (imagefiles.size() == 0)
	{
		cout << "Couldn't open image or video or directory: " << parser.get<String>("video") << endl;
		return -1;
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, parser.get<double>("camera_width"));
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, parser.get<double>("camera_height"));

	int num = 0;

	for (auto& f : imagefiles)
	{
		cout << "image:" << num << "/ " << imagefiles.size() << f << endl;
		if (f == std::string("camera_device"))
		{

		}
		else
		{
			cap.open(f);
		}

		bool image_file = false;
		if (!cap.isOpened())
		{
			image_file = true;
		}
		for (;;)
		{
			Mat frame;
			cap >> frame; // get a new frame from camera/video or read image

			if (!cap.isOpened())
			{
				frame = cv::imread(f);
			}
			if (frame.empty())
			{
				waitKey(500);
				break;
			}

			if (frame.rows < parser.get<int>("min_size") || frame.cols < parser.get<int>("min_size"))
			{
				continue;
			}
			if (frame.rows < 64 || frame.cols < 64)
			{
				cv::resize(frame, frame, cv::Size(128, 128), 0, 0, INTER_CUBIC);
			}

			if (frame.channels() == 4)
				cvtColor(frame, frame, COLOR_BGRA2BGR);

			Mat inputBlob;
			try
			{
				//! [Prepare blob]
				inputBlob = blobFromImage(frame, inScaleFactor,
					Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images
			}
			catch (cv::Exception& err)
			{
				cout << "Exception:" << err.what() << endl;
				continue;
			}
			catch (...)
			{
				cout << "Exception:" << "???" << endl;
				continue;
			}
			//! [Prepare blob]

//! [Set input blob]
			net.setInput(inputBlob, "data"); //set the network input
			//! [Set input blob]

			Mat detection;
			try
			{
				//! [Make forward pass]
				detection = net.forward("detection_out"); //compute output
				//! [Make forward pass]
			}
			catch (cv::Exception& err)
			{
				cout << "Exception:" << err.what() << endl;
				continue;
			}
			catch (...)
			{
				cout << "Exception:" << "???" << endl;
				continue;
			}

			vector<double> layersTimings;
			double freq = getTickFrequency() / 1000;
			double time = net.getPerfProfile(layersTimings) / freq;

			Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

			ostringstream ss;
			ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
			putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

			float confidenceThreshold = parser.get<float>("min_confidence");
			for (int i = 0; i < detectionMat.rows; i++)
			{
				float confidence = detectionMat.at<float>(i, 2);

				if (confidence > confidenceThreshold)
				{
					int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
					int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
					int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
					int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);


					Rect object_tmp((int)xLeftBottom, (int)yLeftBottom,
						(int)(xRightTop - xLeftBottom),
						(int)(yRightTop - yLeftBottom));

					float yy = object_tmp.height*1.65 - object_tmp.height;
					float xx = object_tmp.width*1.65 - object_tmp.width;
					xLeftBottom -= xx * 0.5;
					yLeftBottom -= yy * 0.5;
					xRightTop += xx * 0.5;
					yRightTop += yy * 0.5;
					if (xLeftBottom < 0) xLeftBottom = 0;
					if (yLeftBottom < 0) yLeftBottom = 0;
					if (xRightTop >= frame.cols) xLeftBottom = frame.cols-1;
					if (yRightTop >= frame.rows) yLeftBottom = frame.rows-1;

					Rect object((int)xLeftBottom, (int)yLeftBottom,
						(int)(xRightTop - xLeftBottom),
						(int)(yRightTop - yLeftBottom));

					try
					{
						int xLeftBottom2 = xLeftBottom;
						int yLeftBottom2 = yLeftBottom;
						int xRightTop2 = xRightTop;
						int yRightTop2 = yRightTop;

						if (xRightTop - xLeftBottom > yRightTop - yLeftBottom)
						{
							float d = (xRightTop - xLeftBottom) - (yRightTop - yLeftBottom);
							
							yLeftBottom2 -= d * 0.5;
							yRightTop2 += d * 0.5;

						}
						else
						{
							if (xRightTop - xLeftBottom < yRightTop - yLeftBottom)
							{
								float d = (yRightTop - yLeftBottom) - (xRightTop - xLeftBottom);

								xLeftBottom2 -= d * 0.5;
								xRightTop2 += d * 0.5;
							}
						}

						Rect object2((int)xLeftBottom2, (int)yLeftBottom2,
							(int)(xRightTop2 - xLeftBottom2),
							(int)(yRightTop2 - yLeftBottom2));
						
						Mat cutimg(frame, object2);
						if (cutimg.rows < parser.get<int>("min_size") || cutimg.cols < parser.get<int>("min_size"))
						{
							continue;
						}

						char outfile[256];
						sprintf(outfile, "images\\output%d.png", ++num);
						imwrite(outfile, cutimg);
					}
					catch (...)
					{
						break;
					}

					rectangle(frame, object, Scalar(0, 255, 0));

					ss.str("");
					ss << confidence;
					String conf(ss.str());
					String label = "Face: " + conf;
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
						Size(labelSize.width, labelSize.height + baseLine)),
						Scalar(255, 255, 255), FILLED);
					putText(frame, label, Point(xLeftBottom, yLeftBottom),
						FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				}
			}

			if (parser.get<bool>("view"))imshow("detections", frame);
			if (waitKey(1) >= 0) break;
			if (image_file)
			{
				waitKey(500);
				break;
			}
		}
	}
    return 0;
} // main
