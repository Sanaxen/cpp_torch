/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef __IMAGE_HPP

#define __IMAGE_HPP
#include <filesystem>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"
#endif

#pragma warning(disable : 4244)
#pragma warning(disable : 4018)

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

using namespace std;
namespace cpp_torch
{

	typedef struct Rgb_ {
		float_t b;
		float_t g;
		float_t r;
		float_t alp;
		~Rgb_() {}
		Rgb_() {}
		inline Rgb_(int x, int y, int z)
		{
			r = x;
			g = y;
			b = z;
			alp = 255;
		}
		inline Rgb_(const int* x)
		{
			r = x[0];
			g = x[1];
			b = x[2];
			alp = 255;
		}
		inline Rgb_(const unsigned char* x)
		{
			r = x[0];
			g = x[1];
			b = x[2];
			alp = 255;
		}
		inline Rgb_(const unsigned char x)
		{
			r = x;
			g = x;
			b = x;
			alp = 255;
		}
	} Rgb;


	class Image
	{
	public:
		unsigned int height;
		unsigned int width;
		std::vector<Rgb> data;

		Image()
		{
		}
		~Image()
		{
		}

		inline Image clone()
		{
			Image im;
			im.height = this->height;
			im.width = this->width;
			im.data = this->data;

			return im;
		}
	};


	inline void ImageWrite(const char* filename, Image* img, float scale = 1.0)
	{
		std::vector<unsigned char> data(3 * img->height*img->width);

		const size_t sz = img->height*img->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			data[3 * i + 0] = img->data[i].r*scale;
			data[3 * i + 1] = img->data[i].g*scale;
			data[3 * i + 2] = img->data[i].b*scale;
		}
		stbi_write_bmp(filename, img->width, img->height, 3, (void*)&data[0]);
	}

	template<class T>
	inline Image ToImage(T* data, int x, int y, int channel = 3)
	{
		Image img;

		img.data.resize(x*y);
		img.height = y;
		img.width = x;

		const size_t sz = x * y;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			if (channel == 3)
			{
				img.data[i].r = data[3 * i + 0];
				img.data[i].g = data[3 * i + 1];
				img.data[i].b = data[3 * i + 2];
			}
			if (channel == 2)
			{
				img.data[i].r = data[2 * i + 0];
				img.data[i].g = data[2 * i + 1];
				//img.data[i].b = data[2 * i + 1];
			}
			if (channel == 1)
			{
				img.data[i].r = data[1 * i + 0];
				//img.data[i].g = data[1 * i + 0];
				//img.data[i].b = data[1 * i + 0];
			}
		}
		return img;
	}

	template<class T>
	inline T* ImageTo(Image* img)
	{
		T* data = new T[img->width*img->height * 3];

		const size_t sz = img->height*img->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			data[3 * i + 0] = img->data[i].r;
			data[3 * i + 1] = img->data[i].g;
			data[3 * i + 2] = img->data[i].b;
		}
		return data;
	}

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

		return flist;
	}
	inline Image readImage(const char *filename)
	{
		Image img;

		unsigned char *data = 0;
		int x, y;
		int nbit;
		data = stbi_load(filename, &x, &y, &nbit, 0);
		if (data == NULL)
		{
			printf("image file[%s] read error.\n", filename);
			return img;
		}
		//printf("height %d   width %d \n", y, x);

		img.data.resize(x*y);
		img.height = y;
		img.width = x;

		const size_t sz = x * y;
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			if (nbit == 1)	//8bit
			{
				img.data[i].r = data[i];
				img.data[i].g = data[i];
				img.data[i].b = data[i];
				img.data[i].alp = 255;
			}
			if (nbit == 2)	//16bit
			{
				img.data[i].r = data[i * 2 + 0];
				img.data[i].g = data[i * 2 + 0];
				img.data[i].b = data[i * 2 + 0];
				img.data[i].alp = data[i * 2 + 1];
			}
			if (nbit == 3)	//24
			{
				img.data[i].r = data[i * 3 + 0];
				img.data[i].g = data[i * 3 + 1];
				img.data[i].b = data[i * 3 + 2];
				img.data[i].alp = 255;
			}
			if (nbit == 4)	//32
			{
				img.data[i].r = data[i * 4 + 0];
				img.data[i].g = data[i * 4 + 1];
				img.data[i].b = data[i * 4 + 2];
				img.data[i].alp = data[i * 4 + 3];
			}
		}
		stbi_image_free(data);

		return img;
	}


	class img_greyscale
	{
	public:
		void greyscale(Image* img)
		{
#pragma omp parallel for
			for (int i = 0; i < img->height*img->width; i++)
			{
				double c = (0.299 * img->data[i].r + 0.587 * img->data[i].g + 0.114 * img->data[i].b);
				img->data[i].r = c;
				img->data[i].g = c;
				img->data[i].b = c;
			}
		}
		void greyscale(double* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				double c = (0.299 * data[3 * i + 0] + 0.587 * data[3 * i + 1] + 0.114 * data[3 * i + 2]);
				data[3 * i + 0] = c;
				data[3 * i + 1] = c;
				data[3 * i + 2] = c;
			}
		}
		void greyscale(unsigned char* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				double c = (0.299 * data[3 * i + 0] + 0.587 * data[3 * i + 1] + 0.114 * data[3 * i + 2]);
				data[3 * i + 0] = c;
				data[3 * i + 1] = c;
				data[3 * i + 2] = c;
			}
		}

	};
	class img_gamma
	{
		double gamma_;
	public:
		img_gamma(double gamma)
		{
			gamma_ = gamma;
		}
		void gamma(Image* img)
		{
#pragma omp parallel for
			for (int i = 0; i < img->height*img->width; i++)
			{
				img->data[i].r = 255 * pow(img->data[i].r / 255.0, 1.0 / gamma_);
				img->data[i].g = 255 * pow(img->data[i].g / 255.0, 1.0 / gamma_);
				img->data[i].b = 255 * pow(img->data[i].b / 255.0, 1.0 / gamma_);
			}
		}
		void gamma(double* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = 255 * pow(data[3 * i + 0] / 255.0, 1.0 / gamma_);
				data[3 * i + 1] = 255 * pow(data[3 * i + 1] / 255.0, 1.0 / gamma_);
				data[3 * i + 2] = 255 * pow(data[3 * i + 2] / 255.0, 1.0 / gamma_);
			}
		}
		void gamma(unsigned char* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = 255 * pow(data[3 * i + 0] / 255.0, 1.0 / gamma_);
				data[3 * i + 1] = 255 * pow(data[3 * i + 1] / 255.0, 1.0 / gamma_);
				data[3 * i + 2] = 255 * pow(data[3 * i + 2] / 255.0, 1.0 / gamma_);
			}
		}
	};

	class img_contrast
	{
		int min_table;
		int max_table;
		int diff_table;

		float LUT_HC[255];
		float LUT_LC[255];

	public:
		img_contrast()
		{
			// ルックアップテーブルの生成
			min_table = 50;
			max_table = 205;
			diff_table = max_table - min_table;

			//ハイコントラストLUT作成
			for (int i = 0; i < min_table; i++)	LUT_HC[i] = 0;
			for (int i = min_table; i < max_table; i++)	LUT_HC[i] = 255.0 * (i - min_table) / (float)diff_table;
			for (int i = max_table; i < 255; i++)	LUT_HC[i] = 255.0;

			// ローコントラストLUT作成
			for (int i = 0; i < 255; i++) LUT_LC[i] = min_table + i * (diff_table) / 255.0;
		}

		void high(Image* img)
		{
#pragma omp parallel for
			for (int i = 0; i < img->height*img->width; i++)
			{
				img->data[i].r = (unsigned char)(std::min((float_t)255.0, LUT_HC[(unsigned char)img->data[i].r] * img->data[i].r));
				img->data[i].g = (unsigned char)(std::min((float_t)255.0, LUT_HC[(unsigned char)img->data[i].g] * img->data[i].g));
				img->data[i].b = (unsigned char)(std::min((float_t)255.0, LUT_HC[(unsigned char)img->data[i].b] * img->data[i].b));
			}
		}
		void high(double* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 0]))] * data[3 * i + 0]));
				data[3 * i + 1] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 1]))] * data[3 * i + 1]));
				data[3 * i + 2] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 2]))] * data[3 * i + 2]));
			}
		}
		void high(unsigned char* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = LUT_HC[data[3 * i + 0]] * data[3 * i + 0];
				data[3 * i + 1] = LUT_HC[data[3 * i + 1]] * data[3 * i + 1];
				data[3 * i + 2] = LUT_HC[data[3 * i + 2]] * data[3 * i + 2];
			}
		}
		void low(Image* img)
		{
#pragma omp parallel for
			for (int i = 0; i < img->height*img->width; i++)
			{
				img->data[i].r = (unsigned char)(std::min((float_t)255.0, LUT_LC[(unsigned char)img->data[i].r] * img->data[i].r));
				img->data[i].g = (unsigned char)(std::min((float_t)255.0, LUT_LC[(unsigned char)img->data[i].g] * img->data[i].g));
				img->data[i].b = (unsigned char)(std::min((float_t)255.0, LUT_LC[(unsigned char)img->data[i].b] * img->data[i].b));
			}
		}
		void low(double* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 0]))] * data[3 * i + 0]));
				data[3 * i + 1] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 1]))] * data[3 * i + 1]));
				data[3 * i + 2] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 2]))] * data[3 * i + 2]));
			}
		}
		void low(unsigned char* data, int x, int y)
		{
#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				data[3 * i + 0] = LUT_LC[data[3 * i + 0]] * data[3 * i + 0];
				data[3 * i + 1] = LUT_LC[data[3 * i + 1]] * data[3 * i + 1];
				data[3 * i + 2] = LUT_LC[data[3 * i + 2]] * data[3 * i + 2];
			}
		}
	};

	class img_noize
	{
		std::mt19937 mt;
		double sigma_;
		std::uniform_real_distribution<double> rand_a;
		double r;
	public:
		img_noize(double sigma = 15.0, double r_ = 0.3)
		{
			std::random_device seed_gen;
			std::mt19937 engine(seed_gen());
			mt = engine;
			sigma_ = sigma;
			std::uniform_real_distribution<double> rand_aa(0.0, 1.0);
			rand_a = rand_aa;
			r = r_;
		}

		void noize(Image* img)
		{
			std::uniform_real_distribution<double> d_rand(-sigma_, sigma_);

#pragma omp parallel for
			for (int i = 0; i < img->height*img->width; i++)
			{
				if (rand_a(mt) < r)
				{
					img->data[i].r = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].r + d_rand(mt))));
					img->data[i].g = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].g + d_rand(mt))));
					img->data[i].b = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].b + d_rand(mt))));
				}
			}
		}
		void noize(double* data, int x, int y)
		{
			std::uniform_real_distribution<double> d_rand(-sigma_, sigma_);

#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				if (rand_a(mt) < r)
				{
					data[3 * i + 0] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 0] + d_rand(mt))));
					data[3 * i + 1] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 1] + d_rand(mt))));
					data[3 * i + 2] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 2] + d_rand(mt))));
				}
			}
		}
		void noize(unsigned char* data, int x, int y)
		{
			std::normal_distribution<double> d_rand(0.0, sigma_);

#pragma omp parallel for
			for (int i = 0; i < x*y; i++)
			{
				if (rand_a(mt) < r)
				{
					data[3 * i + 0] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 0] + d_rand(mt))));
					data[3 * i + 1] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 1] + d_rand(mt))));
					data[3 * i + 2] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 2] + d_rand(mt))));
				}
			}
		}
	};

	class img_filter
	{
		double weight[3][3];
	public:
		img_filter(double* filter = NULL)
		{
			if (filter)
			{
				weight[0][0] = filter[0];
				weight[0][1] = filter[1];
				weight[0][2] = filter[2];
				weight[1][0] = filter[3];
				weight[1][1] = filter[4];
				weight[1][2] = filter[5];
				weight[2][0] = filter[6];
				weight[2][1] = filter[7];
				weight[2][2] = filter[8];
			}
			else
			{
				weight[0][0] = 1.0;
				weight[0][1] = 1.0;
				weight[0][2] = 1.0;
				weight[1][0] = 1.0;
				weight[1][1] = 1.0;
				weight[1][2] = 1.0;
				weight[2][0] = 1.0;
				weight[2][1] = 1.0;
				weight[2][2] = 1.0;
			}
		}

		void filter(Image* img)
		{
			const int x = img->width;
			const int y = img->height;

			for (int i = 0; i < y; i++)
			{
				for (int j = 0; j < x; j++)
				{
					double r, g, b;

					r = g = b = 0.0;
					for (int ii = 0; ii < 3; ii++)
					{
						for (int jj = 0; jj < 3; jj++)
						{
							int pos = ((i + ii)*x + (j + jj));
							if (pos >= x * y) continue;
							r += img->data[pos].r * weight[ii][jj];
							g += img->data[pos].g * weight[ii][jj];
							b += img->data[pos].b * weight[ii][jj];
						}
					}
					r /= 9.0;
					g /= 9.0;
					b /= 9.0;
					int pos = i * x + j;
					img->data[pos].r = (unsigned char)std::min(255.0, r);
					img->data[pos].g = (unsigned char)std::min(255.0, g);
					img->data[pos].b = (unsigned char)std::min(255.0, b);
				}
			}
		}
		void filter(double* data, int x, int y)
		{
			Image img = ToImage(data, x, y);
			filter(&img);

			double* data2 = ImageTo<double>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}
		void filter(unsigned char* data, int x, int y)
		{
			Image img = ToImage(data, x, y);
			filter(&img);

			unsigned char* data2 = ImageTo<unsigned char>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}
	};


	class img_rotation
	{
	public:
		img_rotation() {}

		void rotation(Image* img, const double rad)
		{
			const int x = img->width;
			const int y = img->height;
			double centerX = x / 2.0;
			double centerY = y / 2.0;
			double cosRadian = cos(rad);
			double sinRadian = sin(rad);

			//対角の長さに画像を拡大しておく
			double r = sqrt(centerX*centerX + centerY * centerY)*2.0;
			int R = (int)(r + 0.5);

			//増量分
			int fx = (R - x) / 2;
			int fy = (R - y) / 2;

			int Rx = x + 2 * fx;
			int Ry = y + 2 * fy;
			Image RimgI;
			RimgI.data.resize(Rx*Ry);
			RimgI.height = Ry;
			RimgI.width = Rx;

			//拡大領域に複写
#pragma omp parallel for
			for (int i = 0; i < Ry; i++)
			{
				for (int j = 0; j < Rx; j++)
				{
					int pos = i * Rx + j;
					if (i - fy >= 0 && i - fy < y && j - fx >= 0 && j - fx < x)
					{
						RimgI.data[pos] = img->data[(i - fy) * x + (j - fx)];
					}
					else
					{
						//拡大して広げたところは境界値で埋める
						int ii = -1, jj = -1;

						if (i - fy < 0)  ii = 0;
						if (i - fy >= y) ii = y - 1;

						if (j - fx < 0)  jj = 0;
						if (j - fx >= x) jj = x - 1;

						if (ii < 0) ii = 0;
						if (jj < 0) jj = 0;

						pos = i * Rx + j;
						RimgI.data[pos] = img->data[ii * x + jj];
					}
				}
			}

			//回転させるための複製
			std::vector<Rgb> data = RimgI.data;

			centerX = Rx / 2.0;
			centerY = Ry / 2.0;
#pragma omp parallel for
			for (int i = 0; i < Ry; i++)
			{
				for (int j = 0; j < Rx; j++)
				{
					int pos = i * Rx + j;

					int pointX = (int)((j - centerX) * cosRadian - (i - centerY) * sinRadian + centerX);
					int pointY = (int)((j - centerX) * sinRadian + (i - centerY) * cosRadian + centerY);

					// poiuntX, pointYが入力画像の有効範囲にあれば出力画像へ代入する
					if (pointX >= 0 && pointX < Rx && pointY >= 0 && pointY < Ry) {
						RimgI.data[pos] = data[pointY * Rx + pointX];
					}
					else {
						RimgI.data[pos] = Rgb(0, 0, 0);
					}
				}
			}

			//元のサイズで切り出し
#pragma omp parallel for
			for (int i = 0; i < Ry; i++)
			{
				for (int j = 0; j < Rx; j++)
				{
					int pos = i * Rx + j;
					if (i - fy >= 0 && i - fy < y && j - fx >= 0 && j - fx < x)
					{
						img->data[(i - fy) * x + (j - fx)] = RimgI.data[pos];
					}
				}
			}
		}

		void rotation(double* data, int x, int y, const double rad)
		{
			Image img = ToImage(data, x, y);
			rotation(&img, rad);

			double* data2 = ImageTo<double>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}

		void rotation(unsigned char* data, int x, int y, const double rad)
		{
			Image img = ToImage(data, x, y);
			rotation(&img, rad);

			unsigned char* data2 = ImageTo<unsigned char>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}
	};
	class img_sift
	{
	public:
		img_sift() {}

		void sift(Image* img, const int axis, const int delta)
		{
			const int x = img->width;
			const int y = img->height;

			std::vector<Rgb> data = img->data;

			if (axis == 1)
			{
				for (int i = 0; i < y; i++)
				{
					for (int j = 0; j < x - delta; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[i * x + (j + delta)];
					}
					for (int j = x - delta; j < x; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[i * x + (x - 1)];
					}
				}
			}
			if (axis == -1)
			{
				for (int i = 0; i < y; i++)
				{
					for (int j = 0; j < x; j++)
					{
						int pos = i * x + j + delta;

						if (j + delta == x) break;
						img->data[pos] = data[i * x + j];
					}
					for (int j = 0; j < delta; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[i * x + 0];
					}
				}
			}
			if (axis == 2)
			{
				for (int i = 0; i < y - delta; i++)
				{
					for (int j = 0; j < x; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[(i + delta) * x + j];
					}
				}
				for (int i = y - delta; i < y; i++)
				{
					for (int j = 0; j < x; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[(y - 1) * x + j];
					}
				}
			}
			if (axis == -2)
			{
				for (int i = delta; i < y; i++)
				{
					for (int j = 0; j < x; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[(i - delta)* x + j];
					}
				}
				for (int i = 0; i < delta; i++)
				{
					for (int j = 0; j < x; j++)
					{
						int pos = i * x + j;

						img->data[pos] = data[0 * x + j];
					}
				}
			}
		}

		void sift(double* data, int x, int y, const int axis, const int delta)
		{
			Image img = ToImage(data, x, y);
			sift(&img, axis, delta);

			double* data2 = ImageTo<double>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}

		void sift(unsigned char* data, int x, int y, const int axis, const int delta)
		{
			Image img = ToImage(data, x, y);
			sift(&img, axis, delta);

			unsigned char* data2 = ImageTo<unsigned char>(&img);
#pragma omp parallel for
			for (int i = 0; i < x*y * 3; i++)
			{
				data[i] = data2[i];
			}
			delete[] data2;
		}
	};


	class img_padding
	{
	public:
		img_padding() {}

		void padding(Image* img, const int padding_sz, const float padding_value)
		{
			const int x = img->width;
			const int y = img->height;
			int Rx = x + 2 * padding_sz;
			int Ry = y + 2 * padding_sz;

			Image RimgI;
			RimgI.data.resize(Rx*Ry);
			RimgI.height = Ry;
			RimgI.width = Rx;

#pragma omp parallel for
			for (int i = 0; i < Ry; i++)
			{
				for (int j = 0; j < Rx; j++)
				{
					int pos = i * Rx + j;
					if (i - padding_sz >= 0 && i - padding_sz < y && j - padding_sz >= 0 && j - padding_sz < x)
					{
						RimgI.data[pos] = img->data[(i - padding_sz) * x + (j - padding_sz)];
					}
					else
					{
						//拡大して広げたところはpadding_value
						pos = i * Rx + j;
						RimgI.data[pos] = Rgb(padding_value, padding_value, padding_value);
					}
				}
			}
			img->data = RimgI.data;
			img->height = RimgI.height;
			img->width = RimgI.width;
		}

		void padding(double** data, int x, int y, const int padding_sz, const float padding_value)
		{
			Image img = ToImage<double>(*data, x, y);
			padding(&img, padding_sz, padding_value);

			delete[] * data;
			*data = ImageTo<double>(&img);
		}

		void padding(unsigned char** data, int x, int y, const int padding_sz, const float padding_value)
		{
			Image img = ToImage<unsigned char>(*data, x, y);
			padding(&img, padding_sz, padding_value);

			delete[] * data;
			*data = ImageTo<unsigned char>(&img);
		}
	};


	Image vec_t2image(std::vector<float_t>& img, int channel, int height, int width)
	{
		std::vector<float_t> image_data(channel*height*width);

		const size_t sz = height * width;

#pragma omp parallel for
		for (int i = 0; i < sz; i++) {
			for (int c = 0; c < channel; c++) {
				float_t d = img[c * height*width + i];
				if (d < 0) d = 0;
				if (d > 255) d = 255;
				image_data[i] = d;
			}
		}
		return ToImage(&(image_data[0]), height, width, channel);
	}
	std::vector<float_t> image2vec_t(Image* img, int in_channel, int height, int width, float scale = 1.0)
	{
		std::vector<float_t> image_data(in_channel*height*width);
		
		const size_t sz = height * width;

#pragma omp parallel for
		for (int i = 0; i < sz; i++) {
			image_data[0 * height*width + i] = img->data[i].r* scale;
			if (in_channel >= 2)
			{
				image_data[1 * height*width + i] = img->data[i].g* scale;
				if (in_channel == 3)
				{
					image_data[2 * height*width + i] = img->data[i].b* scale;
				}
			}
		}
		return image_data;
	}

	std::vector<float_t> image_channel2vec_t(Image* img, int get_channel, int height, int width, float scale = 1.0)
	{
		std::vector<float_t> image_data(1*height*width);

		const size_t sz = height * width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++) {
			if (get_channel == 1)
			{
				image_data[i] = img->data[i].r* scale;
			}
			if (get_channel == 2)
			{
				image_data[i] = img->data[i].g* scale;
			}
			if (get_channel == 3)
			{
				image_data[i] = img->data[i].b* scale;
			}
		}
		return image_data;
	}

	void ImageAugmentation(std::vector<float_t>& vec, const int y, const int x, const std::string& func = "gamma")
	{
		//訓練データの水増し
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::uniform_real_distribution<> rand(0.0, 1.0);

		Image img = vec_t2image(vec, 3, y, x);
		unsigned char* data_p = ImageTo<unsigned char>(&img);

		std::vector<unsigned char> data;
		data.assign(data_p, data_p + 3 * y*x);

		if (func == "GAMMA")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);

			double g = rand(mt);
			{
				g = 1.2 - g * 2.0;
				//ガンマ補正
#pragma omp parallel for
				for (int i = 0; i < x*y; i++) {
					data2[i * 3 + 0] = 255 * pow(data[i * 3 + 0] / 255.0, 1.0 / g);
					data2[i * 3 + 1] = 255 * pow(data[i * 3 + 1] / 255.0, 1.0 / g);
					data2[i * 3 + 2] = 255 * pow(data[i * 3 + 2] / 255.0, 1.0 / g);
				}
			}
			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}

		if (func == "RL")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);

			//左右反転
#pragma omp parallel for
			for (int i = 0; i < y; i++) {
				for (int j = 0; j < x; j++) {
					int pos = (i*x + j);
					int pos2 = (i*x + x - j - 1);
					data2[pos * 3 + 0] = data[pos2 * 3 + 0];
					data2[pos * 3 + 1] = data[pos2 * 3 + 1];
					data2[pos * 3 + 2] = data[pos2 * 3 + 2];
				}
			}
			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}

		if (func == "COLOR_NOIZE")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);

			float c = rand(mt);
			float rr = 1.0, gg = 1.0, bb = 1.0;
			if (c < 0.3) rr = rand(mt);
			if (c >= 0.3 && c < 0.6) gg = rand(mt);
			if (c >= 0.6) bb = rand(mt);
#pragma omp parallel for
			for (int i = 0; i < x*y; i++) {
				data2[i * 3 + 0] = data[i * 3 + 0] * rr;
				data2[i * 3 + 1] = data[i * 3 + 1] * gg;
				data2[i * 3 + 2] = data[i * 3 + 2] * bb;
			}
			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}

		if (func == "NOIZE")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
			for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

			img_noize nz(15.0, rand(mt));
			nz.noize(&data2[0], x, y);

			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}
		//double g;
		if (func == "ROTATION")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
			for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

			img_rotation rot;
			rot.rotation(&data2[0], x, y, (rand(mt) < 0.5 ? 1.0 : -1.0)*(std::max(0.1, rand(mt))*M_PI / 180.0));

			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}

		if (func == "SIFT")
		{
			std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
			for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

			img_sift s;

			if (rand(mt) < 0.5)
			{
				if (rand(mt) < 0.5)
				{
					s.sift(&data2[0], x, y, 1, (int)(std::max(0.1, rand(mt)) * 5 + 0.5));
				}
				else
				{
					s.sift(&data2[0], x, y, -1, (int)(std::max(0.1, rand(mt)) * 5 + 0.5));
				}
			}
			else
			{
				if (rand(mt) < 0.5)
				{
					s.sift(&data2[0], x, y, 2, (int)(std::max(0.1, rand(mt)) * 5 + 0.5));
				}
				else
				{
					s.sift(&data2[0], x, y, -2, (int)(std::max(0.1, rand(mt)) * 5 + 0.5));
				}
			}

			Image img2 = ToImage<unsigned char>(&data2[0], x, y);
			vec = image2vec_t(&img2, 3, y, x);
			return;
		}
		return;
	}

	inline void clump(float& x, float min_, float max_)
	{
		if (x < min_) x = min_;
		if (x > max_) x = max_;
	}
	/*
	RGB -> YCbCr
	*/
	inline void RGB2YCbCr(float R, float G, float B, float& Y, float& Cb, float& Cr)
	{
		Y = 0.257*R + 0.504*G + 0.098*B + 16;
		Cb = -0.148*R - 0.291*G + 0.439*B + 128;
		Cr = 0.439*R - 0.368*G - 0.071*B + 128;
		//Y = 0.299*R + 0.587*G + 0.114*B;
		//Cr = 0.500*R - 0.419*G - 0.081*B;
		//Cb = -0.169*R - 0.332*G + 0.500*B;

		clump(Y, 0, 255);
		clump(Cr, 0, 255);
		clump(Cb, 0, 255);
	}
	/*
	YCbCr -> RGB
	*/
	inline void YCbCr2RGB(float Y, float Cb, float Cr, float& R, float& G, float& B)
	{
		R = 1.164*(Y - 16) + 1.596*(Cr - 128);
		G = 1.164*(Y - 16) - 0.391*(Cb - 128) - 0.813*(Cr - 128);
		B = 1.164*(Y - 16) + 2.018*(Cb - 128);
		//R = Y + 1.402*Cr;
		//G = Y - 0.714*Cr - 0.344*Cb;
		//B = Y + 1.772*Cb;

		clump(R, 0, 255);
		clump(G, 0, 255);
		clump(B, 0, 255);
	}

	/*
	* @param [IN/OUT] image		RGB -> YCbCr
	*/
	inline void ImageRGB2YCbCr(cpp_torch::Image* image)
	{
		std::vector<cpp_torch::Rgb> data = std::vector<cpp_torch::Rgb>(image->height*image->width);

		const size_t sz = image->height * image->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			RGB2YCbCr(
				image->data[i].r,
				image->data[i].g,
				image->data[i].b,
				data[i].r,
				data[i].g,
				data[i].b);
		}
		image->data = data;
	}

	/*
	* @param [IN/OUT] image		YCbCr -> RGB
	*/
	inline void ImageYCbCr2RGB(cpp_torch::Image* image)
	{
		std::vector<cpp_torch::Rgb> data = std::vector<cpp_torch::Rgb>(image->height*image->width);

		const size_t sz = image->height * image->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			YCbCr2RGB(
				image->data[i].r,
				image->data[i].g,
				image->data[i].b,
				data[i].r,
				data[i].g,
				data[i].b);
		}
		image->data = data;
	}

	/*
	* @param [IN/OUT] imageY  channel =1 -> imageYCbCr(R),imageY(G),imageY(B)
	* @param [IN] imageYCbCr
	* @param [IN] channel channel=(1,2,3)
	*/
	inline void ImageChgChannel(cpp_torch::Image* imageY, cpp_torch::Image* imageYCbCr, int channel)
	{
		const size_t sz = imageY->height * imageY->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			if (channel == 1)
			{
				imageY->data[i].r = imageYCbCr->data[i].r;
			}
			if (channel == 2)
			{
				imageY->data[i].g = imageYCbCr->data[i].g;
			}
			if (channel == 3)
			{
				imageY->data[i].b = imageYCbCr->data[i].b;
			}
		}
	}

	/*
	 * @param [IN/OUT] image  channel =1 -> image(R,G,B) -> image(R)
	 */
	inline void ImageGetChannel(cpp_torch::Image* image, int channel)
	{
		const size_t sz = image->height * image->width;
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			if (channel == 1)
			{
				image->data[i].g = 0;
				image->data[i].b = 0;
			}
			if (channel == 2)
			{
				image->data[i].r = 0;
				image->data[i].b = 0;
			}
			if (channel == 3)
			{
				image->data[i].r = 0;
				image->data[i].g = 0;
			}
		}
	}

}
//#undef STB_IMAGE_IMPLEMENTATION

#endif
