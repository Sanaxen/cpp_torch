/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
#ifndef _DATA_AUGMANTATION_H

#define _DATA_AUGMANTATION_H

namespace cpp_torch
{
	namespace test
	{
		void Image3CannelDataAugment(std::vector<tiny_dnn::vec_t>& train_images, std::vector<tiny_dnn::label_t>& train_labels, const float_t mean, const float_t stddiv, const int image_height, const int image_width, int extend_factor=2, float channel_range = CHANNEL_RANGE)
		{
			std::random_device rnd;
			std::mt19937 mt(rnd());
			std::uniform_int_distribution<> rand(0, 5);
			std::uniform_int_distribution<> rand_index(0, train_images.size() - 1);

			const size_t sz = train_images.size();
			for (int i = 0; i < sz * extend_factor; i++)
			{
				const int index = rand_index(mt);
				tiny_dnn::vec_t& u = train_images[index];

				//{
				//	cpp_torch::Image* bmp = cpp_torch::vec_t2image(u, 3, image_height, image_width);
				//	cpp_torch::ImageWrite("aaa.bmp", bmp);
				//	delete bmp;
				//	//exit(0);
				//}

				std::string func = "";
				switch (rand(mt))
				{
				case 0:func = "GAMMA"; break;
				case 1:func = "RL"; break;
				case 2:func = "COLOR_NOIZE"; break;
				case 3:func = "NOIZE"; break;
				case 4:func = "ROTATION"; break;
				case 5:func = "SIFT"; break;
				}
				cpp_torch::ImageAugmentation(u, image_height, image_width, func);

				tiny_dnn::vec_t v(u.size());
				transform(u.begin(), u.end(), v.begin(),
					[=](float_t c) {return (c / channel_range); }
				);
				train_images.push_back(v);
				train_labels.push_back(train_labels[index]);

				//{
				//	tiny_dnn::vec_t v2(v.size());
				//	transform(v.begin(), v.end(), v2.begin(),
				//		[=](float_t c) {return (c * channel_range); }
				//	);
				//	cpp_torch::Image* bmp = cpp_torch::vec_t2image(v2, 3, image_height, image_width);
				//	cpp_torch::ImageWrite("bbb.bmp", bmp);
				//	delete bmp;
				//	exit(0);
				//}
			}

			const size_t sz2 = train_images.size();
#pragma omp parallel for
			for (int i = 0; i < sz2; i++)
			{
				for (int j = 0; j < train_images[i].size(); j++)
				{
					train_images[i][j] = (train_images[i][j] - mean) / stddiv;
				}
			}
			printf("Augmentation:%d -> %d\n", sz, sz2);
		}
	}
}
#endif
