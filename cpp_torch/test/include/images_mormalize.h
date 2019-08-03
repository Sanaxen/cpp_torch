#ifndef _IMAGES_NORMALIZE_H
#define _IMAGES_NORMALIZE_H

namespace cpp_torch
{
	namespace test
	{
		void images_normalize(tiny_dnn::vec_t& image, float& mean, float& stddiv)
		{
			mean = 0.0;
			stddiv = 0.0;
			for (int k = 0; k < image.size(); k++)
			{
				mean += image[k];
			}
			mean /= image.size();

			for (int k = 0; k < image.size(); k++)
			{
				stddiv += (image[k] - mean)*(image[k] - mean);
			}
			stddiv = sqrt(stddiv / image.size());

			//printf("mean:%f stddiv:%f\n", mean, stddiv);
#pragma omp parallel for
			for (int k = 0; k < image.size(); k++)
			{
				image[k] = (image[k] - mean) / (stddiv + 1.0e-12);
			}
		}
		void images_normalize_(std::vector<tiny_dnn::vec_t>& images)
		{
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				float mean = 0.0;
				float stddiv = 0.0;
				images_normalize(images[i], mean, stddiv);
			}
		}

		void images_normalize(std::vector<tiny_dnn::vec_t>& images, float& mean, float& stddiv)
		{
			mean = 0.0;
			stddiv = 0.0;
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					mean += images[i][k];
				}
			}
			mean /= images.size()*images[0].size();

			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					stddiv += (images[i][k] - mean)*(images[i][k] - mean);
				}
			}
			stddiv = sqrt(stddiv / (images.size()*images[0].size()));

			//printf("mean:%f stddiv:%f\n", mean, stddiv);
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = (images[i][k] - mean) / (stddiv + 1.0e-12);
				}
			}
		}
	}
}
#endif
