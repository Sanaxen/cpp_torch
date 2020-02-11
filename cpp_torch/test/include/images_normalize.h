#ifndef _IMAGES_NORMALIZE_H
#define _IMAGES_NORMALIZE_H

namespace cpp_torch
{
	namespace test
	{
		// mean 0 and variance 1 
		/*
		 * @param image [in/out] image data
		 * @param image [out] image data mean
		 * @param stddiv [out] image data variance
		 */
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

		// mean 0 and variance 1 
		/*
		 * @param image [in/out] image data vector
		 */
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

		// mean 0 and variance 1 
		/*
		 * @param image [in/out] image data vector
		 * @param image [out] image data vector mean
		 * @param stddiv [out] image data vector variance
		 */
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
		// mean 0 and variance 1 
		/*
		 * @param image [in/out] image data vector
		 * @param image [in] image data vector mean
		 * @param stddiv [in] image data vector variance
		 */
		void images_normalize_(std::vector<tiny_dnn::vec_t>& images, const float mean, const float stddiv)
		{
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

		// [-1, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_11(std::vector<tiny_dnn::vec_t>& images)
		{
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = images[i][k]/ 128.0 - 1.0;
				}
			}
		}

#if 10
		// [-1, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_11(std::vector<tiny_dnn::vec_t>& images, float& max, float& min)
		{
			std::vector<float_t> max_(images.size());
			std::vector<float_t> min_(images.size());

			for (int i = 0; i < images.size(); i++)
			{
				max_[i] = -999999.0;
				min_[i] = 999999.0;
				for (int k = 0; k < images[i].size(); k++)
				{
					if (min_[i] > images[i][k]) min_[i] = images[i][k];
					if (max_[i] < images[i][k]) max_[i] = images[i][k];
				}
			}

			max = 0, min = 0;
			for (int i = 0; i < images.size(); i++)
			{
				max += max_[i];
				min += min_[i];
			}
			max /= images.size();
			min /= images.size();
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = 2.0*(images[i][k] - min) / (max - min) - 1.0;
				}
			}
		}
#else
		// [-1, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_11(std::vector<tiny_dnn::vec_t>& images,  float& max_, float& min_)
		{
			max_ = -9999999.0;
			min_ = 9999999.0;
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					if (min_ > images[i][k]) min_ = images[i][k];
					if (max_ < images[i][k]) max_ = images[i][k];
				}
			}

#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = 2.0*(images[i][k] - min_)/ (max_ - min_) - 1.0;
				}
			}
		}
#endif

		// [-1, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_11_(std::vector<tiny_dnn::vec_t>& images, const float max_, const float min_)
		{
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = 2.0*(images[i][k] - min_) / (max_ - min_) - 1.0;
				}
			}
		}
		// [0, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_01(std::vector<tiny_dnn::vec_t>& images, float& max_, float& min_)
		{
			max_ = -9999999.0;
			min_ = 9999999.0;
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					if (min_ > images[i][k]) min_ = images[i][k];
					if (max_ < images[i][k]) max_ = images[i][k];
				}
			}

#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = (images[i][k] - min_) / (max_ - min_);
				}
			}
		}
		// [0, 1]
		/*
		* @param image [in/out]  image data vector
		*/
		void images_normalize_01_(std::vector<tiny_dnn::vec_t>& images, const float max_, const float min_)
		{
#pragma omp parallel for
			for (int i = 0; i < images.size(); i++)
			{
				for (int k = 0; k < images[i].size(); k++)
				{
					images[i][k] = (images[i][k] - min_) / (max_ - min_);
				}
			}
		}

	}
}
#endif
