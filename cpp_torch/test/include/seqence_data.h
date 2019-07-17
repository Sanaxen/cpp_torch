#ifndef _SEQUENCE_DATA_H

#define _SEQUENCE_DATA_H

namespace cpp_torch
{
	namespace test
	{
		class SeqenceData
		{
			std::vector<tiny_dnn::vec_t> train_labels, test_labels;
			std::vector<tiny_dnn::vec_t> train_images, test_images;
			std::vector<tiny_dnn::vec_t> iX, iY;
			std::vector<tiny_dnn::vec_t> nX, nY;
			std::vector<tiny_dnn::vec_t> dataset;
			tiny_dnn::vec_t dataset_min, dataset_maxmin;

			int n_minibatch;
			int y_dim;
			int sequence_length;
			int out_sequence_length;

		public:
			SeqenceData() {}


			void Initialize(std::vector<tiny_dnn::vec_t>& dataset_, int y_dim_, int sequence_length_, int out_sequence_length_, int n_minibatch_)
			{
				dataset = dataset_;
				y_dim = y_dim_;
				sequence_length = sequence_length_;
				out_sequence_length = out_sequence_length_;
				n_minibatch = n_minibatch_;


				for (int i = 0; i < dataset.size(); i++)
				{
					tiny_dnn::vec_t image;
					image.push_back(dataset[i][0]);
					iX.push_back(image);

					tiny_dnn::vec_t label;
					for (int k = 1; k < dataset[i].size(); k++)
					{
						label.push_back(dataset[i][k]);
					}
					iY.push_back(label);
				}
				printf("y_dim:%d == %d\n", y_dim, iY[0].size());

				nY = iY;
				//data normalize
				tiny_dnn::cpp_torch::normalizeMinMax(nY, dataset_min, dataset_maxmin);

				data_set(0.3);
			}
			void get_train_data(std::vector<tiny_dnn::vec_t>& train_images_, std::vector<tiny_dnn::vec_t>& train_labels_)
			{
				train_images_ = train_images;
				train_labels_ = train_labels;
			}
			void get_test_data(std::vector<tiny_dnn::vec_t>& test_images_, std::vector<tiny_dnn::vec_t>& test_labels_)
			{
				test_images_ = test_images;
				test_labels_ = test_labels;
			}

			void add_seq(tiny_dnn::vec_t& y, tiny_dnn::vec_t& Y)
			{
				for (int i = 0; i < y.size(); i++)
				{
					Y.push_back(y[i]);
				}
			}
			tiny_dnn::vec_t seq_vec(tiny_dnn::tensor_t& ny, int start)
			{

				tiny_dnn::vec_t seq;
				for (int k = 0; k < sequence_length; k++)
				{
					add_seq(ny[start + k], seq);
				}
				return seq;
			}

			int error = 0;
			int data_set(float test = 0.3f)
			{
				train_images.clear();
				train_labels.clear();
				test_images.clear();
				test_images.clear();

				printf("n_minibatch:%d sequence_length:%d\n", n_minibatch, sequence_length);
				printf("out_sequence_length:%d\n", out_sequence_length);

				size_t dataAll = iY.size() - sequence_length - out_sequence_length;
				printf("dataset All:%d->", dataAll);
				size_t test_Num = dataAll * test;
				int datasetNum = dataAll - test_Num;
				//datasetNum = (size_t)((float)datasetNum / (float)kTrainBatchSize);
				//datasetNum = kTrainBatchSize*datasetNum;
				//test_Num = dataAll - datasetNum;

				//datasetNum = datasetNum - datasetNum % n_minibatch;
				if (datasetNum == 0 || datasetNum < n_minibatch)
				{
					printf("Too many min_batch or Sequence length\n");
					error = -1;
					return error;
				}
				size_t train_num_max = datasetNum;
				printf("train:%d test:%d\n", datasetNum, test_Num);

				for (int i = 0; i < train_num_max; i++)
				{
					train_images.push_back(seq_vec(nY, i));


					tiny_dnn::vec_t y;
					for (int j = 0; j < out_sequence_length; j++)
					{
						const auto& ny = nY[i + sequence_length + j];
						for (int k = 0; k < y_dim; k++)
						{
							y.push_back(ny[k]);
						}
					}

					train_labels.push_back(y);
				}

				for (int i = train_num_max; i < dataAll; i++)
				{
					test_images.push_back(seq_vec(nY, i));

					tiny_dnn::vec_t y;
					for (int j = 0; j < out_sequence_length; j++)
					{
						const auto& ny = nY[i + sequence_length + j];
						for (int k = 0; k < y_dim; k++)
						{
							y.push_back(ny[k]);
						}
					}

					test_labels.push_back(y);
				}
				printf("train:%d test:%d\n", train_images.size(), test_images.size());
				return 0;
			}


			void sequence_test(
#ifndef TEST
				cpp_torch::network_torch<Net> nn
#else
				cpp_torch::network_torch<cpp_torch::Net> nn
#endif
			)
			{
				int prophecy = 0;
				std::vector<tiny_dnn::vec_t> train(nY.size() + prophecy + sequence_length + out_sequence_length);
				std::vector<tiny_dnn::vec_t> predict(nY.size() + prophecy + sequence_length + out_sequence_length);
				std::vector<tiny_dnn::vec_t> YY = nY;

				YY.resize(nY.size() + prophecy + sequence_length + out_sequence_length);
				for (int i = nY.size(); i < YY.size(); i++)
				{
					YY[i].resize(nY[0].size());
				}
				for (int i = 0; i < YY.size(); i++)
				{
					train[i].resize(nY[0].size());
					predict[i].resize(nY[0].size());
				}


				//The first sequence is as input
				for (int j = 0; j < sequence_length; j++)
				{
					tiny_dnn::vec_t y(y_dim);
					for (int k = 0; k < y_dim; k++)
					{
						y[k] = nY[j][k];
					}
					train[j] = predict[j] = y;
				}

#pragma omp parallel for
				for (int i = 0; i < iY.size() + prophecy; i++)
				{
					tiny_dnn::vec_t next_y = nn.predict(seq_vec(YY, i));
					//output sequence_length 
					for (int j = 0; j < out_sequence_length; j++)
					{
						tiny_dnn::vec_t yy(y_dim);
						tiny_dnn::vec_t y(y_dim);
						for (int k = 0; k < y_dim; k++)
						{
							y[k] = YY[i + sequence_length + j][k];
							yy[k] = next_y[y_dim*j + k];
						}
						train[i + sequence_length + j] = y;
						predict[i + sequence_length + j] = yy;
					}
				}



				FILE* fp = fopen("predict.dat", "w");
				float dt = iX[1][0] - iX[0][0];
				float t = 0;
				for (int i = 0; i < sequence_length; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < 3; k++)
					{
						fprintf(fp, " NaN %f", train[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}

				for (int i = sequence_length - 1; i < train_images.size() - sequence_length; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < 3; k++)
					{
						fprintf(fp, " %f %f", predict[i][k] * dataset_maxmin[k] + dataset_min[k], train[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}

				for (int i = train_images.size() - sequence_length - 1; i < iY.size() - sequence_length; i++)
				{
					fprintf(fp, "%f", t);
					if (i < train_images.size())
					{
						for (int k = 0; k < 3; k++)
						{
							fprintf(fp, " %f %f", predict[i][k] * dataset_maxmin[k] + dataset_min[k], train[i][k] * dataset_maxmin[k] + dataset_min[k]);
						}
					}
					else
					{
						for (int k = 0; k < 3; k++)
						{
							fprintf(fp, " %f NaN", predict[i][k] * dataset_maxmin[k] + dataset_min[k], train[i][k] * dataset_maxmin[k] + dataset_min[k]);
						}
					}
					fprintf(fp, "\n");
					t += dt;
				}

				for (int i = iY.size() - sequence_length - 1; i < iY.size() + prophecy; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < 3; k++)
					{
						fprintf(fp, " %f NaN", predict[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}
				fprintf(fp, "\n");

				fclose(fp);
			}
		};
	}
}
#endif
