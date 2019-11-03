/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
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


		public:
			int prophecy = 0;
			float testsize = 0.3;
			int normalize_type = 1;
			int n_minibatch;
			int y_dim;
			int x_dim;
			int sequence_length;
			int out_sequence_length;

			SeqenceData() {}

			bool add_explanatory_variable = false;	// true to add x to the explanatory variable

			void Initialize(const std::string &data_dir_path, const std::string& filename="sample.csv")
			{
				/*
				csv data format
				t1, data11, data12, ... , data1n
				t2, data21, data22, ... , data2n
				...
				tm, datam1, datam2, ... , datamn

				ti = time index
				*/
				printf("read[%s]\n", (data_dir_path + "/" + filename).c_str());
				CSVReader csv(data_dir_path + "/"+ filename, ',', false);
				std::vector<tiny_dnn::vec_t>& dataset = csv.toTensor();

				/*
				t0: y0(1),..,y0(y_dim), x0(1),...,x0(x_dim)
				t1: y1(1),..,y1(y_dim), x1(1),...,x1(x_dim)
				...
				tn: yn(1),..,yn(y_dim), xn(1),...,xn(x_dim)

				yi(1)     = dataset[i][0]
				yi(2)     = dataset[i][1]
				...
				yi(y_dim) = dataset[i][y_dim-1]
				xi(1)     = dataset[i][y_dim]
				xi(2)     = dataset[i][y_dim+1]
				...
				xi(x_dim) = dataset[i][y_dim+x_dim-1]

				x=Explanatory variable
				y=Objective variable

				yt+1 = F( yt, yt, .., yt-seqlen, xt+1)
				*/

				//[ yt, xt+1]
				std::vector<tiny_dnn::vec_t> yy;
				for (int i = 0; i < dataset.size() - 1; i++)
				{
					tiny_dnn::vec_t y;
					y.push_back(dataset[i][0]);	//time index
					for (int k = 1; k <= y_dim; k++)
					{
						y.push_back(dataset[i][k]);
					}
					if (add_explanatory_variable)
					{
						//Explanatory variable time sifting
						for (int k = 0; k < x_dim; k++)
						{
							y.push_back(dataset[i + 1][y_dim + k]);
						}
					}
					yy.push_back(y);
				}


				for (int i = 0; i < yy.size(); i++)
				{
					tiny_dnn::vec_t image;
					image.push_back(yy[i][0]);
					iX.push_back(image);

					tiny_dnn::vec_t label;
					for (int k = 1; k < yy[i].size(); k++)
					{
						label.push_back(yy[i][k]);
					}
					iY.push_back(label);
				}

				nY = iY;
				if (normalize_type == 1)
				{
					//data normalize
					tiny_dnn::cpp_torch::normalizeMinMax(nY, dataset_min, dataset_maxmin);
				}
				if (normalize_type == 2)
				{
					//data normalize
					tiny_dnn::cpp_torch::normalizeZ(nY, dataset_min, dataset_maxmin);
				}
				data_set(this->testsize);
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
				printf("(test:%f)\n", test);
				printf("test_Num:%d\n", test_Num);
				printf("datasetNum:%d\n", datasetNum);

				//datasetNum = datasetNum - datasetNum % n_minibatch;
				if (datasetNum == 0 || datasetNum < n_minibatch)
				{
					printf("Too many min_batch or Sequence length\n");
					error = -1;
					return error;
				}
				size_t train_num_max = datasetNum;
				if (datasetNum < 0)
				{
					printf("Insufficient data length\n");
					error = -1;
					return error;
				}
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


			template <typename Model>
			void sequence_test(	Model nn)
			{
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
					if (i >= train_images.size() - sequence_length)
					{
						for (int j = 0; j < out_sequence_length; j++)
						{
							for (int k = 0; k < y_dim; k++)
							{
								YY[i + sequence_length + j][k] = predict[i + sequence_length + j][k];
							}
						}
					}
				}


				FILE* fp = fopen("test.dat", "w");
				float dt = iX[1][0] - iX[0][0];
				float t = 0;
				for (int i = 0; i < sequence_length; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < y_dim; k++)
					{
						fprintf(fp, " %f NaN", train[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}
				fclose(fp);

				fp = fopen("predict1.dat", "w");
				for (int i = sequence_length-1; i < train_images.size(); i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < y_dim; k++)
					{
						fprintf(fp, " %f %f", train[i][k] * dataset_maxmin[k] + dataset_min[k], predict[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}
				fclose(fp);

				fp = fopen("predict2.dat", "w");
				int sz = nY.size();

				for (int i = train_images.size()-1; i < sz; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < y_dim; k++)
					{
						fprintf(fp, " %f %f", train[i][k] * dataset_maxmin[k] + dataset_min[k], predict[i][k] * dataset_maxmin[k] + dataset_min[k]);
					}
					fprintf(fp, "\n");
					t += dt;
				}
				fclose(fp);

				fp = fopen("prophecy.dat", "w");
				for (int i = sz-1; i < iY.size() + prophecy; i++)
				{
					fprintf(fp, "%f", t);
					for (int k = 0; k < y_dim; k++)
					{
						fprintf(fp, " NaN %f", predict[i][k] * dataset_maxmin[k] + dataset_min[k]);
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
