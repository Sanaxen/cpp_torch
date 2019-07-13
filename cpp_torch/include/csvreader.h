#ifndef _CSVREADER_H
#define _CSVREADER_H
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#include "third_party/CSVparser/CSVparser_single.hpp"
class CSVReader
{
	csv::Parser* csvfile_;
public:
	CSVReader(char* filename, char separator = ',', bool use_header = true)
	{
		try
		{
			csvfile_ = new csv::Parser(std::string(filename), csv::DataType::eFILE, separator, use_header);
		}
		catch (csv::Error& e)
		{
			printf("CSVReader error:%s\n", e.what());
		}
	}
	CSVReader(std::string filename, char separator = ',', bool use_header = true)
	{
		try
		{
			csvfile_ = new csv::Parser(filename, csv::DataType::eFILE, separator, use_header);
		}
		catch (csv::Error& e)
		{
			printf("CSVReader error:%s\n", e.what());
		}
	}
	~CSVReader()
	{
		if (csvfile_) delete csvfile_;
		csvfile_ = NULL;
	}

	void clear()
	{
		delete csvfile_;
		csvfile_ = NULL;
	}

	std::vector<std::string> getHeader()
	{
		return csvfile_->getHeader();
	}
	std::string getHeader(const int index)
	{
		return csvfile_->getHeaderElement(index);
	}

	std::vector<int> empty_cell;
	std::vector<int> nan_cell;
	tiny_dnn::tensor_t toTensor(int rowMax = -1)
	{
		std::vector<int> empty;
		std::vector<int> nan;

		tiny_dnn::tensor_t& M = toTensor(rowMax, empty, nan);
		empty_cell = empty;
		nan_cell = nan;
		return M;
	}
	tiny_dnn::tensor_t toTensor_removeEmptyRow()
	{
		std::vector<int> empty;
		std::vector<int> nan;

		tiny_dnn::tensor_t& mat = toTensor(-1, empty, nan);
		if (empty.size() == 0 || mat.size() == 1)
		{
			return mat;
		}
		int m = mat.size();

		do
		{
			m--;
			empty.clear();
			nan.clear();
			mat = toTensor(m, empty, nan);
		} while (empty.size() && m >= 1);

		return mat;
	}

	std::vector<std::string> ItemCol(int col)
	{
		csv::Parser& csvfile = *csvfile_;

		int m = csvfile.rowCount();
		int n = csvfile.columnCount();

		std::vector<std::string> items;
		for (int i = 0; i < m; i++)
		{
			items.push_back(csvfile[i][col]);
		}
		return items;
	}

	tiny_dnn::tensor_t toTensor(
		int rowMax,
		std::vector<int>& empty, std::vector<int>& nan)
	{
		csv::Parser& csvfile = *csvfile_;

		int m = csvfile.rowCount();
		int n = csvfile.columnCount();

		printf("m,n:%d, %d\n", m, n);
		fflush(stdout);
		if (rowMax > 0)
		{
			m = rowMax;
		}

		tiny_dnn::tensor_t mat(m, tiny_dnn::vec_t(n,0));

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				csv::Row& row = csvfile.getRow(i);

				if (csvfile[i][j] == "")
				{
					empty.push_back(i*n + j);
					mat[i][j] = 0.0;
					continue;
				}

				std::string cell = row[j];
				const char* value = cell.c_str();

				double v = 0.0;
				if (*value == '+' || *value == '-' || *value == '.' || isdigit(*value))
				{
					bool no_number = false;
					int dot = 0;
					char* p = (char*)cell.c_str();
					while (isspace(*p)) p++;
					if (*p == '+' || *p == '-') p++;
					if (*p == '.')
					{
						p++;
						dot++;
					}
					if (!isdigit(*p)) no_number = true;
					if (!no_number)
					{
						while (isdigit(*p)) p++;
						if (*p == '.' && dot == 0)
						{
							p++;
							dot++;
						}
						if (dot == 2) no_number = true;
						if (!no_number)
						{
							if (isdigit(*p))
							{
								while (isdigit(*p)) p++;
							}
							if (*p == 'E' || *p == 'e')
							{
								p++;
								if (*p == '+' || *p == '-')
								{
									p++;
								}
							}
							if (isdigit(*p))
							{
								while (isdigit(*p)) p++;
							}
						}
						while (isspace(*p)) p++;
						if (*p != '\0') no_number = true;
					}
					if (!no_number)
					{
						sscanf(value, "%lf", &v);
					}
					else
					{
						nan.push_back(i*n + j);
						v = 0;
					}
				}
				else
				{
					nan.push_back(i*n + j);
					v = 0;
				}
				mat[i][j] = v;
			}
		}
		printf("empty cell:%d\n", empty.size());
		printf("nan cell:%d\n", nan.size());
		return mat;
	}
};


#endif
