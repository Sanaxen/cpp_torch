/*
Original is https://github.com/MyBoon/CSVparser
	MIT License
	Copyright (c) 2017 Romain Sylvian

This source is a single header version
	MIT License
	Copyright (c) 2018 Sanaxen
*/
#ifndef     _CSVPARSER_HPP_
# define    _CSVPARSER_HPP_

# include <stdexcept>
# include <string>
# include <vector>
# include <list>
# include <sstream>
# include <iostream>
# include <fstream>
# include <cstdlib>

namespace csv
{
    class Error : public std::runtime_error
    {

      public:
        Error(const std::string &msg):
          std::runtime_error(std::string("CSVparser : ").append(msg))
        {
        }
    };

    class Row
    {
    	public:
			Row::Row(const std::vector<std::string> &header)
				: _header(header) {}

			Row::~Row(void) {
				_values.clear();
				_values.shrink_to_fit();
			}

    	public:
			unsigned int Row::size(void) const
			{
				return _values.size();
			}

			void Row::push(const std::string &value)
			{
				_values.push_back(value);
			}

			bool Row::set(const std::string &key, const std::string &value)
			{
				std::vector<std::string>::const_iterator it;
				int pos = 0;

				for (it = _header.begin(); it != _header.end(); it++)
				{
					if (key == *it)
					{
						_values[pos] = value;
						return true;
					}
					pos++;
				}
				return false;
			}

    	private:
    		const std::vector<std::string> _header;
    		std::vector<std::string> _values;

        public:

            template<typename T>
            const T getValue(unsigned int pos) const
            {
                if (pos < _values.size())
                {
                    T res;
                    std::stringstream ss;
                    ss << _values[pos];
                    ss >> res;
                    return res;
                }
                throw Error("can't return this value (doesn't exist)");
            }
			const std::string Row::operator[](unsigned int valuePosition) const
			{
				if (valuePosition < _values.size())
					return _values[valuePosition];
				throw Error("can't return this value (doesn't exist)");
			}

			const std::string Row::operator[](const std::string &key) const
			{
				std::vector<std::string>::const_iterator it;
				int pos = 0;

				for (it = _header.begin(); it != _header.end(); it++)
				{
					if (key == *it)
						return _values[pos];
					pos++;
				}

				throw Error("can't return this value (doesn't exist)");
			}

			friend std::ostream &operator<<(std::ostream &os, const Row &row)
			{
				for (unsigned int i = 0; i != row._values.size(); i++)
					os << row._values[i] << " | ";

				return os;
			}

			friend std::ofstream &operator<<(std::ofstream &os, const Row &row)
			{
				for (unsigned int i = 0; i != row._values.size(); i++)
				{
					os << row._values[i];
					if (i < row._values.size() - 1)
						os << ",";
				}
				return os;
			}
    };

    enum DataType {
        eFILE = 0,
        ePURE = 1
    };

    class Parser
    {

    public:
		bool use_heade = true;
		Parser::Parser(const std::string &data, const DataType &type, char sep, bool use_heade_)
			: _type(type), _sep(sep), use_heade(use_heade_)
		{
			std::string line;
			if (type == eFILE)
			{
				_file = data;
				std::ifstream ifile(_file.c_str());
				if (ifile.is_open())
				{
					while (ifile.good())
					{
						getline(ifile, line);
						if (line != "")
							_originalFile.push_back(line);
					}
					ifile.close();

					if (_originalFile.size() == 0)
						throw Error(std::string("No Data in ").append(_file));

					parseHeader();
					parseContent();
				}
				else
					throw Error(std::string("Failed to open ").append(_file));
			}
			else
			{
				std::istringstream stream(data);
				while (std::getline(stream, line))
					if (line != "")
						_originalFile.push_back(line);
				if (_originalFile.size() == 0)
					throw Error(std::string("No Data in pure content"));

				parseHeader();
				parseContent();
			}
		}

		Parser::~Parser(void)
		{
			std::vector<Row *>::iterator it;

			for (it = _content.begin(); it != _content.end(); it++)
				delete *it;

			_content.clear();
			_content.shrink_to_fit();
		}

    public:
		Row &Parser::getRow(unsigned int rowPosition) const
		{
			if (rowPosition < _content.size())
				return *(_content[rowPosition]);
			throw Error("can't return this row (doesn't exist)");
		}
		unsigned int Parser::rowCount(void) const
		{
			return _content.size();
		}

		unsigned int Parser::columnCount(void) const
		{
			return _header.size();
		}
		std::vector<std::string> Parser::getHeader(void) const
		{
			return _header;
		}

		const std::string Parser::getHeaderElement(unsigned int pos) const
		{
			if (pos >= _header.size())
				throw Error("can't return this header (doesn't exist)");
			return _header[pos];
		}
		const std::string &Parser::getFileName(void) const
		{
			return _file;
		}

    public:
		bool Parser::deleteRow(unsigned int pos)
		{
			if (pos < _content.size())
			{
				delete *(_content.begin() + pos);
				_content.erase(_content.begin() + pos);
				return true;
			}
			return false;
		}
		bool Parser::addRow(unsigned int pos, const std::vector<std::string> &r)
		{
			Row *row = new Row(_header);

			for (auto it = r.begin(); it != r.end(); it++)
				row->push(*it);

			if (pos <= _content.size())
			{
				_content.insert(_content.begin() + pos, row);
				return true;
			}
			return false;
		}

		void Parser::sync(void) const
		{
			if (_type == DataType::eFILE)
			{
				std::ofstream f;
				f.open(_file, std::ios::out | std::ios::trunc);

				// header
				unsigned int i = 0;
				for (auto it = _header.begin(); it != _header.end(); it++)
				{
					f << *it;
					if (i < _header.size() - 1)
						f << ",";
					else
						f << std::endl;
					i++;
				}

				for (auto it = _content.begin(); it != _content.end(); it++)
					f << **it << std::endl;
				f.close();
			}
		}

    protected:
		void Parser::parseHeader(void)
		{
			std::stringstream ss(_originalFile[0]);
			std::string item;

			while (std::getline(ss, item, _sep))
			{
				if (item.c_str()[0] == '\"')
				{
					size_t len = strlen(item.c_str());
					if (item.c_str()[len - 1] == '\"')
					{
						_header.push_back(item);
						continue;
					}
					else
					{
						std::string wrk;
						std::getline(ss, wrk, _sep);
						size_t len = strlen(wrk.c_str());
						if (wrk.c_str()[len - 1] == '\"')
						{
							_header.push_back(item+"," + wrk);
							continue;
						}
					}
				}
				else
				{
					_header.push_back(item);
				}
			}
		}
		void Parser::parseContent(void)
		{
			std::vector<std::string>::iterator it;

			it = _originalFile.begin();
			if (use_heade)	it++; // skip header

			for (; it != _originalFile.end(); it++)
			{
				bool quoted = false;
				int tokenStart = 0;
				unsigned int i = 0;

				Row *row = new Row(_header);

				for (; i != it->length(); i++)
				{
					if (it->at(i) == '"')
						quoted = ((quoted) ? (false) : (true));
					else if (it->at(i) == ',' && !quoted)
					{
						row->push(it->substr(tokenStart, i - tokenStart));
						tokenStart = i + 1;
					}
				}

				//end
				row->push(it->substr(tokenStart, it->length() - tokenStart));

				// if value(s) missing
				if (row->size() != _header.size())
					throw Error("corrupted data !");
				_content.push_back(row);
			}
		}

    private:
        std::string _file;
        const DataType _type;
        const char _sep;
        std::vector<std::string> _originalFile;
        std::vector<std::string> _header;
        std::vector<Row *> _content;

    public:
		Row &Parser::operator[](unsigned int rowPosition) const
		{
			return Parser::getRow(rowPosition);
		}
	};
}

#endif /*!_CSVPARSER_HPP_*/
