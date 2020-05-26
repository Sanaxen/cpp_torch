#ifndef _UTILS_H
#define _UTILS_H

/*
	Copyright (c) 2013, Taiga Nomi and the respective contributors
	All rights reserved.

	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
*/
#pragma once
#include <string.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>

namespace tiny_dnn {
	typedef float float_t;
	typedef size_t label_t;
	typedef std::vector<float_t> vec_t;
	typedef std::vector<vec_t> tensor_t;

	/**
	 * error exception class for tiny-dnn
	 **/
	class nn_error : public std::exception {
	public:
		explicit nn_error(const std::string &msg) : msg_(msg) {}
		const char *what() const throw() override { return msg_.c_str(); }

	private:
		std::string msg_;
	};

	struct result {
		result() : num_success(0), num_total(0) {}

		float_t accuracy() const { return float_t(num_success * 100.0 / num_total); }

		template <typename Char, typename CharTraits>
		void print_summary(std::basic_ostream<Char, CharTraits> &os) const {
			os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total
				<< ")" << std::endl;
		}

		template <typename Char, typename CharTraits>
		void print_detail(std::basic_ostream<Char, CharTraits> &os) const {
			print_summary(os);
			auto all_labels = labels();

			os << std::setw(5) << "*"
				<< " ";
			for (auto c : all_labels) os << std::setw(5) << c << " ";
			os << std::endl;

			for (auto r : all_labels) {
				os << std::setw(5) << r << " ";
				const auto row_iter = confusion_matrix.find(r);
				for (auto c : all_labels) {
					int count = 0;
					if (row_iter != confusion_matrix.end()) {
						const auto &row = row_iter->second;
						const auto col_iter = row.find(c);
						if (col_iter != row.end()) {
							count = col_iter->second;
						}
					}
					os << std::setw(5) << count << " ";
				}
				os << std::endl;
			}
		}

		std::set<label_t> labels() const {
			std::set<label_t> all_labels;
			for (auto const &r : confusion_matrix) {
				all_labels.insert(r.first);
				for (auto const &c : r.second) all_labels.insert(c.first);
			}
			return all_labels;
		}

		int num_success;
		int num_total;
		std::map<label_t, std::map<label_t, int>> confusion_matrix;
	};
}
#endif
