#pragma once

#include <boost/parameter.hpp>
#include <list>
#include <string>
#include <iostream>
#include <ios>

#include "model.hpp"

namespace deepcpp {

	class ModelBuilder {
	public:
		ModelBuilder() = default;

		ModelBuilder* Conv2D(const int a,
							const std::pair<std::size_t, std::size_t> b,
							std::string padding,
							std::string activation,
							const std::list<std::size_t> input_shape = {}) {
			return this;
		}

		ModelBuilder* MaxPooling2D(std::list<int> pool_size, std::list<int> strides) {
			return this;
		}

		ModelBuilder* Flatten() {
			return this;
		}

		ModelBuilder* Dense(int a, std::string b) {
			return this;
		}

		ModelBuilder* Build() {
			return this;
		}

		ModelBuilder* compile(std::string type, std::string optimizer, std::list<std::string> metrics) {
			return this;
		}

		ModelBuilder* fit(const deepcpp::train::Batch<std::string>& train_data, const deepcpp::train::Batch<std::string>& validation_data, const std::size_t epochs, const std::size_t batch_size, const bool verbose) {
			return this;
		}

		ModelBuilder* save(const std::string output_path) {
			return this;
		}
	};
}
