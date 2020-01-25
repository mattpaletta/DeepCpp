#pragma once
#include "batch.hpp"

namespace deepcpp {
	namespace train {
		template<typename T>
		Batch<tensorflow::Tensor> OneHotEncode(const Batch<T> inputs, std::function<int(T)>&& f) {
			auto max = std::max_element(inputs.begin(), inputs.end(), [](Batch<T> a, Batch<T> b) {
				return a < b;
			});
			Batch<T> b;
			for (auto& i : inputs) {
				auto index = f(i);
				// TODO: Make tensors from index;
			}
			return b;
		};
	}
}
