#pragma once
#include "model.hpp"

namespace deepcpp {
	namespace train {
		template<typename T>
		using Item = std::pair<tensorflow::Tensor, T>;

		template<typename T>
		using Batch = std::vector<Item<T>>;

		template<typename T>
		std::pair<Batch<T>, Batch<T>> train_test_split(Batch<T> data, const float test_size);

		tensorflow::Status ReadTensorFromImageFile(tensorflow::ClientSession& session, tensorflow::Scope graph, std::string& file_name, tensorflow::Output image_tensor_var, tensorflow::Tensor& outTensor);

		tensorflow::Status ReadFileTensors(std::string& base_folder_name, std::vector<std::pair<std::string, float>> v_folder_label, std::vector<std::pair<tensorflow::Tensor, float>>& file_tensors);

		Batch<std::string> ReadBatches(std::string& base_folder_name, std::vector<std::pair<std::string, float>> v_folder_label, int batch_size);
	}
}
