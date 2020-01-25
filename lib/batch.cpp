#include "batch.hpp"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/cc/ops/standard_ops.h"

tensorflow::Status deepcpp::train::ReadTensorFromImageFile(tensorflow::ClientSession& session, tensorflow::Scope graph, std::string& file_name, tensorflow::Output image_tensor_var, tensorflow::Tensor& outTensor) {
	auto file_name_var = tensorflow::ops::Placeholder(graph.WithOpName("input"), tensorflow::DT_STRING);
	if (!tensorflow::str_util::EndsWith(file_name, ".jpg") && !tensorflow::str_util::EndsWith(file_name, ".jpeg")) {
		return tensorflow::errors::InvalidArgument("Image must be jpeg encoded");
	}
	std::vector<tensorflow::Tensor> out_tensors;
	TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {image_tensor_var}, &out_tensors));
	outTensor = out_tensors[0]; // shallow copy
	return tensorflow::Status::OK();
}

tensorflow::Status deepcpp::CNN::ReadFileTensors(std::string& base_folder_name, std::vector<std::pair<std::string, float>> v_folder_label, std::vector<std::pair<tensorflow::Tensor, float>>& file_tensors) {
	//validate the folder
	tensorflow::Env* penv = tensorflow::Env::Default();
	TF_RETURN_IF_ERROR(penv->IsDirectory(base_folder_name));
	//get the files
	bool b_shuffle = false;
	for(auto p: v_folder_label) {
		std::string folder_name = tensorflow::io::JoinPath(base_folder_name, p.first);
		TF_RETURN_IF_ERROR(penv->IsDirectory(folder_name));
		std::vector<std::string> file_names;
		TF_RETURN_IF_ERROR(penv->GetChildren(folder_name, &file_names));
		for (std::string file: file_names) {
			std::string full_path = tensorflow::io::JoinPath(folder_name, file);
			tensorflow::Tensor i_tensor;
			TF_RETURN_IF_ERROR(ReadTensorFromImageFile(full_path, i_tensor));
			size_t s = file_tensors.size();
			if (b_shuffle) {
				//suffle the images
				int i = rand() % s;
				file_tensors.emplace(file_tensors.begin()+i, std::make_pair(i_tensor, p.second));
			} else {
				file_tensors.push_back(std::make_pair(i_tensor, p.second));
			}
		}
		b_shuffle = true;
	}
	return tensorflow::Status::OK();
}


deepcpp::train::Batch<std::string> deepcpp::train::ReadBatches(std::string& base_folder_name, std::vector<std::pair<std::string, float>> v_folder_label, int batch_size) {
	std::vector<tensorflow::Tensor> image_batches;
	std::vector<tensorflow::Tensor> label_batches;

	deepcpp::train::Batch<std::string> batches;

	std::vector<std::pair<tensorflow::Tensor, float>> all_files_tensors;
	TF_CHECK_OK(ReadFileTensors(base_folder_name, v_folder_label, all_files_tensors));
	auto start_i = all_files_tensors.begin();
	auto end_i = all_files_tensors.begin() + batch_size;
	std::size_t num_batches = all_files_tensors.size() / batch_size;
	if (num_batches * batch_size < all_files_tensors.size()) {
		num_batches++;
	}
	for (int b = 0; b < num_batches; b++) {
		if (end_i > all_files_tensors.end()) {
			end_i = all_files_tensors.end();
		}

		std::vector<std::pair<tensorflow::Tensor, float>> one_batch(start_i, end_i);

		// need to break the pairs
		std::vector<tensorflow::Input> one_batch_image, one_batch_lbl;
		for (auto p: one_batch) {
			one_batch_image.push_back(tensorflow::Input(p.first));
			tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
			t.scalar<float>()(0) = p.second;
			one_batch_lbl.push_back(tensorflow::Input(t));
		}
		tensorflow::InputList one_batch_inputs(one_batch_image);
		tensorflow::InputList one_batch_labels(one_batch_lbl);
		auto root = tensorflow::Scope::NewRootScope();
		auto stacked_images = tensorflow::ops::Stack(root, one_batch_inputs);
		auto stacked_labels = tensorflow::ops::Stack(root, one_batch_labels);
		TF_CHECK_OK(root.status());
		tensorflow::ClientSession session(root);
		std::vector<tensorflow::Tensor> out_tensors;
		TF_CHECK_OK(session.Run({}, {stacked_images, stacked_labels}, &out_tensors));
		image_batches.push_back(out_tensors[0]);
		label_batches.push_back(out_tensors[1]);
		start_i = end_i;
		if (start_i == all_files_tensors.end()) {
			break;
		}
		end_i = start_i+batch_size;
	}
	return batches;
}
