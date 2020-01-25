#pragma once

#include <iostream>
#include <map>
#include <fstream>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/public/session.h"


namespace deepcpp {

class CNN {
private:
	tensorflow::Scope i_root; //graph for loading images into tensors
	const int image_side; //assuming quare picture
	const int image_channels; //RGB
	//load image vars
	tensorflow::Output file_name_var;
	tensorflow::Output image_tensor_var;
	//data augmentation
	tensorflow::Scope a_root;
	tensorflow::Output aug_tensor_input;
	tensorflow::Output aug_tensor_output;
	//training and validating the CNN
	tensorflow::Scope t_root; //graph
	std::unique_ptr<tensorflow::ClientSession> t_session;
	std::unique_ptr<tensorflow::Session> f_session;
	//CNN vars
	tensorflow::Output input_batch_var;
	std::string input_name = "input";
	tensorflow::Output input_labels_var;
	tensorflow::Output drop_rate_var; //use real drop rate in training and 1 in validating
	std::string drop_rate_name = "drop_rate";
	tensorflow::Output skip_drop_var; //use 0 in trainig and 1 in validating
	std::string skip_drop_name = "skip_drop";
	tensorflow::Output out_classification;
	std::string out_name = "output_classes";
	tensorflow::Output logits;
	//Network maps
	std::map<std::string, tensorflow::Output> m_vars;
	std::map<std::string, tensorflow::TensorShape> m_shapes;
	std::map<std::string, tensorflow::Output> m_assigns;
	//Loss variables
	std::vector<tensorflow::Output> v_weights_biases;
	std::vector<tensorflow::Operation> v_out_grads;
	tensorflow::Output out_loss_var;
	tensorflow::InputList MakeTransforms(int batch_size, tensorflow::Input a0, tensorflow::Input a1, tensorflow::Input a2, tensorflow::Input b0, tensorflow::Input b1, tensorflow::Input b2);
public:
	CNN(int side, int channels) : i_root(tensorflow::Scope::NewRootScope()), t_root(tensorflow::Scope::NewRootScope()), a_root(tensorflow::Scope::NewRootScope()), image_side(side), image_channels(channels) {}
	tensorflow::Status CreateGraphForImage(bool unstack);
	tensorflow::Status ReadTensorFromImageFile(std::string& file_name, tensorflow::Tensor& outTensor);
	tensorflow::Status ReadFileTensors(std::string& folder_name, std::vector<std::pair<std::string, float>> v_folder_label, std::vector<std::pair<tensorflow::Tensor, float>>& file_tensors);
	tensorflow::Status ReadBatches(std::string& folder_name, std::vector<std::pair<std::string, float>> v_folder_label, int batch_size, std::vector<tensorflow::Tensor>& image_batches, std::vector<tensorflow::Tensor>& label_batches);
	tensorflow::Input XavierInit(tensorflow::Scope scope, int in_chan, int out_chan, int filter_side = 0);
	tensorflow::Input AddConvLayer(std::string idx, tensorflow::Scope scope, int in_channels, int out_channels, int filter_side, tensorflow::Input input);
	tensorflow::Input AddDenseLayer(std::string idx, tensorflow::Scope scope, int in_units, int out_units, bool bActivation, tensorflow::Input input);
	tensorflow::Status CreateGraphForCNN(int filter_side);
	tensorflow::Status CreateOptimizationGraph(float learning_rate);
	tensorflow::Status Initialize();
	tensorflow::Status TrainCNN(tensorflow::Tensor& image_batch, tensorflow::Tensor& label_batch, std::vector<float>& results, float& loss);
	tensorflow::Status ValidateCNN(tensorflow::Tensor& image_batch, tensorflow::Tensor& label_batch, std::vector<float>& results);
	tensorflow::Status Predict(tensorflow::Tensor& image, int& result);
	tensorflow::Status FreezeSave(std::string& file_name);
	tensorflow::Status LoadSavedModel(std::string& file_name);
	tensorflow::Status PredictFromFrozen(tensorflow::Tensor& image, int& result);
	tensorflow::Status CreateAugmentGraph(int batch_size, int image_side, float flip_chances, float max_angles, float sscale_shift_factor);
	tensorflow::Status RandomAugmentBatch(tensorflow::Tensor& image_batch, tensorflow::Tensor& augmented_batch);
	tensorflow::Status WriteBatchToImageFiles(tensorflow::Tensor& image_batch, std::string folder_name, std::string image_name);
};

CNN ImageModel(const int image_size, const int image_channels);

}
