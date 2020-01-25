#include "model.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/cc/ops/image_ops.h"

deepcpp::CNN ImageModel(const int image_size, const int image_channels) {
	deepcpp::CNN model(image_size, image_channels);
	auto s = model.CreateGraphForImage(true);
	TF_CHECK_OK(s);
	return model;
}

tensorflow::Status deepcpp::CNN::CreateGraphForImage(bool unstack) {
	file_name_var = tensorflow::ops::Placeholder(this->i_root.WithOpName("input"), tensorflow::DT_STRING);
	auto file_reader = tensorflow::ops::ReadFile(this->i_root.WithOpName("file_readr"), file_name_var);

	auto image_reader = tensorflow::ops::DecodeJpeg(this->i_root.WithOpName("jpeg_reader"), file_reader, tensorflow::ops::DecodeJpeg::Channels(image_channels));

	auto float_caster = tensorflow::ops::Cast(this->i_root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
	auto dims_expander = tensorflow::ops::ExpandDims(this->i_root.WithOpName("dim"), float_caster, 0);
	auto resized = tensorflow::ops::ResizeBilinear(this->i_root.WithOpName("size"), dims_expander, tensorflow::ops::Const(this->i_root, {image_side, image_side}));
	auto div = tensorflow::ops::Div(this->i_root.WithOpName("normalized"), resized, {255.f});

	if (unstack) {
		auto output_list = tensorflow::ops::Unstack(i_root.WithOpName("fold"), div, 1);
		image_tensor_var = output_list.output[0];
	} else {
		image_tensor_var = div;
	}

	return i_root.status();
}

tensorflow::Input deepcpp::CNN::XavierInit(tensorflow::Scope scope, int in_chan, int out_chan, int filter_side) {
	float std;
	tensorflow::Tensor t;
	if(filter_side == 0) { //Dense
		std = sqrt(6.f/(in_chan+out_chan));
		tensorflow::Tensor ts(tensorflow::DT_INT64, {2});
		auto v = ts.vec<tensorflow::int64>();
		v(0) = in_chan;
		v(1) = out_chan;
		t = ts;
	} else { //Conv
		std = std::sqrt(6.f / (filter_side * filter_side * (in_chan + out_chan)));
		tensorflow::Tensor ts(tensorflow::DT_INT64, {4});
		auto v = ts.vec<tensorflow::int64>();
		v(0) = filter_side;
		v(1) = filter_side;
		v(2) = in_chan;
		v(3) = out_chan;
		t = ts;
	}
	auto rand = tensorflow::ops::RandomUniform(scope, t, tensorflow::DT_FLOAT);
	return tensorflow::ops::Multiply(scope, tensorflow::ops::Sub(scope, rand, 0.5f), std*2.f);
}

tensorflow::Input deepcpp::CNN::AddConvLayer(std::string idx, tensorflow::Scope scope, int in_channels, int out_channels, int filter_side, tensorflow::Input input) {
	tensorflow::TensorShape sp({filter_side, filter_side, in_channels, out_channels});
	m_vars["W"+idx] = tensorflow::ops::Variable(scope.WithOpName("W"), sp, tensorflow::DT_FLOAT);
	m_shapes["W"+idx] = sp;
	m_assigns["W"+idx+"_assign"] = tensorflow::ops::Assign(scope.WithOpName("W_assign"), m_vars["W"+idx], XavierInit(scope, in_channels, out_channels, filter_side));
	sp = {out_channels};
	m_vars["B"+idx] = tensorflow::ops::Variable(scope.WithOpName("B"), sp, tensorflow::DT_FLOAT);
	m_shapes["B"+idx] = sp;
	m_assigns["B"+idx+"_assign"] = tensorflow::ops::Assign(scope.WithOpName("B_assign"), m_vars["B"+idx], tensorflow::Input::Initializer(0.f, sp));
	auto conv = tensorflow::ops::Conv2D(scope.WithOpName("Conv"), input, m_vars["W"+idx], {1, 1, 1, 1}, "SAME");
	auto bias = tensorflow::ops::BiasAdd(scope.WithOpName("Bias"), conv, m_vars["B"+idx]);
	auto relu = tensorflow::ops::Relu(scope.WithOpName("Relu"), bias);
	return tensorflow::ops::MaxPool(scope.WithOpName("Pool"), relu, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");
}

tensorflow::Input deepcpp::CNN::AddDenseLayer(std::string idx, tensorflow::Scope scope, int in_units, int out_units, bool bActivation, tensorflow::Input input) {
	tensorflow::TensorShape sp = {in_units, out_units};
	m_vars["W"+idx] = tensorflow::ops::Variable(scope.WithOpName("W"), sp, tensorflow::DT_FLOAT);
	m_shapes["W"+idx] = sp;
	m_assigns["W"+idx+"_assign"] = tensorflow::ops::Assign(scope.WithOpName("W_assign"), m_vars["W"+idx], XavierInit(scope, in_units, out_units));
	sp = {out_units};
	m_vars["B"+idx] = tensorflow::ops::Variable(scope.WithOpName("B"), sp, tensorflow::DT_FLOAT);
	m_shapes["B"+idx] = sp;
	m_assigns["B"+idx+"_assign"] = tensorflow::ops::Assign(scope.WithOpName("B_assign"), m_vars["B"+idx], tensorflow::Input::Initializer(0.f, sp));
	auto dense = tensorflow::ops::Add(scope.WithOpName("Dense_b"), tensorflow::ops::MatMul(scope.WithOpName("Dense_w"), input, m_vars["W"+idx]), m_vars["B"+idx]);
	if(bActivation)
		return tensorflow::ops::Relu(scope.WithOpName("Relu"), dense);
	else
		return dense;
}

tensorflow::Status deepcpp::CNN::CreateGraphForCNN(int filter_side) {
	//input image is batch_sizex150x150x3
	input_batch_var = tensorflow::ops::Placeholder(t_root.WithOpName("input"), tensorflow::DT_FLOAT);
	drop_rate_var = tensorflow::ops::Placeholder(t_root.WithOpName("drop_rate"), tensorflow::DT_FLOAT);//see class member for help
	skip_drop_var = tensorflow::ops::Placeholder(t_root.WithOpName("skip_drop"), tensorflow::DT_FLOAT);//see class member for help

	// Start Conv+Maxpool No 1. filter size 3x3x3 and we have 32 filters
	tensorflow::Scope scope_conv1 = t_root.NewSubScope("Conv1_layer");
	int in_channels = image_channels;
	int out_channels = 32;
	auto pool1 = this->AddConvLayer("1", scope_conv1, in_channels, out_channels, filter_side, input_batch_var);
	int new_side = std::ceil((float)image_side / 2); //max pool is reducing the size by factor of 2

	// Conv+Maxpool No 2
	tensorflow::Scope scope_conv2 = t_root.NewSubScope("Conv2_layer");
	in_channels = out_channels;
	out_channels = 64;
	auto pool2 = this->AddConvLayer("2", scope_conv2, in_channels, out_channels, filter_side, pool1);
	new_side = std::ceil((float)new_side / 2);

	// Conv+Maxpool No 3
	tensorflow::Scope scope_conv3 = t_root.NewSubScope("Conv3_layer");
	in_channels = out_channels;
	out_channels = 128;
	auto pool3 = this->AddConvLayer("3", scope_conv3, in_channels, out_channels, filter_side, pool2);
	new_side = ceil((float)new_side / 2);

	// Conv+Maxpool No 4
	tensorflow::Scope scope_conv4 = t_root.NewSubScope("Conv4_layer");
	in_channels = out_channels;
	out_channels = 128;
	auto pool4 = this->AddConvLayer("4", scope_conv4, in_channels, out_channels, filter_side, pool3);
	new_side = ceil((float)new_side / 2);

	// Flatten
	tensorflow::Scope flatten = t_root.NewSubScope("flat_layer");
	int flat_len = new_side * new_side * out_channels;
	auto flat = tensorflow::ops::Reshape(flatten, pool4, {-1, flat_len});

	// Dropout
	tensorflow::Scope dropout = t_root.NewSubScope("Dropout_layer");
	auto rand = tensorflow::ops::RandomUniform(dropout, tensorflow::ops::Shape(dropout, flat), tensorflow::DT_FLOAT);

	// binary = floor(rand + (1 - drop_rate) + skip_drop);
	auto binary = tensorflow::ops::Floor(dropout, tensorflow::ops::Add(dropout, rand, tensorflow::ops::Add(dropout, tensorflow::ops::Sub(dropout, 1.f, drop_rate_var), skip_drop_var)));
	auto after_drop = tensorflow::ops::Multiply(dropout.WithOpName("dropout"), tensorflow::ops::Div(dropout, flat, drop_rate_var), binary);
	// Dense No 1
	int in_units = flat_len;
	int out_units = 512;
	tensorflow::Scope scope_dense1 = t_root.NewSubScope("Dense1_layer");
	auto relu5 = AddDenseLayer("5", scope_dense1, in_units, out_units, true, after_drop);

	// Dense No 2
	in_units = out_units;
	out_units = 256;
	tensorflow::Scope scope_dense2 = t_root.NewSubScope("Dense2_layer");
	auto relu6 = AddDenseLayer("6", scope_dense2, in_units, out_units, true, relu5);

	// Dense No 3
	in_units = out_units;
	out_units = 1;
	tensorflow::Scope scope_dense3 = t_root.NewSubScope("Dense3_layer");
	auto logits = AddDenseLayer("7", scope_dense3, in_units, out_units, false, relu6);
	out_classification = tensorflow::ops::Sigmoid(t_root.WithOpName("Output_Classes"), logits);
	return t_root.status();
}

tensorflow::Status deepcpp::CNN::CreateOptimizationGraph(float learning_rate) {
	input_labels_var = tensorflow::ops::Placeholder(t_root.WithOpName("inputL"), tensorflow::DT_FLOAT);
	tensorflow::Scope scope_loss = t_root.NewSubScope("Loss_scope");
	out_loss_var = tensorflow::ops::Mean(scope_loss.WithOpName("Loss"), tensorflow::ops::SquaredDifference(scope_loss, out_classification, input_labels_var), {0});
	TF_CHECK_OK(scope_loss.status());
	std::vector<tensorflow::Output> weights_biases;
	for(std::pair<tensorflow::string, tensorflow::Output> i: m_vars)
		weights_biases.push_back(i.second);
	std::vector<tensorflow::Output> grad_outputs;
	TF_CHECK_OK(AddSymbolicGradients(t_root, {out_loss_var}, weights_biases, &grad_outputs));
	int index = 0;
	for(std::pair<tensorflow::string, tensorflow::Output> i: m_vars) {
		//Applying Adam
		std::string s_index = std::to_string(index);
		auto m_var = tensorflow::ops::Variable(t_root, m_shapes[i.first], tensorflow::DT_FLOAT);
		auto v_var = tensorflow::ops::Variable(t_root, m_shapes[i.first], tensorflow::DT_FLOAT);
		m_assigns["m_assign"+s_index] = tensorflow::ops::Assign(t_root, m_var, tensorflow::Input::Initializer(0.f, m_shapes[i.first]));
		m_assigns["v_assign"+s_index] = tensorflow::ops::Assign(t_root, v_var, tensorflow::Input::Initializer(0.f, m_shapes[i.first]));

		auto adam = tensorflow::ops::ApplyAdam(t_root, i.second, m_var, v_var, 0.f, 0.f, learning_rate, 0.9f, 0.999f, 0.00000001f, {grad_outputs[index]});
		v_out_grads.push_back(adam.operation);
		index++;
	}
	return t_root.status();
}

tensorflow::Status deepcpp::CNN::Initialize() {
	if(!t_root.ok())
		return t_root.status();
	std::vector<tensorflow::Output> ops_to_run;
	for(std::pair<tensorflow::string, tensorflow::Output> i: m_assigns)
		ops_to_run.push_back(i.second);
	t_session = std::unique_ptr<tensorflow::ClientSession>(new tensorflow::ClientSession(t_root));
	TF_CHECK_OK(t_session->Run(ops_to_run, nullptr));

	/*
	GraphDef graph;
	TF_RETURN_IF_ERROR(t_root.ToGraphDef(&graph));
	SummaryWriterInterface* w;
	TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs", ".cnn-graph", Env::Default(), &w));
	TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
	*/
	return tensorflow::Status::OK();
}

tensorflow::Status deepcpp::CNN::TrainCNN(tensorflow::Tensor& image_batch, tensorflow::Tensor& label_batch, std::vector<float>& results, float& loss) {
	if(!t_root.ok())
		return t_root.status();
	std::vector<tensorflow::Tensor> out_tensors;
	// Inputs: batch of images, labels, drop rate and do not skip drop.
	//Extract: Loss and result. Run also: Apply Adam commands
	TF_CHECK_OK(t_session->Run({{input_batch_var, image_batch}, {input_labels_var, label_batch}, {drop_rate_var, 0.5f}, {skip_drop_var, 0.f}}, {out_loss_var, out_classification}, v_out_grads, &out_tensors));
	loss = out_tensors[0].scalar<float>()(0);

	//both labels and results are shaped [20, 1]
	auto mat1 = label_batch.matrix<float>();
	auto mat2 = out_tensors[1].matrix<float>();
	for(int i = 0; i < mat1.dimension(0); i++)
		results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
	return tensorflow::Status::OK();
}

tensorflow::Status deepcpp::CNN::ValidateCNN(tensorflow::Tensor& image_batch, tensorflow::Tensor& label_batch, std::vector<float>& results) {
	if(!t_root.ok())
		return t_root.status();
	std::vector<tensorflow::Tensor> out_tensors;
	//Inputs: batch of images, drop rate 1 and skip drop.
	TF_CHECK_OK(t_session->Run({{input_batch_var, image_batch}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
	auto mat1 = label_batch.matrix<float>();
	auto mat2 = out_tensors[0].matrix<float>();
	for(int i = 0; i < mat1.dimension(0); i++)
		results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
	return tensorflow::Status::OK();
}

tensorflow::Status deepcpp::CNN::Predict(tensorflow::Tensor& image, int& result) {
	if(!t_root.ok())
		return t_root.status();

	std::vector<tensorflow::Tensor> out_tensors;
	// Inputs: image, drop rate 1 and skip drop.
	TF_CHECK_OK(t_session->Run({{input_batch_var, image}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
	auto mat = out_tensors[0].matrix<float>();
	result = (mat(0, 0) > 0.5f)? 1 : 0;
	return tensorflow::Status::OK();
}
