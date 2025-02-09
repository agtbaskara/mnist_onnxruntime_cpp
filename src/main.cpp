#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

Ort::Session session = Ort::Session(nullptr);
Ort::SessionOptions sessionOptions;
OrtCUDAProviderOptions cuda_options;
Ort::MemoryInfo memory_info{ nullptr };					// Used to allocate memory for input
std::vector<const char*>* output_node_names = nullptr;	// output node names
std::vector<const char*>* input_node_names = nullptr;	// Input node names
std::vector<int64_t> input_node_dims;					// Input node dimension
cv::Mat blob;											// Converted input. In this case for the (1,3,640,640)

struct OnnxENV {
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
};

// Settings
void SetSessionOptions(bool UseCuda);
void SetUseCuda();

// Create session
bool LoadWeights(OnnxENV* Env, const wchar_t* ModelPath);
void SetInputNodeNames(std::vector<const char*>* input_node_names);
void SetInputDemensions(std::vector<int64_t> input_node_dims);
void SetOutputNodeNames(std::vector<const char*>* input_node_names);

void SetSessionOptions(bool useCUDA) {
	sessionOptions.SetInterOpNumThreads(1);
	sessionOptions.SetIntraOpNumThreads(1);
	// Optimization will take time and memory during startup
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	// CUDA options. If used.
	if (useCUDA)
	{
		SetUseCuda();
	}
}

void SetUseCuda() {
	cuda_options.device_id = 0;  //GPU_ID
	cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
	cuda_options.arena_extend_strategy = 0;
	// May cause data race in some condition
	cuda_options.do_copy_in_default_stream = 0;
	sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
}

bool LoadWeights(OnnxENV* Env, const char* ModelPath) {
	try {
		// Model path is const wchar_t*
		session = Ort::Session(Env->env, ModelPath, sessionOptions);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	try {	// For allocating memory for input tensors
		memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

void SetInputNodeNames(std::vector<const char*>* names) {
	input_node_names = names;
}

void SetOutputNodeNames(std::vector<const char*>* names) {
	output_node_names = names;
}

void SetInputDemensions(std::vector<int64_t> Dims) {
	input_node_dims = Dims;
}

bool PreProcess(cv::Mat frame, std::vector<Ort::Value>& inputTensor) {
	// this will make the input into 1, 1, 28, 28
	// this is leftover from yolo preprocess

	blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(28, 28), (0, 0, 0), false, false);
	size_t input_tensor_size = blob.total();
	try {
		inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

int Inference(cv::Mat frame, std::vector<Ort::Value>& OutputTensor) {
	std::vector<Ort::Value>InputTensor;
	bool error = PreProcess(frame, InputTensor);
	if (!error) return NULL;
	try {
		OutputTensor = session.Run(Ort::RunOptions{ nullptr }, input_node_names->data(), InputTensor.data(), InputTensor.size(), output_node_names->data(), 1);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return -1;
	}
	return OutputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();	// Number of elements in output. Num_of-detected * 7
}

std::vector<float> softmax(std::vector<float> x) {
    float max_val = *std::max_element(x.begin(), x.end());
    float sum_exp = 0;

    for (float& v : x) {
        v = std::exp(v - max_val);
        sum_exp += v;
    }

    for (float& v : x) v /= sum_exp;

    return x;
}

int main() {
    // debug check onnx runtime providers
    // auto providers = Ort::GetAvailableProviders();
    // for (auto provider : providers) {
    //     std::cout << provider << std::endl;
    // }

    // set model parameters
	std::vector<const char*> input_node_names = {"Input3"}; // Input node names
	std::vector<int64_t> input_dims = {1, 1, 28, 28};
	std::vector<const char*> output_node_names = {"Plus214_Output_0"}; // Output node names
	std::vector<Ort::Value> OutputTensor; // Holds the result of inference
	OnnxENV Env;

    // set model settings
	SetSessionOptions(true); // set true to use CUDA
	LoadWeights(&Env, "models/mnist-12.onnx");	// Load weight and create session
	SetInputDemensions(input_dims);
	SetInputNodeNames(&input_node_names);
	SetOutputNodeNames(&output_node_names);
    
	// load image
    cv::Mat frame = cv::imread("test_image/1.png");

	// preprocess for mnist
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    cv::Mat frame_inverted;
    cv::bitwise_not(frame_gray, frame_inverted);

	// inference
    int numDetected = -1; //left over from yolo, should be same as output tensor size
    numDetected = Inference(frame_inverted, OutputTensor);
    
	// get output tensor
    float* Result = OutputTensor.front().GetTensorMutableData<float>();

    std::vector<float> raw_output;

	// put output as vector
    for (int i = 0; i < numDetected; i++) {
        raw_output.push_back(Result[i]);
    }
    OutputTensor.clear();

    std::vector<float> softmax_output = softmax(raw_output);

	// debug print softmax output
    // for (int i = 0; i < softmax_output.size(); i++) {
    //     std::cout << softmax_output[i] << std::endl;
    // }

    // Get index of max element
    int max_index = std::distance(softmax_output.begin(), std::max_element(softmax_output.begin(), softmax_output.end()));

    std::cout << "Prediction: " << max_index << std::endl;
	std::cout << "Confidence: " << softmax_output[max_index] << std::endl;

    return 0;
}
