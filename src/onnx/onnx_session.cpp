#include "onnx_session.h"
#include <iostream>
#include <stdexcept>

namespace leaxer {

OnnxSession::OnnxSession(const std::string& model_path, int num_threads)
    : model_path_(model_path)
    , env_(ORT_LOGGING_LEVEL_WARNING, "leaxer-qwen")
{
    // Configure session options
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (num_threads > 0) {
        options_.SetIntraOpNumThreads(num_threads);
    }
    
    // Enable memory pattern optimization
    options_.EnableMemPattern();
    options_.EnableCpuMemArena();
    
    // Load the model
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model '" + model_path + "': " + e.what());
    }
    
    // Cache input/output names
    cacheNames();
}

OnnxSession::~OnnxSession() = default;

OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

void OnnxSession::cacheNames() {
    // Cache input names
    size_t num_inputs = session_->GetInputCount();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.push_back(name.get());
    }
    
    // Cache output names
    size_t num_outputs = session_->GetOutputCount();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(name.get());
    }
}

std::vector<Ort::Value> OnnxSession::run(
    const std::vector<std::string>& input_names,
    std::vector<Ort::Value>& inputs
) {
    // Convert string names to const char*
    std::vector<const char*> input_names_cstr;
    input_names_cstr.reserve(input_names.size());
    for (const auto& name : input_names) {
        input_names_cstr.push_back(name.c_str());
    }
    
    std::vector<const char*> output_names_cstr;
    output_names_cstr.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    try {
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            inputs.data(),
            inputs.size(),
            output_names_cstr.data(),
            output_names_.size()
        );
        return outputs;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX inference failed: ") + e.what());
    }
}

std::vector<Ort::Value> OnnxSession::run(std::vector<Ort::Value>& inputs) {
    return run(input_names_, inputs);
}

std::vector<std::string> OnnxSession::getInputNames() const {
    return input_names_;
}

std::vector<std::string> OnnxSession::getOutputNames() const {
    return output_names_;
}

std::vector<int64_t> OnnxSession::getInputShape(size_t index) const {
    if (index >= input_names_.size()) {
        throw std::out_of_range("Input index out of range");
    }
    auto type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<int64_t> OnnxSession::getOutputShape(size_t index) const {
    if (index >= output_names_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    auto type_info = session_->GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

void OnnxSession::printModelInfo() const {
    std::cout << "ONNX Model: " << model_path_ << "\n";
    std::cout << "Inputs (" << input_names_.size() << "):\n";
    for (size_t i = 0; i < input_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << input_names_[i] << " : [";
        auto shape = getInputShape(i);
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            if (shape[j] < 0) {
                std::cout << "?";
            } else {
                std::cout << shape[j];
            }
        }
        std::cout << "]\n";
    }
    
    std::cout << "Outputs (" << output_names_.size() << "):\n";
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << output_names_[i] << " : [";
        auto shape = getOutputShape(i);
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            if (shape[j] < 0) {
                std::cout << "?";
            } else {
                std::cout << shape[j];
            }
        }
        std::cout << "]\n";
    }
}

} // namespace leaxer
