#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace leaxer {

/**
 * ONNX Runtime session wrapper for inference
 * 
 * Thread-safety: Ort::Session::Run is thread-safe, but this wrapper
 * is designed for single-threaded use. Create separate OnnxSession
 * instances for parallel inference.
 */
class OnnxSession {
public:
    /**
     * Load an ONNX model from file
     * @param model_path Path to the .onnx model file
     * @param num_threads Number of intra-op threads (0 = let ORT decide)
     */
    explicit OnnxSession(const std::string& model_path, int num_threads = 0);
    
    ~OnnxSession();
    
    // Non-copyable, movable
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;
    
    /**
     * Run inference with given inputs
     * @param input_names Names of input tensors (must match model)
     * @param inputs Input tensor values
     * @return Output tensor values
     */
    std::vector<Ort::Value> run(
        const std::vector<std::string>& input_names,
        std::vector<Ort::Value>& inputs
    );
    
    /**
     * Run inference using model's default input/output names
     * @param inputs Input tensor values (order must match getInputNames())
     * @return Output tensor values
     */
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs);
    
    // Model introspection
    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    
    /**
     * Get input tensor shape (dynamic dims shown as -1)
     * @param index Input index
     * @return Shape vector
     */
    std::vector<int64_t> getInputShape(size_t index) const;
    
    /**
     * Get output tensor shape (dynamic dims shown as -1)
     * @param index Output index
     * @return Shape vector
     */
    std::vector<int64_t> getOutputShape(size_t index) const;
    
    /**
     * Get the model path
     */
    const std::string& getModelPath() const { return model_path_; }
    
    /**
     * Print model info for debugging
     */
    void printModelInfo() const;

private:
    std::string model_path_;
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Cached names for convenience
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    void cacheNames();
};

} // namespace leaxer
