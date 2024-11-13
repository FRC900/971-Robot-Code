#include "frc971/orin/engine.h"
#include <ctype.h>                       // for isspace
#include <opencv2/core/hal/interface.h>  // for CV_32FC1
#include <algorithm>                     // for remove_if, copy, shuffle
#include <filesystem>                    // for directory_iterator, begin, end
#include <fstream>                       // for operator<<, endl, basic_ostream
#include <iostream>                      // for cout, cerr
#include <iterator>                      // for istream_iterator, back_inser...
#include <opencv2/core.hpp>              // for merge
#include <opencv2/core/mat.inl.hpp>      // for _InputArray::_InputArray
#include <opencv2/imgcodecs.hpp>         // for imread
#include <random>                        // for random_device, default_rando...
#include <stdexcept>                     // for runtime_error, logic_error
#include <utility>                       // for move, pair
#include "NvInferLegacyDims.h"           // for Dims4
#include "NvOnnxParser.h"                // for createParser, IParser
#include "cuda_runtime_api.h"            // for cudaStreamSynchronize, cudaG...
#include "frc971/orin/cuda_utils.h"      // for cudaSafeCall
#include "frc971/orin/gpu_image.h"       // for GpuImage

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string& dirPath) {
    std::vector<std::string> filepaths;
    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path());
    }
    return filepaths;
}

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

template<class CALIBRATOR>
Engine<CALIBRATOR>::Engine(const Options &options)
    : m_options(options)
{
    cudaSafeCall(cudaStreamCreate(&m_inferenceCudaStream));
    m_timings = std::make_unique<Timings>();
    m_timings->setEnabled(false);
}

template<class CALIBRATOR>
bool Engine<CALIBRATOR>::build(const std::string &modelPath, const std::string &onnxModelFilename)
{
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options, modelPath, onnxModelFilename);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    const std::string onnxModelPath = modelPath + "/" + onnxModelFilename;
    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    if (!parser->parse(buffer.data(), buffer.size())) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    if (input0Batch == -1) {
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else if (input0Batch == 1) {
        std::cout << "Model only supports fixed batch size of 1" << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of 1. Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    } else {
        throw std::runtime_error("Implementation currently only supports dynamic batch sizes or a fixed batch size of 1 (your batch size is fixed to "
        + std::to_string(input0Batch) + ")");
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
    } else if (m_options.precision == Precision::INT8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        config->setFlag(BuilderFlag::kINT8);

        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        const auto calibrationFileName = modelPath + "/" + onnxModelFilename.substr(0, onnxModelFilename.find_last_of('.')) + ".calib";

        m_calibrator = std::make_unique<CALIBRATOR>(m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2], m_inferenceCudaStream,
                                                    m_options.calibrationDataDirectoryPath, calibrationFileName,
                                                    inputName);
        config->setInt8Calibrator(m_calibrator.get());
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    cudaSafeCall(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    cudaSafeCall(cudaStreamDestroy(profileStream));
    return true;
}

template<class CALIBRATOR>
Engine<CALIBRATOR>::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        cudaSafeCall(cudaFree(buffer));
    }

    m_buffers.clear();
    cudaSafeCall(cudaStreamDestroy(m_inferenceCudaStream));
}

template<class CALIBRATOR>
bool Engine<CALIBRATOR>::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    if (cudaSetDevice(m_options.deviceIndex) != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        std::cout << "tensorName = " << tensorName << " tensorShape = " << tensorShape.d[0] << ", " << tensorShape.d[1] << ", " << tensorShape.d[2] << ", " << tensorShape.d[3] << std::endl;
        if (tensorType == TensorIOMode::kINPUT) {
            m_inputDims.push_back(tensorShape);
            m_inputTensorNames.emplace_back(tensorName);
        } else if (tensorType == TensorIOMode::kOUTPUT) {
            m_outputDims.push_back(tensorShape);
            m_outputTensorNames.emplace_back(tensorName);
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }
    return true;
}

// Used to fix the size of dynamic sized inputs and outputs.
template<class CALIBRATOR>
bool Engine<CALIBRATOR>::allocateInputOutputTensors(bool &buffersResized)
{
    // Check to see if this set of inputs is a different
    // size than the previous ones. If not, don't bother
    // reallocating, just reuse the previously allocated buffers
    bool dimsChanged{false};
    if (m_engine->getNbIOTensors() != static_cast<int>(m_prevTensorDims.size()))
    {
        dimsChanged = true;
    }
    for (int i = 0; !dimsChanged && (i < m_engine->getNbIOTensors()); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        const auto curr_dims = m_context->getTensorShape(tensorName);
        if (m_prevTensorDims[i].nbDims != curr_dims.nbDims)
        {
            dimsChanged = true;
        }
        // Skip checking dim[0], since that is batch size.
        // Buffers are sized for the max possible batch size, so running
        // different batch sizes doesn't require reallocating buffer space
        for (int j = 1; !dimsChanged && (j < m_prevTensorDims[i].nbDims); j++)
        {
            if (m_prevTensorDims[i].d[j] != curr_dims.d[j])
            {
                dimsChanged = true;
            }
        }
    }
    if (!dimsChanged)
    {
        buffersResized = false;
        return true;
    }
    buffersResized = true;
    std::cout << " Regenerating input / output tensors" << std::endl;
    m_needNewCudaGraph = true;
    m_prevTensorDims.clear();
    
    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    cudaSafeCall(cudaStreamSynchronize(m_inferenceCudaStream));
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        // Use context here rather than engine to get the size of 
        // dynamic sized output tensors now that we've supplied
        // the actual input size.
        const auto tensorShape = m_context->getTensorShape(tensorName);
        const auto engineTensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        std::cout << "\ttensorName = " << tensorName <<
         " tensorShape = " << tensorShape.d[0] << ", " << tensorShape.d[1] << ", " << tensorShape.d[2] << ", " << tensorShape.d[3] <<
         " engineTensorShape = " << engineTensorShape.d[0] << ", " << engineTensorShape.d[1] << ", " << engineTensorShape.d[2] << ", " << engineTensorShape.d[3] <<
          std::endl;
        m_prevTensorDims.push_back(tensorShape);
        cudaSafeCall(cudaFreeAsync(m_buffers[i], m_inferenceCudaStream));
        if (tensorType == TensorIOMode::kINPUT) {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            cudaSafeCall(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), m_inferenceCudaStream));
        } else if (tensorType == TensorIOMode::kOUTPUT) {
            // The binding is an output
            uint32_t outputLenFloat = 1;

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            cudaSafeCall(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), m_inferenceCudaStream));
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
        m_tensorNameToBufferMap[tensorName] = i;
        if (!m_context->setTensorAddress(tensorName, m_buffers[i]))
        {
            std::cout << "setTensorAddress failed" << std::endl;
            return false;
        }
    }

    return true;
}

template <class CALIBRATOR>
bool Engine<CALIBRATOR>::runInference(const std::vector<std::vector<GpuImage<uint8_t>>> &inputs,
                                      bool &buffersResized,
                                      const int32_t forceBatchSize)
{
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        std::cout << "inputs.size() = " << inputs.size() << " numInputs = " << numInputs << std::endl;
        return false;
    }

    const auto batchSize = (forceBatchSize <= 0) ? static_cast<int32_t>(inputs[0].size()) : forceBatchSize;

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
            return false;
        }
    }

    m_timings->start("Set_dynamic_tensors", m_inferenceCudaStream);
    // Use the size of the requested inputs to set any dynamic tensor dimensions
    m_contextInputDims.clear();
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto dims = inputDimsFromInputImage(batchInput[0], m_inputDims[i]);

        const nvinfer1::Dims4 inputDims = {batchSize, dims.d[1], dims.d[2], dims.d[3]};
        // printf("setInputShape, dims = %d %d %d %d\n", inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]);
        if (!m_context->setInputShape(m_inputTensorNames[i].c_str(), inputDims)) // Define the batch size
        {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Context setInputShape failed" << std::endl;
            return false;
        }
        m_contextInputDims.push_back(inputDims);
    }
    m_timings->end("Set_dynamic_tensors");
    // Update output shapes from newly supplied inputs
    m_timings->start("Infer_shapes", m_inferenceCudaStream);
    if (const int rc = m_context->inferShapes(0, nullptr))
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Context inferShapes failed, rc = " << rc << std::endl;
        return false;
    }
    m_timings->end("Infer_shapes");

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    m_timings->start("Allocate_input_tensors", m_inferenceCudaStream);
    // Allocate (or reallocate, if sizes have changed) buffers for all input and output tensors
    if (!allocateInputOutputTensors(buffersResized)) {
        throw std::runtime_error("Error, allocateInputOutputTensors");
    }
    m_timings->end("Allocate_input_tensors");

    m_timings->start("Blob_from_gpu_image", m_inferenceCudaStream);
    // Preprocess inputs directly into buffers allocated for them
    for (size_t i = 0; i < numInputs; ++i) {
        // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format. 
        // The following method converts NHWC to NCHW.
        // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
        // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing, put the result in m_buffer[i]
        blobFromGpuImageWrappers(inputs[i], i);
    }
    m_timings->end("Blob_from_gpu_image");
    if (batchSize != m_lastBatchSize)
    {
        m_needNewCudaGraph = true;
        m_lastBatchSize = batchSize;
    }

    m_timings->start("new_cuda_graph", m_inferenceCudaStream);
    if (m_useCudaGraph && m_needNewCudaGraph)
    {
        std::cout << "Model bindings changed, generating CUDA graph" << std::endl;
        // Call enqueueV3() once after an input shape change to update internal state.
        m_context->enqueueV3(m_inferenceCudaStream);
        cudaSafeCall(cudaStreamSynchronize(m_inferenceCudaStream));
        cudaSafeCall(cudaStreamBeginCapture(m_inferenceCudaStream, cudaStreamCaptureModeGlobal));
        m_context->enqueueV3(m_inferenceCudaStream);
        cudaSafeCall(cudaStreamEndCapture(m_inferenceCudaStream, &m_cudaGraph));
        cudaSafeCall(cudaGraphInstantiate(&m_cudaGraphInstance, m_cudaGraph, NULL, NULL, 0));
        cudaSafeCall(cudaStreamSynchronize(m_inferenceCudaStream));
        m_needNewCudaGraph = false;
    }
    m_timings->end("new_cuda_graph");

    m_timings->start("inference", m_inferenceCudaStream);
    if (m_useCudaGraph)
    {
        cudaSafeCall(cudaGraphLaunch(m_cudaGraphInstance, m_inferenceCudaStream));
    }
    else
    {
        bool status = m_context->enqueueV3(m_inferenceCudaStream);
        if (!status)
        {
            return false;
        }
    }
    m_timings->end("inference");

    return true;
}

template <class CALIBRATOR>
std::string Engine<CALIBRATOR>::serializeEngineOptions(const Options &options, const std::string &modelPath, const std::string &onnxModelFilename)
{
    std::string engineName = modelPath + "/" + onnxModelFilename.substr(0, onnxModelFilename.find_last_of('.')) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName+= "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::FP32){
        engineName += ".fp32";
    } else {
        engineName += ".int8";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);

    return engineName;
}

template<class CALIBRATOR>
void Engine<CALIBRATOR>::getDeviceNames(std::vector<std::string>& deviceNames) const {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.emplace_back(prop.name);
    }
}

template<class CALIBRATOR>
void Engine<CALIBRATOR>::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

template<class CALIBRATOR>
void Engine<CALIBRATOR>::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

template<class CALIBRATOR>
const float *Engine<CALIBRATOR>::getBufferByName(const std::string &name, const size_t index) const{
    const auto it = m_tensorNameToBufferMap.find(name);
    if (it == m_tensorNameToBufferMap.cend())
    {
        std::cerr << "Could not read tensor buffer name " << name << std::endl;
        return nullptr;
    }
    size_t offset = 0;
    if (index)
    {
        const auto tensorShape = m_context->getTensorShape(name.c_str());
        offset = index;
        // Index 0 should be batch size - skip that, since the 
        // point here is to index through each output in the batch
        for (int i = 1; i < tensorShape.nbDims; i++)
        {
            offset *= tensorShape.d[i];
        }
    }
    return reinterpret_cast<const float *>(m_buffers[it->second]) + offset;
}

template<class CALIBRATOR>
const nvinfer1::Dims Engine<CALIBRATOR>::getContextDim(const std::string &name) const
{
    return m_context->getTensorShape(name.c_str());
}

template<class CALIBRATOR>
void Engine<CALIBRATOR>::setTimingEnable(const bool enable)
{
    m_timings->setEnabled(enable);
}

template<class CALIBRATOR>
void Engine<CALIBRATOR>::setUseCudaGraph(const bool useCudaGraph)
{
    m_useCudaGraph = useCudaGraph;
}

template<class CALIBRATOR>
cv::Mat Engine<CALIBRATOR>::getDebugImage(const size_t imageIdx)
{
    const auto inputDims = getContextInputDims()[0];
    std::cout << "Input dims = " << inputDims.d[0] << " " << inputDims.d[1] << " " << inputDims.d[2] << " " << inputDims.d[3] << std::endl;
    const int outputHeight = inputDims.d[2];
    const int outputWidth = inputDims.d[3];
    const float *destBuffer = getBufferByName("input");
    const size_t channelStride = outputWidth * outputHeight;
    cudaSafeCall(cudaStreamSynchronize(getCudaStream()));
    cv::Mat fin_img(outputHeight, outputWidth, CV_32FC1);
    cudaSafeCall(cudaMemcpyAsync(fin_img.data, destBuffer + imageIdx * channelStride, channelStride * sizeof(float), cudaMemcpyDeviceToHost, getCudaStream()));
    return fin_img;
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               cudaStream_t cudaStream,
                                               const std::string &calibDataDirPath,
                                               const std::string &calibTableName,
                                               const std::string &inputBlobName,
                                               bool readCache)
        : m_inputW(inputW)
        , m_inputH(inputH)
        , m_cudaStream(cudaStream)
        , m_batchSize(batchSize)
        , m_calibTableName(calibTableName)
        , m_inputBlobName(inputBlobName)
        , m_readCache(readCache)
{
    // Allocate GPU memory to hold the entire batch
    size_t inputCount = static_cast<size_t>(3) * inputW * inputH * batchSize;
    cudaSafeCall(cudaMalloc(&m_deviceInput, inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath)) {
        throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int32_t nbBindings) noexcept {
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

#if 0 // TODO - fix me
    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<GpuImageWrapper> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
        std::cout << "Reading image " << i << ": " << m_imgPaths[i] << std::endl;
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty()){
            std::cout << "Fatal error: Unable to read image at path: " << m_imgPaths[i] << std::endl;
            return false;
        }

        GpuImageWrapper gpuImg;
        gpuImg.upload(cpuImg, m_cudaStream);

        inputImgs.emplace_back(std::move(gpuImg));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    blobFromGpuImageWrappers(inputImgs);

    m_imgIdx+= m_batchSize;
    if (std::string(names[0]) != m_inputBlobName) {
        std::cout << "Error: Incorrect input name provided!" << std::endl;
        return false;
    }
    bindings[0] = m_deviceInput;
    cudaSafeCall(cudaStreamSynchronize(m_cudaStream));
#endif
    return true;
}

void const *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept {
    std::cout << "Searching for calibration cache: " << m_calibTableName << std::endl;
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good()) {
        std::cout << "Reading calibration cache: " << m_calibTableName << std::endl;
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length) noexcept {
    std::cout << "Writing calib cache: " << m_calibTableName << " Size: " << length << " bytes" << std::endl;
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    cudaSafeCall(cudaFree(m_deviceInput));
};


// #include "deeptag_ros/detection_engine.h"
// template class Engine<DetectionEngineCalibrator>;
#include "frc971/orin/decoder_engine.h"
template class Engine<DecoderEngineCalibrator>;