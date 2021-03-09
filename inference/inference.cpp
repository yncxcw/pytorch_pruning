#include "include/inference.h"

#include "cuda_runtime_api.h"

namespace trtInference {
//! \breif build the network engine
bool ImageInference::build() {

    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtInference::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    // Building INetwork objects in full dimensions mode with dynamic shape support 
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser
        = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtInference::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto constructed = constructNetWork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }
    
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), trtInference::TRTDestroyer<nvinfer1::ICudaEngine>());

    if(!engine) {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    inputDims = network->getInput(0)->getDimensions();
    // NCHW
    ASSERT(inputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    outputDims = network->getOutput(0)->getDimensions();
    // NC
    ASSERT(outputDims.nbDims == 2);
    classes = outputDims.d[1];
    is_built = true;
}


//! \brief Parse on ONNX model and create a TRT network
bool ImageInference::constructNetWork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
            TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
            TRTUniquePtr<nvonnxparser::IParser>& parser){

    // Parse onnx file
    // TODO add logging severity
    auto parsed = parser->parseFromFile(param.onnx_file.c_str(), static_cast<int>(trtInference::gLogger.getReportableSeverity()));

    if (!parsed) {
        return false;
    }
  
    builder->setMaxBatchSize(param.max_batch_size); 
    // 1 << 20 = 16 MB 
    config->setMaxWorkspaceSize(1<<20);
    
    if (param.fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    if (param.int8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        trtInference::setAllTensorScales(network.get(), 127, 127);
    }

    return true;   
}

//! \brief Running inference on input data and copy results back to output
template<typename DType>
bool ImageInference::inference(size_t batch_size, DType* input, DType* output) {
    if(!is_built) {
        return false;
    }
    
    auto context = TRTUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        return false;
    }

    //The caller needs to make sure that strlen(input) == batch_size * C * H * W
    size_t input_size  = batch_size * model_input_size(); 
    // Output would be batch_size * C where C = num_classes
    size_t output_size = batch_size * model_output_size();

    //TODO (weich) Implement slab allocator
    void* input_buffer; 
    void* output_buffer;
    CHECK_CUDA(cudaMalloc(&input_buffer, input_size*sizeof(DType))); 
    CHECK_CUDA(cudaMalloc(&output_buffer, output_size*sizeof(DType)));
    
    //Create CUDA steam
    //TODO (weich) Wrap CUDAsteam in class.
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    //DMA copy input data to device and run inference on the input, copy the output batch to the host
    CHECK_CUDA(cudaMemcpyAsync(input_buffer, input, input_size, cudaMemcpyHostToDevice, stream));
    void* buffers[2] = {input_buffer, output_buffer};
    //TODO (weich) Add cuda event profiling
    auto status = context->enqueue(batch_size, buffers, stream, /*cudaEvent_t*/ nullptr);
    if (!status) {
        return false;
    }
    CHECK_CUDA(cudaMemcpyAsync(output, output_buffer, output_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //Release steam and buffers
    cudaStreamDestroy(stream);
    CHECK_CUDA(cudaFree(input_buffer));
    CHECK_CUDA(cudaFree(output_buffer));

    return true;
}

template bool ImageInference::inference<float>(size_t batch_size, float* input, float* output);
 
template bool ImageInference::inference<double>(size_t batch_size, double* input, double* output);

template bool ImageInference::inference<int>(size_t batch_size, int* input, int* output);
}// end trtInference namespace
