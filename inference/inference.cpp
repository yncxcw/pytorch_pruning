#include "include/inference.h"

#include "cuda_runtime_api.h"

#include <iostream>


namespace trtInference {
//! \breif build the network engine
bool ImageInference::build() {

    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtInference::gLogger.getTRTLogger()));
    if (!builder) {
        std::cout << "createInferBuilder failed.";
        return false;
    }

    // Building INetwork objects in full dimensions mode with dynamic shape support 
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) { 
        std::cout << "createNetworkV2 failed.";     
        return false;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) { 
        std::cout << "createBuilder failed.";
        return false;
    }

    auto parser
        = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtInference::gLogger.getTRTLogger()));
    if (!parser) { 
        std::cout << "createParser failed.";
        return false;
    }

    auto constructed = constructNetWork(builder, network, config, parser);
    if (!constructed) { 
        std::cout << "constructNetWork failed.";
        return false;
    }
    
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), trtInference::TRTDestroyer<nvinfer1::ICudaEngine>());

    if(!engine) { 
        std::cout << "buildEngineWithConfig failed.";
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
    return true;
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
    // 1 << 30 = 1G 
    config->setMaxWorkspaceSize(1<<30);
    
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
float ImageInference::inference(size_t batch_size, DType* input, DType* output) {
    if(!is_built) {
        std::cout << "Inference engine is not built" <<std::endl;
        return -1;
    }
    
    auto context = TRTUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cout << "createExecutionContext failed" << std::endl;
        return -1;
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

    //TODO (weich) Wrap cuda even in class
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start, stream));
    auto status = context->enqueue(batch_size, buffers, stream, /*cudaEvent_t*/ nullptr);
    CHECK_CUDA(cudaEventRecord(end, stream)); 
    CHECK_CUDA(cudaEventSynchronize(end));
    float infertime;
    CHECK_CUDA(cudaEventElapsedTime(&infertime, start, end));    

    if (!status) {
        std::cout << "context->enqueue fauld." << std::endl;
        return -1;
    }
    CHECK_CUDA(cudaMemcpyAsync(output, output_buffer, output_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    //Release steam and buffers
    cudaStreamDestroy(stream);
    CHECK_CUDA(cudaFree(input_buffer));
    CHECK_CUDA(cudaFree(output_buffer));

    return infertime;
}

template float ImageInference::inference<float>(size_t batch_size, float* input, float* output);
 
template float ImageInference::inference<double>(size_t batch_size, double* input, double* output);

template float ImageInference::inference<int>(size_t batch_size, int* input, int* output);
}// end trtInference namespace
