#include "include/inference.h"


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

    assert(network->getNbInputs() == 1);
    inputDims = network->getInput(0)->getDimensions();
    // NCHW
    assert(inputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    outputDims = network->getOutput(0)->getDimensions();
    // NC
    assert(outputDims.nbDims == 2);


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


bool ImageInference::inference() {

    // Code to random generate images
    

}
 
}// end trtInference namespace
