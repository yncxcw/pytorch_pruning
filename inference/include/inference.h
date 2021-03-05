// Implementation of a generic image classification class 
#pragma once

#include "logging.h"
#include "buffer.h"
#include "common.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace trtInference {

template<typename T>
struct TRTDestroyer {

    //TRT instances needs to be called with destroy explicitly prior to exit.
    void operator()(T *t) {
        if(t)
            t->destroy();
    }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroyer<T>>;


//! \brief TRT inference engine for image classification.
class ImageInference {

public:
    ImageInference(trtInference::Param& param)
        :param(param)
        ,engine(nullptr)
    {
    }

    //! \breif build the network engine
    bool build();

    //! \Run inference
    bool inference();

private:
    trtInference::Param param;

    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;

    //Number of classes to classify
    size_t classes{0};

    // TensorRT engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    //! \brief Parse on ONNX model and create a TRT network
    bool constructNetWork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser);

};

} // end trtInference namespace
