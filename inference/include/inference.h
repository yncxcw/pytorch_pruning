// Implementation of a generic image classification class 
#pragma once

#include "logging.h"
#include "common.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <memory>

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
    template<typename DType>
    bool inference(size_t batch_size, DType* input, DType* output);

    //! \Return input size as H*W*C
    size_t model_input_size(){
        ASSERT(!is_built);
        auto ndims = inputDims.nbDims;
        size_t size = 1;
        for (int i=0; i<ndims; i++) {
            size = size * inputDims.d[i];
        }
        return size;
    }

    //! \Return output size as C
    size_t model_output_size() {
        ASSERT(!is_built);
        auto ndims = outputDims.nbDims;
        size_t size = 1;
        for (int i=0; i<ndims; i++) {
            size = size * outputDims.d[i];
        }
        return size;

    }

private:
    trtInference::Param param;

    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;

    // Number of classes to classify
    size_t classes{0};

    // Is the TRT engine buit successfuly
    bool is_built{false};

    // TensorRT engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    //! \brief Parse on ONNX model and create a TRT network
    bool constructNetWork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser);

};

} // end trtInference namespace
