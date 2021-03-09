//Args structure used to parse args to inference class
#pragma once
#include <iostream>
#include <string>
#include "NvInfer.h"

// TODO (weich): Use logger to output error message
#define ASSERT(condition)                                                     \
    do {                                                                      \
        if(!condition)  {                                                     \
            std::cout << "Condition failed on" << #condition << std::endl;    \
            abort();                                                          \
        }                                                                     \
    } while(0)                                                                

#define CHECK_CUDA(status)                                                    \
    do {                                                                      \
        if (status) {                                                         \
            std::cout <<"Cuda failure: " << status << std::endl;              \
            abort();                                                          \
        }                                                                     \
    } while(0)                                                                   


namespace trtInference {

struct Param {
    Param(int32_t max_batch_size, bool int8, bool fp16, std::string onnx_file):
        max_batch_size(max_batch_size),
        int8(int8),
        fp16(fp16),
        onnx_file(onnx_file)
    {
    }
    int32_t max_batch_size{1};    // Number of images in a batch
    bool int8{false};             // Running inference in int8 mode.
    bool fp16{false};             // Running inference in fp16 mode.
    std::string onnx_file;        // Path to onnx model. 
}; 


// Ensures that every tensor used by a network has a scale.
//
// All tensors in a network must have a range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales for the entire network.
//
// If a tensor does not have a scale, it is assigned inScales or outScales as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its scale is assigned inScales.
// * Otherwise its scale is assigned outScales.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where scaling factors are asymmetric.
inline void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                ASSERT(input->setDynamicRange(-inScales, inScales));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    ASSERT(output->setDynamicRange(-inScales, inScales));
                }
                else
                {
                    ASSERT(output->setDynamicRange(-outScales, outScales));
                }
            }
        }
    }
}

inline  size_t getElementSize(nvinfer1::DataType type) {
    switch (type)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

}
