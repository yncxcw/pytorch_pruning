//Running the trt inference on an onnx model.

#include "include/common.h"
#include "include/inference.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <memory>

//! \Funtion to verify if trt works as expected
void image_inference_test(std::unique_ptr<trtInference::ImageInference>& image_inference, size_t batch_size) {
    // Test Code to trt engine
    auto input_size = image_inference->model_input_size();
    auto output_size = image_inference->model_output_size();

    auto input_buffer = std::make_unique<float>(batch_size*input_size);
    auto output_buffer = std::make_unique<float>(batch_size*output_size);
    image_inference->inference<float>(batch_size, input_buffer.get(), output_buffer.get());
    std::cout << "Done with one inference!" << std::endl;
}


int main() {

    trtInference::Param param(32, false, false, "");
    auto image_inference = std::make_unique<trtInference::ImageInference>(param);
    if(!image_inference->build()) {
        std::cout << "Image inference building is not successfull" << std::endl;
    }

    image_inference_test(image_inference, 10);
    return 0;
}
