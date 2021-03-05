cmake_minimum_required(VERSION 3.13)

project(TRT_INFERENCE LANGUAGES CXX CUDA)

include_directories(include)

# Make this configurable
# Include cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# Include tensorrt
include_directories(/usr/include/x86_64-linux-gnu) 
link_directories(/usr/lib/x86_64-linux-gnu)

add_library(inference_lib inference.cpp)
set(MAIN_DEP_LIBS
    # TRT runtime 
    nvinfer
    # Onnx parser
    nvonnxparser
    inference_lib
)

add_executable(main main.cpp)

target_link_libraries(main
    ${MAIN_DEP_LIBS}
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)
