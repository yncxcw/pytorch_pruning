cmake_minimum_required(VERSION 3.13)

project(TRT_INFERENCE LANGUAGES CXX CUDA)



# Make this configurable
# Include cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# Include tensorrt
include_directories(/usr/include/x86_64-linux-gnu) 
link_directories(/usr/lib/x86_64-linux-gnu)

include_directories(include)

aux_source_directory(. SOURCES)

set(MAIN_DEP_LIBS
    # TRT runtime 
    nvinfer
    # Onnx parser
    nvonnxparser
    # cuda runtime
    cudart
)

add_executable(main ${SOURCES})

target_link_libraries(main
    ${MAIN_DEP_LIBS}
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)
