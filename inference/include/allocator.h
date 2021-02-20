//Implementation of Memoery allocator.
#pragma once

#include <cuda_runtime_api.h>
#include <memory>

namespace trtInference {

class Allocator {

    virtual bool operator()(void** ptr, const size_t size) const = 0;

};

class Free {

    virtual void operator()(void* ptr) const = 0;

};


class GPUAllocator: public Allocator {

    bool operator()(void** ptr, size_t size) const {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class GPUFree: public Free {

    void operator()(void* ptr) const {
        cudaFree(ptr);
    }
};

class CPUAllocator: public Allocator {

    bool operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class CPUFree: public Free {

    void operator()(void* ptr) const {
        free(ptr);
    }
};


} // namespace trtInference 
