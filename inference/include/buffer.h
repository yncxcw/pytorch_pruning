// Implementation of a GPU & CPU buffer
#pragma once

#include "allocator.h"
#include "common.h"

#include "NvInfer.h"


namespace trtInference {

template<typename AllocFunc, typename FreeFunc>
class Buffer {

    public:
        Buffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            :_size(0)
            ,_capacity(0)
            ,_type(type)
            ,_buffer(nullptr){
        
        }

        Buffer(size_t size, nvinfer1::DataType type) {
            _size = size;
            _capacity = size;
            _type = type;
            auto nbytes = this->nbytes();
            if (!allocFun(&_buffer, nbytes)) {
                throw std::bad_alloc();
            }
        }

        Buffer(Buffer&& buffer)
            :_size(buffer._size)
            ,_capacity(buffer._capacity)
            ,_type(buffer._type)
            ,_buffer(buffer._buffer)
        {
            buffer._size = 0;
            buffer._capacity = 0;
            buffer._type = nvinfer1::DataType::kFLOAT;
            buffer._buffer = nullptr;
        }

        // TODO Allow this when copyFunc is added.
        Buffer(const Buffer& buffer) = delete;

        // Copy assignment
        Buffer& operator=(Buffer& buffer) = delete;

        // Move assignment
        Buffer& operator=(Buffer&& buffer) {
            if (this == &buffer) {
                return *this;
            }

            freeFun(_buffer);
            _size = buffer._size;
            _capacity = buffer._capacity;
            _type = buffer._type;
            _buffer = buffer._buffer;
        }

        void* data() {
            return _buffer;
        }

        const void* data() const {
            return _buffer;
        }

        size_t size() const {
            return _size;
        }

        size_t nbytes() const {
            return _size * getElementSize(_type);
        }

        void resize(size_t size) {
            _size = size;
            if (size > _capacity) {
                freeFun(_buffer);
                if (!allocFn(&_buffer, this->nbytes())) {
                    throw std::bad_alloc();
                }
                _capacity = size;
            }

        }

        ~Buffer() {
            freeFun(_buffer);
        }

    private:
        size_t _size{0}, _capacity{0};
        // Could this be a more generic c++ dtype.
        nvinfer1::DataType _type;
        // Pointer to the buffer.
        void* _buffer;
        AllocFunc allocFun;
        FreeFunc freeFun;
}; // class Buffer

using GPUBuffer = Buffer<GPUAllocator, GPUFree>;
using CPUBuffer = Buffer<CPUAllocator, CPUFree>;


} // namespace trtInference
