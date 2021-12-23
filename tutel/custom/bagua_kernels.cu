#define THRUST_IGNORE_CUB_VERSION_CHECK
#define CUDACHECK(cmd) do { cudaError_t e = cmd; if( e != cudaSuccess ) { printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)
#define ALIGN_SIZE(size, align) (((size) + (align) - 1) / (align) * (align))
#define DIVUP(x, y) (((x)+(y)-1)/(y))

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cub/cub.cuh>

const float eps = 1e-7;

// Reference: https://github.com/dmlc/cub/blob/master/cub/thread/thread_operators.cuh
struct Sum {
    /// Boolean sum operator, returns <tt>a + b</tt>
    template<typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct Max
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return ((b > a) ? b : a);
    }

};

template <>
__device__ __forceinline__ half Max::operator()<half>(const half &a, const half &b) const {
#if __CUDA_ARCH__ >= 530
	return __hgt(b, a) ? b: a;
#else
	return (__half2float(b) > __half2float(a) ? b : a);
#endif
}


struct Min
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return ((b < a) ? b : a);
    }

};

template <>
__device__ __forceinline__ half Min::operator()<half>(const half &a, const half &b) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(b, a) ? b: a;
#else
    return (__half2float(b) < __half2float(a) ? b : a);
#endif
}


template<typename T>
__device__ inline float __accum_to_float(float a, T b) {
    return a + b;
}

template<>
__device__ inline float __accum_to_float<half>(float a, half b) {
    return a + __half2float(b);
}

template<typename T, bool average>
__device__ inline T __from_float(float a, int n, T placeholder) {
   if (average) {
       return a / n;
   } else {
       return a;
   }
}

template<>
__device__ inline half __from_float<half, true>(float a, int n, half placeholder) {
   return  __float2half(a / n);
}

template<>
__device__ inline half __from_float<half, false>(float a, int n, half placeholder) {
   return  __float2half(a);
}

template<typename T>
size_t array_min_max_size(
        const T *input_array,
        int num_items,
        T *output_array,
        cudaStream_t stream) {

     void *dev_buffer = NULL;
     size_t dev_buffer_bytes = 0;

    CUDACHECK(cub::DeviceReduce::Min(
                dev_buffer,
                dev_buffer_bytes,
                input_array,
                output_array,
                num_items,
                stream));

    return dev_buffer_bytes;
}

template<>
size_t array_min_max_size<half>(
        const half *input_array,
        int num_items,
        half *output_array,
        cudaStream_t stream) {

    void *dev_buffer = NULL;
    size_t dev_buffer_bytes = 0;

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array,
            num_items,
            Min(),
            __float2half(65504),  // FIXME
            stream);

    return dev_buffer_bytes;
}

template<typename T>
void array_min_max(
        const T *input_array,
        int num_items,
        void *dev_buffer,
        size_t dev_buffer_bytes,
        T *output_array,
        cudaStream_t stream) {

    CUDACHECK(cub::DeviceReduce::Min(
                dev_buffer,
                dev_buffer_bytes,
                input_array,
                output_array,
                num_items,
                stream));
    
    CUDACHECK(cub::DeviceReduce::Max(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array + 1,
            num_items,
            stream));
}

template<>
void array_min_max<half>(
        const half *input_array,
        int num_items,
        void *dev_buffer,
        size_t dev_buffer_bytes,
        half *output_array,
        cudaStream_t stream) {

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array,
            num_items,
            Min(),
            __float2half(65504),  // FIXME
            stream);

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array + 1,
            num_items,
            Max(),
            __float2half(-65504),  // FIXME
            stream);
}

template<typename T>
__device__ inline uint8_t __minmax_uint8_compress(T f, float scale, float lower_bound, float upper_bound) {
    float level = f * scale;
    level = min(level, upper_bound);
    return level - lower_bound;

}

template<>
__device__ inline uint8_t __minmax_uint8_compress<float>(float f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(f * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<>
__device__ inline uint8_t __minmax_uint8_compress<half>(half f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(__half2float(f) * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<typename T>
__device__ inline T __minmax_uint8_decompress(uint8_t i, float scale, float lower_bound, float upper_bound, T placeholder) {
    return (i + lower_bound) / scale;
}

template<>
__device__ inline half __minmax_uint8_decompress<half>(uint8_t i, float scale, float lower_bound, float upper_bound, half placeholder) {
    return __float2half((i + lower_bound) / scale);
}

template<typename T>
__device__ inline float __load_as_float(T * array) {
    return array[0];
}

template<>
__device__ inline float __load_as_float<half>(half * array) {
    return __half2float(array[0]);
}

template<typename T>
__device__ inline void __store_float(T * array, float data) {
    array[0] = data;
}

template<>
__device__ inline void __store_float<half>(half * array, float data) {
    array[0] = __float2half(data);
}


template<typename T>
__global__ void
compress_float_to_uint8(T *input, int chunk_size, int chunk_offset, int num_chunks, uint8_t *output,
                      size_t output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float min_ = __load_as_float(reinterpret_cast<T *>(output + idy * chunk_offset));
    float max_ = __load_as_float(reinterpret_cast<T *>(output + idy * chunk_offset + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;
    for (int i = idx; i < chunk_size; i += blockDim.x * gridDim.x) {
        int k = idy * chunk_size + i;
        int o = idy * chunk_offset + 32 + i;
        output[o] = __minmax_uint8_compress(input[k], scale, lower_bound, upper_bound);
    }

    if (idx == 0) {
        // write max min to output buffer
        __store_float(reinterpret_cast<T *>(output + idy * chunk_offset), min_);
        __store_float(reinterpret_cast<T *>(output + idy * chunk_offset + sizeof(T)), max_);
    }
}

template<typename T>
__global__ void
compress_float_to_uint8_vector(
        T *input, int num_chunks, long* chunks_offset,
        uint8_t *output, long* outputs_offset, T* min_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float min_ = __load_as_float(min_max);
    float max_ = __load_as_float(min_max + 1);

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;
    int chunk_offset = 0;
    int output_offset = 0;
    if (idy > 0) {
        chunk_offset = chunks_offset[idy - 1];
        output_offset = outputs_offset[idy - 1];
    }
    for (int i = idx; i < (chunks_offset[idy] - chunk_offset); i += blockDim.x * gridDim.x) {
        int k = chunk_offset + i;
        int o = output_offset + i + 32;
        output[o] = __minmax_uint8_compress(input[k], scale, lower_bound, upper_bound);
    }

    if (idx == 0 && chunks_offset[idy] > chunk_offset) {
        // write max min to output buffer
        __store_float(reinterpret_cast<T *>(output + output_offset), min_);
        __store_float(reinterpret_cast<T *>(output + output_offset + sizeof(T)), max_);
    }
}

template<typename T>
__global__ void
decompress_uint8_to_float(uint8_t *input, size_t input_size, int chunk_size, int chunk_offset, int num_chunks,
                          T *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float min_ = __load_as_float(reinterpret_cast<T *>(input + idy * chunk_offset));
    const float max_ = __load_as_float(reinterpret_cast<T *>(input + idy * chunk_offset + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;

    for (int i = idx; i < chunk_size; i += blockDim.x * gridDim.x) {
        int k = idy * chunk_size + i;
        int o = idy * chunk_offset + 32 + i;
        output[k] = __minmax_uint8_decompress(input[o], scale, lower_bound, upper_bound, output[k]);
    }
}

template<typename T>
__global__ void
decompress_uint8_to_float_vector(uint8_t *input, long* inputs_offset, T *output, long* outputs_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int input_offset = 0;
    int output_offset = 0;
    if (idy > 0) {
        input_offset = inputs_offset[idy - 1];
        output_offset = outputs_offset[idy - 1];
    }

    if (output_offset >= outputs_offset[idy]) return;

    const float min_ = __load_as_float(reinterpret_cast<T *>(input + input_offset));
    const float max_ = __load_as_float(reinterpret_cast<T *>(input + input_offset + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;

    for (int i = idx; i < (outputs_offset[idy] - output_offset); i += blockDim.x * gridDim.x) {
        int k = output_offset + i;
        int o = input_offset + 32 + i;
        output[k] = __minmax_uint8_decompress(input[o], scale, lower_bound, upper_bound, output[k]);
    }
}

template<typename T>
void compress_float_to_uint8_host(T *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    int chunk_offset = output_size / num_chunks;
    int remaining_elem = input_num_element;
    for (int i = 0; i < num_chunks; i++) {
        if ((target_chunk == -1) || (i == target_chunk)) {
            array_min_max(input + i * chunk_size, std::min(remaining_elem, chunk_size), dev_buffer, dev_size,
                          reinterpret_cast<T *>(output + i * chunk_offset), stream);
        }
        remaining_elem -= chunk_size;
    }

    if (target_chunk == -1) {
        dim3 num_blocks(DIVUP(chunk_size, 1024), num_chunks);
        compress_float_to_uint8<<<num_blocks, 1024, 0, stream>>>(input, chunk_size, chunk_offset, num_chunks, output,
                                                               output_size);
    } else {
        dim3 num_blocks(DIVUP(chunk_size, 1024), 1);
        T *chunk_input = input + target_chunk * chunk_size;
        uint8_t *chunk_output = output + target_chunk * chunk_offset;

        compress_float_to_uint8<<<num_blocks, 1024, 0, stream>>>(chunk_input, chunk_size, chunk_offset, 1, chunk_output,
                                                               chunk_offset);
    }
    CUDACHECK(cudaGetLastError());
}

template<typename T>
void compress_float_to_uint8_host_vector(
        T *input, int input_num_element, int max_chunk_size, int num_chunks, long* chunks_offset,
        uint8_t *output, long* outputs_offset, T* min_max,
        void *dev_buffer, size_t dev_size, cudaStream_t stream) {
    array_min_max(input, input_num_element, dev_buffer, dev_size, min_max, stream);

    dim3 num_blocks(DIVUP(max_chunk_size, 1024), num_chunks);
    compress_float_to_uint8_vector<<<num_blocks, 1024, 0, stream>>>(
            input, num_chunks, chunks_offset, 
            output, outputs_offset, min_max);
    CUDACHECK(cudaGetLastError());
}

template<typename T>
void decompress_uint8_to_float_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, T *output,
                                   cudaStream_t stream) {

    int chunk_offset = input_size / num_chunks;
    dim3 num_blocks(DIVUP(chunk_size, 1024), num_chunks);
    decompress_uint8_to_float<<<num_blocks, 1024, 0, stream>>>(input, input_size,
                                                             chunk_size, chunk_offset, num_chunks, output);
    CUDACHECK(cudaGetLastError());
}

template<typename T>
void decompress_uint8_to_float_host_vector(
        uint8_t *input, int max_chunk_size, int num_chunks, long* inputs_offset,
        T *output, long* outputs_offset, cudaStream_t stream) {

    dim3 num_blocks(DIVUP(max_chunk_size, 1024), num_chunks);
    decompress_uint8_to_float_vector<<<num_blocks, 1024, 0, stream>>>(input, inputs_offset, output, outputs_offset); 
    CUDACHECK(cudaGetLastError());
}

extern "C" {
void compress_f32_to_uint8_host(float *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    compress_float_to_uint8_host(input, input_num_element, chunk_size, num_chunks, output, output_size, dev_buffer, dev_size, target_chunk, stream);
}

void decompress_uint8_to_f32_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, float *output,
                                  cudaStream_t stream) {
    decompress_uint8_to_float_host(input, input_size, chunk_size, num_chunks, output, stream);
}

void compress_f16_to_uint8_host(half *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    compress_float_to_uint8_host(input, input_num_element, chunk_size, num_chunks, output, output_size, dev_buffer, dev_size, target_chunk, stream);
}

void decompress_uint8_to_f16_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, half *output, 
		                  cudaStream_t stream) {
    decompress_uint8_to_float_host(input, input_size, chunk_size, num_chunks, output, stream);
}

size_t array_min_max_size_f32_host(float *input, int input_num_element, float *output, cudaStream_t stream) {
    return array_min_max_size(input, input_num_element, output, stream);
}

size_t array_min_max_size_f16_host(half *input, int input_num_element, half *output, cudaStream_t stream) {
    return array_min_max_size(input, input_num_element, output, stream);
}

void compress_f16_to_uint8_host_vector(
        half *input, int input_num_element, int max_chunk_size, int num_chunks, long* chunks_offset,
        uint8_t *output, long* outputs_offset, half* min_max,
        void *dev_buffer, size_t dev_size, cudaStream_t stream) {
    if (max_chunk_size == 0) return;
    compress_float_to_uint8_host_vector(
        input, input_num_element, max_chunk_size, num_chunks, chunks_offset,
        output, outputs_offset, min_max,
        dev_buffer, dev_size, stream);
}

void decompress_uint8_to_f16_host_vector(
        uint8_t *input, int max_chunk_size, int num_chunks, long* inputs_offset,
        half *output, long* outputs_offset, cudaStream_t stream) {
    if (max_chunk_size == 0) return;
    decompress_uint8_to_float_host_vector(input, max_chunk_size, num_chunks, inputs_offset,
            output, outputs_offset, stream);
}

}

