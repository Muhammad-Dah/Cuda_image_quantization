#include "image_quantization.h"
#define HIST_SIZE 256
#define THREADS_PER_BLOCK 1024
#define IMG_SIZE (IMG_HEIGHT * IMG_WIDTH)
#define INDEXES_PER_THREAD (IMG_SIZE / THREADS_PER_BLOCK)


/* ------------------------------------------ */
/* ------------------------------------------ */
/* 				 GPU kernels 				  */
/* ------------------------------------------ */
/* ------------------------------------------ */
__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
	int increment;
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if (tid >= stride) {
			increment = arr[tid - stride];
		}
		__syncthreads();
		if (tid >= stride) {
			arr[tid] += increment;
		}
		__syncthreads();
	}
    return;
}

__global__ void process_image_kernel(uchar *all_in, uchar *all_out) {
    __shared__ int hist[THREADS_PER_BLOCK]; // max #threads in block is 1024

    //prepare histpgram	
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid < HIST_SIZE){
		hist[tid]=0;
	}
	
    __syncthreads();
	int array_offset = (bid * IMG_SIZE) + (tid * INDEXES_PER_THREAD);
    for (int stride=0; stride < INDEXES_PER_THREAD; stride++) {
        int offset = array_offset + stride ;
        atomicAdd(&hist[all_in[offset]], 1);
    }
    __syncthreads();
    //cdf : cumulative distribution function
    prefix_sum(hist, HIST_SIZE);
    __syncthreads();
    //map
    for (int stride=0; stride < INDEXES_PER_THREAD; ++stride) {
        int offset = array_offset + stride;
        int temp = ((N_COLORS * hist[all_in[offset]]) / IMG_SIZE);
        all_out[offset] = (HIST_SIZE * temp) / N_COLORS;
    }
    return;
}


/* ------------------------------------------ */
/* ------------------------------------------ */
/* 				 GPU memory helpers			  */
/* ------------------------------------------ */
/* ------------------------------------------ */
/* Allocate + Free GPU memory helpers for a single image.*/

inline void allocate_gpu_helper(uchar ** pointer, int pixels_number){
	CUDA_CHECK(cudaMalloc(pointer , pixels_number * sizeof(uchar)));
}
inline void free_gpu_helper(uchar * pointer){
	CUDA_CHECK(cudaFree(pointer));
}	
inline void copy_gpu_helper(uchar * device, uchar * host, int pixels_number, cudaMemcpyKind kind){
    CUDA_CHECK(cudaMemcpy(device, host, pixels_number*sizeof(uchar), kind));
}	

/* ------------------------------------------ */
/* ------------------------------------------ */
/* 				 serial GPU					  */
/* ------------------------------------------ */
/* ------------------------------------------ */

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *image_in_gpu;
	uchar *image_out_gpu;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;


	allocate_gpu_helper(&context->image_in_gpu, IMG_HEIGHT * IMG_WIDTH);
	allocate_gpu_helper(&context->image_out_gpu, IMG_HEIGHT * IMG_WIDTH);
    return context;
}
/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{

    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
	for (int image_index=0; image_index<N_IMAGES ; ++image_index) {
		
        int image_offset = image_index * IMG_HEIGHT * IMG_WIDTH;
		copy_gpu_helper(context->image_in_gpu, images_in+image_offset, IMG_HEIGHT*IMG_WIDTH, cudaMemcpyHostToDevice);
		
        process_image_kernel<<<1, THREADS_PER_BLOCK>>>(context->image_in_gpu, context->image_out_gpu);
        cudaDeviceSynchronize();
		
		copy_gpu_helper(images_out+image_offset, context->image_out_gpu, IMG_HEIGHT*IMG_WIDTH, cudaMemcpyDeviceToHost);
	}
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    // free resources allocated in task_serial_init
	free_gpu_helper(context->image_in_gpu);
	free_gpu_helper(context->image_out_gpu);
    free(context);
}


/* ------------------------------------------ */
/* ------------------------------------------ */
/* 				 Bulk GPU					  */
/* ------------------------------------------ */
/* ------------------------------------------ */

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // define bulk-GPU memory buffers
	uchar *image_in_gpu;
	uchar *image_out_gpu;
};

/* Allocate GPU memory for all the input and output images.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    // allocate GPU memory for a all input images and all output images
	allocate_gpu_helper(&context->image_in_gpu, IMG_SIZE * N_IMAGES);
	allocate_gpu_helper(&context->image_out_gpu, IMG_SIZE * N_IMAGES);
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    // copy all input images from images_in to the GPU memory you allocated
    // invoke a kernel with N_IMAGES threadblocks, each working on a different image
    // copy output images from GPU memory to images_out
	copy_gpu_helper(context->image_in_gpu, images_in, IMG_SIZE*N_IMAGES, cudaMemcpyHostToDevice);
    
	process_image_kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>(context->image_in_gpu, context->image_out_gpu);
    cudaDeviceSynchronize();
   
	copy_gpu_helper(images_out, context->image_out_gpu, IMG_SIZE*N_IMAGES, cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    // free resources allocated in gpu_bulk_init
	free_gpu_helper(context->image_in_gpu);
	free_gpu_helper(context->image_out_gpu);
    free(context);
}

