#include "CUDA_KERNEL.h"

namespace StableFluidsCuda {


	__device__ void set_corner_gpu(float* x, int N);
	__device__ void set_bnd_gpu(int b, float* x, int N, int tid);
	__device__ void lin_solve_gpu(int b, float* x, float* x0, float a, float c, int iter, int N, int tid);
	__global__ void diffuse_gpu(int b, float* x, float* x0, float diff, float dt, int iter, int N);
	__global__ void project_gpu(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	__global__ void advect_gpu(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N);
}