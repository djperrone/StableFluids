#include "CUDA_KERNEL.h"
#include "Fluid.cuh"

namespace StableFluidsCuda {


	__device__ void set_corner_gpu(float* x, int N);
	__device__ void set_bnd_gpu(int b, float* x, int N, int tid);

	__device__ void lin_solve_gpu(int b, float* x, float* x0, float a, float c, int iter, int N, int tid);

	__global__ void diffuse_gpu(int b, float* x, float* x0, int iter, FluidData* data);
	//__global__ void diffuse_gpu(int b, float* x, float* x0, float diff, float dt, int iter, int N);
	__global__ void project_gpu(float* velocX, float* velocY, float* p, float* div, int iter, FluidData* data);
	//__global__ void project_gpu(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	__global__ void advect_gpu(int b, float* d, float* d0, float* velocX, float* velocY, FluidData* data);
	//__global__ void advect_gpu(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N);

	








	/*__global__ void set_bnd_gpu(int b, float* x, int N);
	__global__ void set_bnd_gpu_edge(int b, float* x, int N);
	__global__ void lin_solve_gpu(int b, float* x, float* x0, float a, float c, int iter, int N);
	__global__ void diffuse_gpu(int b, float* x, float* x0, float diff, float dt, int iter, int N);
	__global__ void project_gpu0(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	__global__ void project_gpu1(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	__global__ void project_gpu2(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	__global__ void advect_gpu(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N);

	void set_bnd_wrapper(int b, float* x, int N);

	void project_wrapper(float* velocX, float* velocY, float* p, float* div, int iter, int N);

	void set_bnd_cpu_ref(int b, float* x, int N);
	void lin_solve_cpu_ref(int b, float* x, float* x0, float a, float c, int iter, int N);
	void diffuse_cpu_ref(int b, float* x, float* x0, float diff, float dt, int iter, int N);
	void project_cpu_ref(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	void advect_cpu_ref(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N);*/
	
}
