#include "CUDA_KERNEL.h"
#include "CudaMath.cuh"

namespace CudaMathTest {

	__global__ void TestIdentity_gpu();
	__global__ void TestTranslation_gpu();
	__global__ void TestScale_gpu();
	void MatMul44Test_cpu();
	void MatMul44Test_cpu2();
	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();
	void MatMul44BatchTest_cpu();

}
