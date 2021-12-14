#pragma once

#include "CUDA_KERNEL.h"


#define ZERO_FLAT_MATRIX(Mat)  memset(Mat.mat, 0, 16 * sizeof(float));

namespace CudaMath {

	typedef union
	{
		int vec[2];

		struct
		{
			int x, y;
		};
	}Vector2i;

	typedef union
	{
		float vec[2];

		struct
		{
			float x, y;
		};
	}Vector2f;

	typedef union 
	{
		float vec[3];

		struct
		{
			float x, y, z;
		};
	}Vector3f;

	typedef union  
	{
		float vec[4];
		struct
		{
			float x, y, z, w;
		};
	}Vector4f;

	typedef union   
	{
		float mat[16];
		Vector4f rows[4];	
	}Matrix44f;



#define MAKE_IDENTITY(dest)\
dest.rows[0] = CudaMath::Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = CudaMath::Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = CudaMath::Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = CudaMath::Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_SCALE(dest, scale)\
dest.rows[0] = CudaMath::Vector4f({ scale, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = CudaMath::Vector4f({ 0.0f, scale, 0.0f, 0.0f });\
dest.rows[2] = CudaMath::Vector4f({ 0.0f, 0.0f, scale, 0.0f });\
dest.rows[3] = CudaMath::Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_TRANSLATION(dest, vec)\
dest.rows[0] = CudaMath::Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = CudaMath::Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = CudaMath::Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = CudaMath::Vector4f({ vec.x, vec.y, vec.z, 1.0f });

#define MAKE_TRANSLATION_xyz(dest, x,y,z)\
dest.rows[0] = CudaMath::Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = CudaMath::Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = CudaMath::Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = CudaMath::Vector4f({ x,    y,    z,    1.0f });

//#define CLAMP(value,min,max) if(value < min) value = min; if(value > max) value = max;

#define MIN(x, minVal) (x <= minVal ? x : minVal)
#define MAX(x, maxVal) (x >= maxVal ? x : maxVal)

#define CLAMP(val, min, max) MIN(MAX(val, min), max))

	__global__ void MatMul44Batch_gpu(Matrix44f* grid, Matrix44f* B, Matrix44f* C, int N);
	void MatMul44Batch_cpu(Matrix44f* grid, Matrix44f* B, Matrix44f* C, int numParticles);
	
	__global__ void MatMul44_gpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int numParticles);	
	void MatMul44_cpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int N);
	
	void __global__ MakeTranslationMatrices_gpu(Matrix44f* matrices, CudaMath::Vector3f* locations, size_t n);
	void MakeTranslationMatrices_cpu(Matrix44f* matrices, CudaMath::Vector3f* locations, size_t n);

	void UpdateColors_cpu(CudaMath::Vector4f backgroundColor, CudaMath::Vector4f colorMask, float* densityVals, CudaMath::Vector4f* outColors, size_t n);
	__global__ void UpdateColors_gpu(CudaMath::Vector4f backgroundColor, CudaMath::Vector4f colorMask, float* densityVals,  CudaMath::Vector4f* outColors, size_t n);
}
