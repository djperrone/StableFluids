#pragma once
#include "CUDA_KERNEL.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#define IX(x, y) (glm::clamp(x,0,N-1) + glm::clamp(y,0,N-1)*N)

//#define IX(x, y) (x + y*N)
#define NUM_THREADS 256

#include "Benchmark/timer.h"
#include "Benchmark/CudaTimer.cuh"

namespace StableFluidsCuda {

    typedef struct
    {
        int size;
        float diff;
        float visc;
        float dt;

    }FluidData;

    typedef struct FluidSquare {

        FluidData data;
        float* density0;
        float* density;

        float* Vx;
        float* Vy;

        float* Vx0;
        float* Vy0;
    } FluidSquare;

    void FluidSquareCreate(FluidSquare* sq, int size, float diffusion, float viscosity, float dt);
    void FluidSquareCreate_cpu(FluidSquare* sq, int size, float diffusion, float viscosity, float dt);
    void CopyToCPU(FluidSquare sq_cpu, FluidSquare sq_gpu, int N);

    void FluidSquareFree(FluidSquare* sq);
    void FluidSquareFree_cpu(FluidSquare* sq);

    void FluidSquareStep(FluidSquare* sq);
    void FluidSquareStep(FluidSquare* sq, Timer& timer);
    void FluidSquareStep(FluidSquare* sq, CudaTimer& timer);

    // device, global ??....
    __global__ void FluidSquareAddDensity_gpu(float* density, int x, int y, float amount, int N);
    __global__ void FluidSquareAddVelocity_gpu(float*velocityX, float* velocityY, int x, int y, float amountX, float amountY, int N);
    void FluidSquareAddDensity(FluidSquare* sq, int x, int y, float amount);
    void FluidSquareAddVelocity(FluidSquare* sq, int x, int y, float amountX, float amountY);

}