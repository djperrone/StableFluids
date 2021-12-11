#include "Fluid.cuh"
#include "Novaura/CudaGLInterop/helper_cuda.h"
#include "utilities.cuh"


namespace StableFluidsCuda {

    void FluidSquareCreate(FluidSquare* sq, int size, float diffusion, float viscosity, float dt) {
        printf(__FUNCTION__);
        //FluidData* data = nullptr;
        checkCudaErrors(cudaMalloc((void**)&sq->data, sizeof(FluidData)));
        FluidData localData = { size, diffusion, viscosity, dt };
        checkCudaErrors(cudaMemcpy(sq->data, (void*)&localData, sizeof(FluidData), cudaMemcpyHostToDevice));
        printf("%i\n", __LINE__);  
        
      
        int N = size;       
       
       
        checkCudaErrors(cudaMalloc((void**)&sq->density0, sizeof(float) * N * N));
        cudaMemset(sq->density0, 0,sizeof(float) * N * N);

        printf("%i\n", __LINE__);

        checkCudaErrors(cudaMalloc((void**)&sq->density, sizeof(float) * N * N));
        cudaMemset(sq->density, 0, sizeof(float) * N * N);

        printf("%i\n", __LINE__);

        checkCudaErrors(cudaMalloc((void**)&sq->Vx, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy, sizeof(float) * N * N));
        cudaMemset(sq->Vx, 0, sizeof(float) * N * N);
        cudaMemset(sq->Vy, 0, sizeof(float) * N * N);


        checkCudaErrors(cudaMalloc((void**)&sq->Vx0, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy0, sizeof(float) * N * N));
        cudaMemset(sq->Vx0, 0, sizeof(float) * N * N);
        cudaMemset(sq->Vy0,0, sizeof(float) * N * N);
        printf("%i\n", __LINE__);

       // return sq;
    }

    void FluidSquareCreate_cpu(FluidSquare* sq, int size, float diffusion, float viscosity, float dt)
    {
        printf(__FUNCTION__);

        int N = size;

        sq->data = (FluidData*)malloc(sizeof(FluidData));
        
        sq->data->size = size;
        sq->data->diff = diffusion;
        sq->data->visc = viscosity;
        sq->data->dt = dt;

        sq->density0 = (float*)calloc(N * N, sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N, sizeof(float));
        sq->Vy = (float*)calloc(N * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N, sizeof(float));
        sq->Vy0 = (float*)calloc(N * N, sizeof(float));
        printf(__FUNCTION__);


    }

    //FluidSquare* FluidSquareCreate_cpu(int size, float diffusion, float viscosity, float dt)
    //{
    //   /* printf(__FUNCTION__);

    //    FluidSquare* sq = (FluidSquare*)malloc(sizeof(FluidSquare));
    //    int N = size;

    //    sq->data = { size, diffusion, viscosity, dt };

    //    sq->density0 = (float*)calloc(N * N, sizeof(float));
    //    sq->density = (float*)calloc(N * N, sizeof(float));

    //    sq->Vx = (float*)calloc(N * N, sizeof(float));
    //    sq->Vy = (float*)calloc(N * N, sizeof(float));

    //    sq->Vx0 = (float*)calloc(N * N, sizeof(float));
    //    sq->Vy0 = (float*)calloc(N * N, sizeof(float));*/

    //    //return sq;
    //}

    void CopyToCPU(FluidSquare* sq_cpu, FluidSquare* sq_gpu, int N)
    {       
        cudaMemcpy(sq_cpu->density0, (void*)sq_gpu->density0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy((void*)sq_cpu->density, (void*)sq_gpu->density, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu->Vx, (void*)sq_gpu->Vx, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu->Vy, (void*)sq_gpu->Vy, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu->Vx0, (void*)sq_gpu->Vx0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu->Vy0, (void*)sq_gpu->Vy0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu->data, (void*)sq_gpu->data, sizeof(FluidData), cudaMemcpyDeviceToHost);
    }

    void FluidSquareFree(FluidSquare sq) {      

        checkCudaErrors(cudaFree(sq.density0));
        checkCudaErrors(cudaFree(sq.density));

        checkCudaErrors(cudaFree(sq.Vx));
        checkCudaErrors(cudaFree(sq.Vy));

        checkCudaErrors(cudaFree(sq.Vx0));
        checkCudaErrors(cudaFree(sq.Vy0));

        checkCudaErrors(cudaFree(sq.data));
    }

    void FluidSquareFree_cpu(FluidSquare* sq)
    {
        free(sq->density0);
        free(sq->density);

        free(sq->Vx);
        free(sq->Vy);

        free(sq->Vx0);
        free(sq->Vy0);

        free(sq->data);
    }

    __global__ void FluidSquareAddDensity_gpu(float* density, int x, int y, float amount, int n_per_side)
    {
        //printf(__FUNCTION__);
      /*  printf("x: %d\n", x);
        printf("y: %d\n", y);
        printf("amount: %f\n", amount);*/
       // printf("cpu add density amount gpu = %f\n", amount);

       // int N = sq.data->size;
        int N = n_per_side;
        int index = IX(x, y);

        if (index < 0 || index >= N * N) return;
       
        density[index] += amount;
        printf("tid: %d, gpu add density amount gpu = %f\n",threadIdx.x + blockIdx.x, density[index]);

    }

    __global__ void FluidSquareAddVelocity_gpu(FluidSquare sq, int x, int y, float amountX, float amountY, int n_per_side)
    {
       //printf(__FUNCTION__);
      // printf("\n");

        int N = sq.data->size;
        int index = IX(x, y);
        
        if (index < 0 || index >= N * N) return;

        sq.Vx[index] += amountX;
        sq.Vy[index] += amountY;
    }

    void FluidSquareAddDensity(FluidSquare sq, int x, int y, float amount, int n_per_side) {

       // printf(__FUNCTION__);
       // printf("\n");
       // printf("cpu add density amount gpu wrapper = %f\n", amount);
       // float* density_gpu;
       // float density_cpu[100] = { 0 };

       // cudaMalloc((void**)&density_gpu, sizeof(float) * 100);
      //  cudaMemset((void*)density_gpu, 0, sizeof(float) * 100);
      
        FluidSquareAddDensity_gpu CUDA_KERNEL(1, 1)(sq.density, x, y, amount, n_per_side);       

       // cudaMemcpy((void*)density_cpu, (void*)density_gpu, sizeof(float) * 100, cudaMemcpyDeviceToHost);
       // printf("density value on float array: %f\n", density_cpu[IX(5,5)]);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
        {
            printf("fensity kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
            __debugbreak;
        }
    }

    void FluidSquareAddVelocity(FluidSquare sq, int x, int y, float amountX, float amountY, int n_per_side) {
       // printf(__FUNCTION__);
       // printf("\n");

        FluidSquareAddVelocity_gpu CUDA_KERNEL(1, 1)(sq, x, y, amountX, amountY, n_per_side);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
        {
            printf("velocty kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
            __debugbreak;
        }
    }

    __global__ void DebugPrint(FluidData data)
    {
        printf("data size: %i\n", data.size);
        printf("data diff: %i\n", data.diff);
        printf("data visc: %i\n", data.visc);
        printf("data dt: %i\n", data.dt);
    }

    __global__ void DebugPrint2(int n)
    {
       // n += threadIdx.x;
        printf("2 data size: %i\n",n);
    }

    void FluidSquareStep(FluidSquare sq, FluidSquare sq_cpu, int N) {
       // printf(__FUNCTION__);

       // int N = sq_cpu.data->size;
     /*   float visc = sq_cpu.data->visc;
        float diff = sq_cpu.data->diff;
        float dt = sq_cpu.data->dt;*/



      /*  float* Vx = sq_cpu.Vx;
        float* Vy = sq_cpu.Vy;
        float* Vx0 = sq_cpu.Vx0;
        float* Vy0 = sq_cpu.Vy0;
        float* density0 = sq_cpu.density0;
        float* density = sq_cpu.density;      */

        int blks = (N * N + NUM_THREADS - 1) / NUM_THREADS;

        diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS)  (1, sq.Vx0, sq.Vx, 4, sq.data);
        cudaDeviceSynchronize();

       /* diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS) (2, sq.Vy0, sq.Vy,4, sq.data);
        cudaDeviceSynchronize();

        project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq.Vx0, sq.Vy0, sq.Vx, sq.Vy, 4, sq.data);

        cudaDeviceSynchronize();
        advect_gpu CUDA_KERNEL (blks, NUM_THREADS) (1, sq.Vx, sq.Vx0, sq.Vx0, sq.Vy0, sq.data);
        cudaDeviceSynchronize();

        advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (2, sq.Vy, sq.Vy0, sq.Vx0, sq.Vy0, sq.data);
        cudaDeviceSynchronize();

        project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq.Vx, sq.Vy, sq.Vx0, sq.Vy0, 4, sq.data);
        cudaDeviceSynchronize();

        diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS) (0, sq.density0, sq.density, 4, sq.data);
        cudaDeviceSynchronize();

        advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (0, sq.density, sq.density0, sq.Vx, sq.Vy, sq.data);
        cudaDeviceSynchronize();*/
    }
}
