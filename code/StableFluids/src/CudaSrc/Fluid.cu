#include "Fluid.cuh"
#include "Novaura/CudaGLInterop/helper_cuda.h"
#include "utilities.cuh"



namespace StableFluidsCuda {

    void FluidSquareCreate(FluidSquare* sq, int size, float diffusion, float viscosity, float dt) {

     
        FluidData localData = { size, diffusion, viscosity, dt };   
     

        int N = size;
        sq->data = { size, diffusion, viscosity, dt };       


        checkCudaErrors(cudaMalloc((void**)&sq->density0, sizeof(float) *N* N));
        cudaMemset(sq->density0, 0, sizeof(float) * N * N);      

        checkCudaErrors(cudaMalloc((void**)&sq->density, sizeof(float) * N * N));
        cudaMemset(sq->density, 0, sizeof(float) * N * N);    

        checkCudaErrors(cudaMalloc((void**)&sq->Vx, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy, sizeof(float) * N * N));
        cudaMemset(sq->Vx, 0, sizeof(float) * N * N);
        cudaMemset(sq->Vy, 0, sizeof(float) * N * N);

        checkCudaErrors(cudaMalloc((void**)&sq->Vx0, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy0, sizeof(float) * N * N));
        cudaMemset(sq->Vx0, 0, sizeof(float) * N * N);
        cudaMemset(sq->Vy0, 0, sizeof(float) * N * N);       
    }

    void FluidSquareCreate_cpu(FluidSquare* sq, int size, float diffusion, float viscosity, float dt)
    {
        int N = size;

        sq->data = { (int)sqrt(size), diffusion, viscosity, dt };    

      
        sq->density0 = (float*)calloc(N * N, sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N, sizeof(float));
        sq->Vy = (float*)calloc(N * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N, sizeof(float));
        sq->Vy0 = (float*)calloc(N * N, sizeof(float));
      
    }

    FluidSquare* FluidSquareCreate_cpu(int size, float diffusion, float viscosity, float dt)
    {
        FluidSquare* sq = (FluidSquare*)malloc(sizeof(FluidSquare));
        int N = size;

        sq->data = { N, diffusion, viscosity, dt };

        sq->density0 = (float*)calloc(N * N, sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N, sizeof(float));
        sq->Vy = (float*)calloc(N * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N, sizeof(float));
        sq->Vy0 = (float*)calloc(N * N, sizeof(float));

        return sq;
    }

    void CopyToCPU(FluidSquare sq_cpu, FluidSquare sq_gpu, int N)
    {       
        cudaMemcpy(sq_cpu.density0, (void*)sq_gpu.density0, sizeof(float)*N  * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu.density, (void*)sq_gpu.density, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu.Vx, (void*)sq_gpu.Vx, sizeof(float) * N *  N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu.Vy, (void*)sq_gpu.Vy, sizeof(float) * N *  N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu.Vx0, (void*)sq_gpu.Vx0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu.Vy0, (void*)sq_gpu.Vy0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        //memcpy(&sq_cpu.data, &sq_gpu.data, sizeof(FluidData));        
    }

    void FluidSquareFree(FluidSquare* sq) {

        checkCudaErrors(cudaFree(sq->density0));
        checkCudaErrors(cudaFree(sq->density));

        checkCudaErrors(cudaFree(sq->Vx));
        checkCudaErrors(cudaFree(sq->Vy));

        checkCudaErrors(cudaFree(sq->Vx0));
        checkCudaErrors(cudaFree(sq->Vy0));

        //checkCudaErrors(cudaFree(sq));
    }

    void FluidSquareFree_cpu(FluidSquare* sq)
    {
        free(sq->density0);
        free(sq->density);

        free(sq->Vx);
        free(sq->Vy);

        free(sq->Vx0);
        free(sq->Vy0);

        //free(sq);
    }

    __global__ void FluidSquareAddDensity_gpu(float* density, int x, int y, float amount, int N)
    {
        int index = IX(x, y);

        if (index < 0 || index >= N * N) return;

        density[index] += amount;
      //  printf("tid: %d, gpu add density amount gpu = %f\n", threadIdx.x + blockIdx.x, density[index]);
    }
    __global__ void FluidSquareAddVelocity_gpu(float* velocityX, float* velocityY, int x, int y, float amountX, float amountY, int N)
    {
        int index = IX(x, y);

        if (index < 0 || index >= N * N) return;

        velocityX[index] += amountX;
        velocityY[index] += amountY;
    }

    void FluidSquareAddDensity(FluidSquare* sq, int x, int y, float amount) {
      //  printf("add density cpu\n");
        int n = sq->data.size;
        FluidSquareAddDensity_gpu CUDA_KERNEL(1, 1)(sq->density, x, y, amount,n);
    }

    void FluidSquareAddVelocity(FluidSquare* sq, int x, int y, float amountX, float amountY) {

        FluidSquareAddVelocity_gpu CUDA_KERNEL(1, 1)(sq->Vx, sq->Vy, x, y, amountX, amountY, sq->data.size);
    }

    void FluidSquareStep(FluidSquare* sq, Timer& timer) {

        int N = sq->data.size;
        float visc = sq->data.visc;
        float diff = sq->data.diff;
        float dt = sq->data.dt;      
       
        int blks = (N * N + NUM_THREADS - 1) / NUM_THREADS;
        if (timer.GPU == false)
        {
           // timer.SetFunctionName("diffuse_gpu vx");
           // timer.Start();

            timer.BeginTimeFunction("diffuse_gpu vx");
            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS)  (1, sq->Vx0, sq->Vx, visc, dt, 4, N);
            cudaDeviceSynchronize();
            timer.EndTimeFunction();

            //timer.Flush();

            //timer.SetFunctionName("diffuse_gpu vy");
            //timer.Start();

            timer.BeginTimeFunction("diffuse_gpu vy");
            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS) (2, sq->Vy0, sq->Vy, visc, dt, 4, N);
            cudaDeviceSynchronize();
            timer.EndTimeFunction();
           // timer.Flush();

           // timer.SetFunctionName("project 1");
           // timer.Start();

            timer.BeginTimeFunction("project 1");
            project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq->Vx0, sq->Vy0, sq->Vx, sq->Vy, 4, N);
            cudaDeviceSynchronize();
            timer.EndTimeFunction();
           // timer.Flush();

            /*timer.SetFunctionName("advect_gpu vx");
            timer.Start();*/

            timer.BeginTimeFunction("advect_gpu vx");
            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (1, sq->Vx, sq->Vx0, sq->Vx0, sq->Vy0, dt, N);
            cudaDeviceSynchronize();
            timer.EndTimeFunction();

            //timer.Flush();

          /*  timer.SetFunctionName("advect_gpu vy");
            timer.Start();*/
            timer.BeginTimeFunction("advect_gpu vy");
            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (2, sq->Vy, sq->Vy0, sq->Vx0, sq->Vy0, dt, N);
            cudaDeviceSynchronize();
            timer.EndTimeFunction();

           // timer.Flush();

            //timer.SetFunctionName("project_gpu 2");
            //timer.Start();
            timer.BeginTimeFunction("project_gpu 2");
            project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq->Vx, sq->Vy, sq->Vx0, sq->Vy0, 4, N);
            cudaDeviceSynchronize();
            //timer.Flush();
            timer.EndTimeFunction();


           /* timer.SetFunctionName("diffuse_gpu density");
            timer.Start();*/
            timer.BeginTimeFunction("diffuse_gpu density");

            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS) (0, sq->density0, sq->density, diff, dt, 4, N);
            cudaDeviceSynchronize();
            //timer.Flush();
            timer.EndTimeFunction();


            /*timer.SetFunctionName("advect_gpu density");
            timer.Start();*/
            timer.BeginTimeFunction("advect_gpu density");

            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (0, sq->density, sq->density0, sq->Vx, sq->Vy, dt, N);
            //timer.Flush();
            timer.EndTimeFunction();


            timer.GPU = true;
        }
        else
        {
           
            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS)  (1, sq->Vx0, sq->Vx, visc, dt, 4, N);
            cudaDeviceSynchronize();
          

            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS) (2, sq->Vy0, sq->Vy, visc, dt, 4, N);
            cudaDeviceSynchronize();
           

         
            project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq->Vx0, sq->Vy0, sq->Vx, sq->Vy, 4, N);
            cudaDeviceSynchronize();
           

         
            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (1, sq->Vx, sq->Vx0, sq->Vx0, sq->Vy0, dt, N);
            cudaDeviceSynchronize();
          

           
            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (2, sq->Vy, sq->Vy0, sq->Vx0, sq->Vy0, dt, N);
            cudaDeviceSynchronize();
          

           
            project_gpu CUDA_KERNEL(blks, NUM_THREADS) (sq->Vx, sq->Vy, sq->Vx0, sq->Vy0, 4, N);
            cudaDeviceSynchronize();
           
           
            diffuse_gpu CUDA_KERNEL(blks, NUM_THREADS) (0, sq->density0, sq->density, diff, dt, 4, N);
            cudaDeviceSynchronize();
           

          
            advect_gpu CUDA_KERNEL(blks, NUM_THREADS) (0, sq->density, sq->density0, sq->Vx, sq->Vy, dt, N);
           
        }
    }
}