#include "sapch.h"
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include "fluid.h"
#include "utilities.h"
namespace StableFluids {

    FluidSquare* FluidSquareCreate(int size, float diffusion, float viscosity, float dt) {
        //spdlog::info(__FUNCTION__);
        FluidSquare* sq = (FluidSquare*)malloc(sizeof(FluidSquare));
        int N = size;

        sq->size = size;
        sq->dt = dt;
        sq->diff = diffusion;
        sq->visc = viscosity;

        sq->density0 = (float*)calloc(N * N , sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N , sizeof(float));
        sq->Vy = (float*)calloc(N  * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N , sizeof(float));
        sq->Vy0 = (float*)calloc(N * N , sizeof(float));

        return sq;
    }

    void FluidSquareFree(FluidSquare* sq) {
        //spdlog::info(__FUNCTION__);


        free(sq->density0);
        free(sq->density);

        free(sq->Vx);
        free(sq->Vy);

        free(sq->Vx0);
        free(sq->Vy0);

        free(sq);
    }

    void FluidSquareAddDensity(FluidSquare* sq, int x, int y, float amount) {
        //spdlog::info(__FUNCTION__);
        
       // printf("cpu add density amount = %f\n", amount);
        int N = sq->size;
        sq->density[IX(x, y)] += amount;
       // printf("123cpu add density amount = %f\n", sq->density[IX(x, y)]);

    }

    void FluidSquareAddVelocity(FluidSquare* sq, int x, int y, float amountX, float amountY) {
        //spdlog::info(__FUNCTION__);

        int N = sq->size;
        int index = IX(x, y);

        sq->Vx[index] += amountX;
        sq->Vy[index] += amountY;
    }

    void FluidSquareStep(FluidSquare* sq, Timer& timer) {
        
        int N = sq->size;
        float visc = sq->visc;
        float diff = sq->diff;
        float dt = sq->dt;
        float* Vx = sq->Vx;
        float* Vy = sq->Vy;
        float* Vx0 = sq->Vx0;
        float* Vy0 = sq->Vy0;
        float* density0 = sq->density0;
        float* density = sq->density;

        if (timer.CPU == false)
        {
            timer.SetFunctionName("diffuse_cpu vx");
            timer.Start();
            diffuse(1, Vx0, Vx, visc, dt, 4, N);
            timer.Flush();

            timer.SetFunctionName("diffuse_cpu vy");
            timer.Start();
            diffuse(2, Vy0, Vy, visc, dt, 4, N);
            timer.Flush();

            timer.SetFunctionName("project 1");
            timer.Start();
            project(Vx0, Vy0, Vx, Vy, 4, N);
            timer.Flush();


            timer.SetFunctionName("advect_cpu vx");
            timer.Start();
            advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
            timer.Flush();

            timer.SetFunctionName("advect_cpu vy");
            timer.Start();
            advect(2, Vy, Vy0, Vx0, Vy0, dt, N);
            timer.Flush();

            timer.SetFunctionName("project_cpu 2");
            timer.Start();
            project(Vx, Vy, Vx0, Vy0, 4, N);
            timer.Flush();

            timer.SetFunctionName("diffuse_cpu density");
            timer.Start();
            diffuse(0, density0, density, diff, dt, 4, N);
            timer.Flush();

            timer.SetFunctionName("advect_cpu density");
            timer.Start();
            advect(0, density, density0, Vx, Vy, dt, N);
            timer.Flush();
            timer.CPU = true;
        }
        else
        {
            diffuse(1, Vx0, Vx, visc, dt, 4, N);
            diffuse(2, Vy0, Vy, visc, dt, 4, N);

            project(Vx0, Vy0, Vx, Vy, 4, N);

            advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
            advect(2, Vy, Vy0, Vx0, Vy0, dt, N);

            project(Vx, Vy, Vx0, Vy0, 4, N);

            diffuse(0, density0, density, diff, dt, 4, N);
            advect(0, density, density0, Vx, Vy, dt, N);
        }
    }


    void FluidSquareStep(FluidSquare* sq) {

        int N = sq->size;
        float visc = sq->visc;
        float diff = sq->diff;
        float dt = sq->dt;
        float* Vx = sq->Vx;
        float* Vy = sq->Vy;
        float* Vx0 = sq->Vx0;
        float* Vy0 = sq->Vy0;
        float* density0 = sq->density0;
        float* density = sq->density;

        diffuse(1, Vx0, Vx, visc, dt, 4, N);
        diffuse(2, Vy0, Vy, visc, dt, 4, N);

        project(Vx0, Vy0, Vx, Vy, 4, N);

        advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
        advect(2, Vy, Vy0, Vx0, Vy0, dt, N);

        project(Vx, Vy, Vx0, Vy0, 4, N);

        diffuse(0, density0, density, diff, dt, 4, N);
        advect(0, density, density0, Vx, Vy, dt, N);
    }
}