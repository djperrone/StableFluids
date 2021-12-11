#pragma once
#include <glm/glm.hpp>
//#define IX(x, y) (x + y*N)
#define IX(x, y) (glm::clamp(x,0,N-1) + glm::clamp(y,0,N-1)*N)

namespace StableFluids {

    typedef struct {
        int size;
        float dt;
        float diff;
        float visc;

        float* density0;
        float* density;

        float* Vx;
        float* Vy;

        float* Vx0;
        float* Vy0;
    } FluidSquare;

    FluidSquare* FluidSquareCreate(int size, float diffusion, float viscosity, float dt);
    void FluidSquareFree(FluidSquare* sq);

    void FluidSquareAddDensity(FluidSquare* sq, int x, int y, float amount);
    void FluidSquareAddVelocity(FluidSquare* sq, int x, int y, float amountX, float amountY);

    void FluidSquareStep(FluidSquare* sq);
}
