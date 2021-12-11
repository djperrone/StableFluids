#include "sapch.h"

//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>
//#include "fluid.h"
//#include "utilities.h"
//#include "common.h"
//
//#define NSTEPS 100
//namespace StableFluids {
//
//
//    int main(int argc, char** argv)
//    {
//        if (find_option(argc, argv, "-h") >= 0)
//        {
//            printf("Options:\n");
//            printf("-h to see this help\n");
//            printf("-n <int> to set the number of particles\n");
//            printf("-d <int> to set the diffustion\n");
//            printf("-v <int> to set the viscosity\n");
//            printf("-dt <int> to set the time step\n");
//            printf("-o <filename> to specify the output file name\n");
//            return 0;
//        }
//
//        int n = read_int(argc, argv, "-n", 1000);
//        float d = read_float(argc, argv, "-d", 0);
//        float v = read_float(argc, argv, "-v", .00001);
//        float dt = read_float(argc, argv, "-dt", .2);
//
//        int n_per_side = (int)sqrt(n);
//
//        char* savename = read_string(argc, argv, "-o", NULL);
//        FILE* fsave = savename ? fopen(savename, "w") : NULL;
//
//        FluidSquare* sq = FluidSquareCreate(n_per_side, d, v, dt);
//
//        printf("n: %d, n_per_side: %d, d: %f, v: %f, dt: %f\n", n, n_per_side, d, v, dt);
//
//        // add initial density and velocity
//        FluidSquareAddDensity(sq, n_per_side / 2, n_per_side / 2, 100);
//        FluidSquareAddVelocity(sq, n_per_side / 2, n_per_side / 2, 7, 7);
//
//        //printf("Velocity at midpoint start: %f %f\n", sq->Vx[n_per_side/2 + n_per_side/2 * n_per_side], sq->Vy[n_per_side/2 + n_per_side/2 * n_per_side]);
//
//        //
//        //  simulate a number of time steps
//        //
//        // double simulation_time = read_timer( );
//        for (int i = 0; i < NSTEPS; i++) {
//            FluidSquareStep(sq);
//        }
//
//        //printf("Velocity at midpoint end: %f %f\n", sq->Vx[n_per_side/2 + n_per_side/2 * n_per_side], sq->Vy[n_per_side/2 + n_per_side/2 * n_per_side]);
//
//        // simulation_time = read_timer( ) - simulation_time;
//
//        // printf( "n = %d, simulation time = %g seconds\n", n, simulation_time);
//
//        // free( particles );
//        // if( fsave )
//        //     fclose( fsave );
//
//        return 0;
//    }
//}