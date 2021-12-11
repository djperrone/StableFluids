#include "utilities.cuh"
#include "Fluid.cuh"
#include <math.h>
#include <stdio.h>

namespace StableFluidsCuda {

	__device__ void set_corner_gpu(float* x, int N)
	{
		//printf(__FUNCTION__);

		x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
		x[IX(0, N - 1)] = 0.5 * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
		x[IX(N - 1, 0)] = 0.5 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
		x[IX(N - 1, N - 1)] = 0.5 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);
	}

	__device__ void set_bnd_gpu(int b, float* x, int N, int tid)
	{
		if (tid == 0) {
			set_corner_gpu(x, N);
		}

		if (tid < 1 || tid >= N - 1) return;
		//printf(__FUNCTION__);


		x[IX(tid, 0)] = b == 2 ? -x[IX(tid, 1)] : x[IX(tid, 1)];
		x[IX(tid, N - 1)] = b == 2 ? -x[IX(tid, N - 2)] : x[IX(tid, N - 2)];
		x[IX(0, tid)] = b == 1 ? -x[IX(1, tid)] : x[IX(1, tid)];
		x[IX(N - 1, tid)] = b == 1 ? -x[IX(N - 2, tid)] : x[IX(N - 2, tid)];
	}

	// diffuse call: b, x(Vx0), x0(Vx), a, 1 + 6 * a, iter, N, tid
	// project call: 0, p, div, 1, 6, iter, N, tid
	__device__ void lin_solve_gpu(int b, float* x, float* x0, float a, float c, int iter, int N, int tid)
	{
		// check this if it doesnt work
		int i = tid % N;
		int j = tid / N;

		if (i < 1 || i > N - 2) return;
		if (j < 1 || j > N - 2) return;
		//printf(__FUNCTION__);

		float cRecip = 1.0 / c;
		for (int k = 0; k < iter; k++) {
			x[IX(i, j)] =
				(x0[IX(i, j)]
					+ a * (x[IX(i + 1, j)]
						+ x[IX(i - 1, j)]
						+ x[IX(i, j + 1)]
						+ x[IX(i, j - 1)]
						)) * cRecip;

			set_bnd_gpu(b, x, N, tid);
		}
	}

	// 1, Vx0, Vx, visc, dt, 4, N
	__global__ void diffuse_gpu(int b, float* x, float* x0, int iter, FluidData* data)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int N = data->size;
		if (tid >= N * N) return;
		//printf("N: %i\n", N);

		float a = data->dt * data->diff * (N - 2) * (N - 2);
		lin_solve_gpu(b, x, x0, a, 1 + 6 * a, iter, N, tid);
	}

	// Vx0, Vy0, Vx, Vy, 4, N
	__global__ void project_gpu(float* velocX, float* velocY, float* p, float* div, int iter, FluidData* data)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int N = data->size;

		if (tid >= N * N) return;
		//printf(__FUNCTION__);

		int i = tid % N;
		int j = tid / N;

		if (i < 1 || i > N - 2) return;
		if (j < 1 || j > N - 2) return;


		div[IX(i, j)] = -0.5f * (
			velocX[IX(i + 1, j)]
			- velocX[IX(i - 1, j)]
			+ velocY[IX(i, j + 1)]
			- velocY[IX(i, j - 1)]
			) / N;
		p[IX(i, j)] = 0;

		set_bnd_gpu(0, div, N, tid);
		set_bnd_gpu(0, p, N, tid);
		lin_solve_gpu(0, p, div, 1, 6, iter, N, tid);


		velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
			- p[IX(i - 1, j)]) * N;
		velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
			- p[IX(i, j - 1)]) * N;

		set_bnd_gpu(1, velocX, N, tid);
		set_bnd_gpu(2, velocY, N, tid);
	}

	//1, Vx, Vx0, Vx0, Vy0, dt, N
	__global__ void advect_gpu(int b, float* d, float* d0, float* velocX, float* velocY, FluidData* data)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int N = data->size;

		if (tid >= N * N) return;

		int i = tid % N;
		int j = tid / N;
		//printf(__FUNCTION__);

		if (i < 1 || i > N - 2) return;
		if (j < 1 || j > N - 2) return;

		float i0, i1, j0, j1;

		float dtx = data->dt * (N - 2);
		float dty = data->dt * (N - 2);

		float s0, s1, t0, t1;
		float tmp1, tmp2, x, y;

		float Nfloat = N;
		float ifloat = i;
		float jfloat = j;

		tmp1 = dtx * velocX[tid];
		tmp2 = dty * velocY[tid];
		x = ifloat - tmp1;
		y = jfloat - tmp2;

		if (x < 0.5f) x = 0.5f;
		if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
		i0 = floorf(x);
		i1 = i0 + 1.0f;
		if (y < 0.5f) y = 0.5f;
		if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
		j0 = floorf(y);
		j1 = j0 + 1.0f;

		s1 = x - i0;
		s0 = 1.0f - s1;
		t1 = y - j0;
		t0 = 1.0f - t1;

		int i0i = i0;
		int i1i = i1;
		int j0i = j0;
		int j1i = j1;

		d[tid] =
			s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)])
			+ s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);

		set_bnd_gpu(b, d, N,tid);
	}
}
