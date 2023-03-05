namespace StableFluids {

	void set_bnd(int b, float* x, int N);
	void lin_solve(int b, float* x, float* x0, float a, float c, int iter, int N);
	void diffuse(int b, float* x, float* x0, float diff, float dt, int iter, int N);
	void project(float* velocX, float* velocY, float* p, float* div, int iter, int N);
	void advect(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N);
}
