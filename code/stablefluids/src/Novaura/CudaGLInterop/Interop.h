#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cudagl.h>



namespace CudaGLInterop {

	//void InitDevices();
    // cuda samlples
  /*  int gpuDeviceInit(int devID);
    const char* _ConvertSMVer2ArchName(int major, int minor);
    int _ConvertSMVer2Cores(int major, int minor);
    int gpuGetMaxGflopsDeviceId();
    int findCudaDevice(int argc, const char** argv);
    int findIntegratedGPU();
    bool checkCudaCapabilities(int major_version, int minor_version);*/


    //--------------------------------------------------------

	void SetDefaultCudaDevice();
	void RegisterCudaGLBuffer(struct cudaGraphicsResource* positionsVBO_CUDA, GLuint* positionsVBO);

	void MapCudaGLMatrixBuffer(struct cudaGraphicsResource* positionsVBO_CUDA, size_t* num_bytes, glm::mat4* matrices);

	void UnMapCudaGLMatrixBuffer(struct cudaGraphicsResource* positionsVBO_CUDA);

}

