#include <sapch.h>
#include "Interop.h"

#include "Novaura/Novaura.h"
#include "helper_cuda.h"

namespace CudaGLInterop {

	void RegisterCudaGLBuffer(struct cudaGraphicsResource* positionsVBO_CUDA, GLuint* positionsVBO)
	{
		//cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
		//cudaGetDevice();
		int devID = findCudaDevice(1, nullptr);
		spdlog::info(__FUNCTION__);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, *positionsVBO, cudaGraphicsMapFlagsNone));
	}

	/*void UnRegisterCudaGLBuffer(cudaGraphicsResource* positionsVBO_CUDA, unsigned int positionsVBO)
	{
		cudaGraphicsUnregisterResource(positionsVBO_CUDA);
	}*/

	void MapCudaGLMatrixBuffer(struct cudaGraphicsResource* positionsVBO_CUDA, size_t* num_bytes, glm::mat4* matrices)
	{
		spdlog::info(__FUNCTION__);

		//checkCudaErrors(cudaGraphicsMapResources(1, positionsVBO_CUDA, 0));
		//size_t num_bytes;
		//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&matrices, num_bytes, positionsVBO_CUDA));
	}

	void UnMapCudaGLMatrixBuffer(struct cudaGraphicsResource* positionsVBO_CUDA)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0));
	}



	void SetDefaultCudaDevice()
	{
		cudaGLSetGLDevice(0);
		//cudaError_t cuda_err;
		//unsigned int gl_device_count;
		//int gl_device_id;
		//int cuda_device_id;
		//cuda_err = cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll);
		////SetDevice(cuda_device_id));
		//cuda_err = cudaSetDevice(cuda_device_id);

		//struct cudaDeviceProp props;
		//cuda_err = cudaGetDeviceProperties(&props, gl_device_id);
		//printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);

		//cuda_err = cudaGetDeviceProperties(&props, cuda_device_id);
		//printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);
		//cudaGLSetGLDevice(cudaGetMaxGflopsDeviceId());


	}

	/*int gpuDeviceInit(int devID) {
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr,
			"gpuDeviceInit() CUDA error: "
			"no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	if (devID < 0) {
		devID = 0;
	}

	if (devID > device_count - 1) {
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
			device_count);
		fprintf(stderr,
			">> gpuDeviceInit (-device=%d) is not a valid"
			" GPU device. <<\n",
			devID);
		fprintf(stderr, "\n");
		return -devID;
	}

	int computeMode = -1, major = 0, minor = 0;
	checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
	checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
	checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
	if (computeMode == cudaComputeModeProhibited) {
		fprintf(stderr,
			"Error: device is running in <Compute Mode "
			"Prohibited>, no threads can use cudaSetDevice().\n");
		return -1;
	}

	if (major < 1) {
		fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaSetDevice(devID));
	printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

	return devID;
}*/

	//int _ConvertSMVer2Cores(int major, int minor) {
	//	// Defines for GPU Architecture types (using the SM version to determine
	//	// the # of cores per SM
	//	typedef struct {
	//		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
	//		// and m = SM minor version
	//		int Cores;
	//	} sSMtoCores;

	//	sSMtoCores nGpuArchCoresPerSM[] = {
	//		{0x30, 192},
	//		{0x32, 192},
	//		{0x35, 192},
	//		{0x37, 192},
	//		{0x50, 128},
	//		{0x52, 128},
	//		{0x53, 128},
	//		{0x60,  64},
	//		{0x61, 128},
	//		{0x62, 128},
	//		{0x70,  64},
	//		{0x72,  64},
	//		{0x75,  64},
	//		{0x80,  64},
	//		{0x86, 128},
	//		{-1, -1} };

	//	int index = 0;

	//	while (nGpuArchCoresPerSM[index].SM != -1) {
	//		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
	//			return nGpuArchCoresPerSM[index].Cores;
	//		}

	//		index++;
	//	}

	//	// If we don't find the values, we default use the previous one
	//	// to run properly
	//	printf(
	//		"MapSMtoCores for SM %d.%d is undefined."
	//		"  Default to use %d Cores/SM\n",
	//		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	//	return nGpuArchCoresPerSM[index - 1].Cores;
	//}

	//const char* _ConvertSMVer2ArchName(int major, int minor) {
	//	// Defines for GPU Architecture types (using the SM version to determine
	//	// the GPU Arch name)
	//	typedef struct {
	//		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
	//		// and m = SM minor version
	//		const char* name;
	//	} sSMtoArchName;

	//	sSMtoArchName nGpuArchNameSM[] = {
	//		{0x30, "Kepler"},
	//		{0x32, "Kepler"},
	//		{0x35, "Kepler"},
	//		{0x37, "Kepler"},
	//		{0x50, "Maxwell"},
	//		{0x52, "Maxwell"},
	//		{0x53, "Maxwell"},
	//		{0x60, "Pascal"},
	//		{0x61, "Pascal"},
	//		{0x62, "Pascal"},
	//		{0x70, "Volta"},
	//		{0x72, "Xavier"},
	//		{0x75, "Turing"},
	//		{0x80, "Ampere"},
	//		{0x86, "Ampere"},
	//		{-1, "Graphics Device"} };

	//	int index = 0;

	//	while (nGpuArchNameSM[index].SM != -1) {
	//		if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
	//			return nGpuArchNameSM[index].name;
	//		}

	//		index++;
	//	}

	//	// If we don't find the values, we default use the previous one
	//	// to run properly
	//	printf(
	//		"MapSMtoArchName for SM %d.%d is undefined."
	//		"  Default to use %s\n",
	//		major, minor, nGpuArchNameSM[index - 1].name);
	//	return nGpuArchNameSM[index - 1].name;
	//}

	//// This function returns the best GPU (with maximum GFLOPS)
	//int gpuGetMaxGflopsDeviceId() {
	//	int current_device = 0, sm_per_multiproc = 0;
	//	int max_perf_device = 0;
	//	int device_count = 0;
	//	int devices_prohibited = 0;

	//	uint64_t max_compute_perf = 0;
	//	checkCudaErrors(cudaGetDeviceCount(&device_count));

	//	if (device_count == 0) {
	//		fprintf(stderr,
	//			"gpuGetMaxGflopsDeviceId() CUDA error:"
	//			" no devices supporting CUDA.\n");
	//		exit(EXIT_FAILURE);
	//	}

	//	// Find the best CUDA capable GPU device
	//	current_device = 0;

	//	while (current_device < device_count) {
	//		int computeMode = -1, major = 0, minor = 0;
	//		checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
	//		checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
	//		checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

	//		// If this GPU is not running on Compute Mode prohibited,
	//		// then we can add it to the list
	//		if (computeMode != cudaComputeModeProhibited) {
	//			if (major == 9999 && minor == 9999) {
	//				sm_per_multiproc = 1;
	//			}
	//			else {
	//				sm_per_multiproc =
	//					_ConvertSMVer2Cores(major, minor);
	//			}
	//			int multiProcessorCount = 0, clockRate = 0;
	//			checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
	//			cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
	//			if (result != cudaSuccess) {
	//				// If cudaDevAttrClockRate attribute is not supported we
	//				// set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
	//				if (result == cudaErrorInvalidValue) {
	//					clockRate = 1;
	//				}
	//				else {
	//					fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
	//						static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
	//					exit(EXIT_FAILURE);
	//				}
	//			}
	//			uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

	//			if (compute_perf > max_compute_perf) {
	//				max_compute_perf = compute_perf;
	//				max_perf_device = current_device;
	//			}
	//		}
	//		else {
	//			devices_prohibited++;
	//		}

	//		++current_device;
	//	}

	//	if (devices_prohibited == device_count) {
	//		fprintf(stderr,
	//			"gpuGetMaxGflopsDeviceId() CUDA error:"
	//			" all devices have compute mode prohibited.\n");
	//		exit(EXIT_FAILURE);
	//	}

	//	return max_perf_device;
	//}

	//// Initialization code to find the best CUDA Device
	//int findCudaDevice(int argc, const char** argv) {
	//	int devID = 0;

	//	// If the command-line has a device number specified, use it
	//	if (checkCmdLineFlag(argc, argv, "device")) {
	//		devID = getCmdLineArgumentInt(argc, argv, "device=");

	//		if (devID < 0) {
	//			printf("Invalid command line parameter\n ");
	//			exit(EXIT_FAILURE);
	//		}
	//		else {
	//			devID = gpuDeviceInit(devID);

	//			if (devID < 0) {
	//				printf("exiting...\n");
	//				exit(EXIT_FAILURE);
	//			}
	//		}
	//	}
	//	else {
	//		// Otherwise pick the device with highest Gflops/s
	//		devID = gpuGetMaxGflopsDeviceId();
	//		checkCudaErrors(cudaSetDevice(devID));
	//		int major = 0, minor = 0;
	//		checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
	//		checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
	//		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
	//			devID, _ConvertSMVer2ArchName(major, minor), major, minor);

	//	}

	//	return devID;
	//}

	//int findIntegratedGPU() {
	//	int current_device = 0;
	//	int device_count = 0;
	//	int devices_prohibited = 0;

	//	checkCudaErrors(cudaGetDeviceCount(&device_count));

	//	if (device_count == 0) {
	//		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
	//		exit(EXIT_FAILURE);
	//	}

	//	// Find the integrated GPU which is compute capable
	//	while (current_device < device_count) {
	//		int computeMode = -1, integrated = -1;
	//		checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
	//		checkCudaErrors(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, current_device));
	//		// If GPU is integrated and is not running on Compute Mode prohibited,
	//		// then cuda can map to GLES resource
	//		if (integrated && (computeMode != cudaComputeModeProhibited)) {
	//			checkCudaErrors(cudaSetDevice(current_device));

	//			int major = 0, minor = 0;
	//			checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
	//			checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
	//			printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
	//				current_device, _ConvertSMVer2ArchName(major, minor), major, minor);

	//			return current_device;
	//		}
	//		else {
	//			devices_prohibited++;
	//		}

	//		current_device++;
	//	}

	//	if (devices_prohibited == device_count) {
	//		fprintf(stderr,
	//			"CUDA error:"
	//			" No GLES-CUDA Interop capable GPU found.\n");
	//		exit(EXIT_FAILURE);
	//	}

	//	return -1;
	//}

	//// General check for CUDA GPU SM Capabilities
	//bool checkCudaCapabilities(int major_version, int minor_version) {
	//	int dev;
	//	int major = 0, minor = 0;

	//	checkCudaErrors(cudaGetDevice(&dev));
	//	checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
	//	checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

	//	if ((major > major_version) ||
	//		(major == major_version &&
	//			minor >= minor_version)) {
	//		printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
	//			_ConvertSMVer2ArchName(major, minor), major, minor);
	//		return true;
	//	}
	//	else {
	//		printf(
	//			"  No GPU device was found that can support "
	//			"CUDA compute capability %d.%d.\n",
	//			major_version, minor_version);
	//		return false;
	//	}
	//}


}

//
//void InitDevices()
//{
//	//cudaError_t cuda_err;
//	//unsigned int gl_device_count;
//	//int gl_device_id;
//	//int cuda_device_id;
//	//cuda_err = cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll);
//	////SetDevice(cuda_device_id));
//	//cuda_err = cudaSetDevice(cuda_device_id);
//
//	//struct cudaDeviceProp props;
//	//cuda_err = cudaGetDeviceProperties(&props, gl_device_id);
//	//printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);
//
//	//cuda_err = cudaGetDeviceProperties(&props, cuda_device_id);
//	//printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);
//
//	//cudaStream_t stream;
//	//cudaEvent_t  event;
//	//struct cudaGraphicsResource* cuda_vbo_resource;
//
//	////cuda_err = cudaStreamCreateWithFlags(&stream, cudaStreamDefault);   // optionally ignore default stream behavior
//	////cuda_err = cudaEventCreateWithFlags(&event, cudaEventBlockingSync); // | cudaEventDisableTiming);
//	//unsigned int vbo;
//	//glGenBuffers(1, &vbo);
//	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	////glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS, particles,
//	////	GL_DYNAMIC_DRAW_ARB);
//
//	//cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
//	//	cudaGraphicsMapFlagsNone);
//
//	//cudaGLUnregisterBufferObject(vbo);
//
//	//glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
//
//}