#pragma once

#include "Novaura/Primitives/Rectangle.h"
#include "Novaura/Camera/Camera.h"
#include "Novaura/Renderer/Shader.h"
#include "CudaSrc/CudaMath.cuh"


namespace Novaura {

	class Renderer
	{
	public:		
		Renderer() = default;
		static void Init();
		static void Init(const Camera& camera);
		static void Clear();
		static void SetClearColor(float r, float g, float b, float a = 1.0f);

		static void BeginScene(Shader& shader, const Camera& camera);
		static void BeginScene(const Camera& camera);
		static void BeginSceneInstanced(const Camera& camera);
		static void DrawRectangle(const Rectangle& rectangle, const glm::vec2& quantity = { 1.0f,1.0f });
		static void DrawRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity = { 1.0f,1.0f });

		static void DrawRotatedRectangle(const Rectangle& rectangle, const glm::vec2& quantity = { 1.0f,1.0f });
		static void DrawRotatedRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity = { 1.0f,1.0f });

		static void DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity = { 1.0f,1.0f });
		static void DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity = { 1.0f,1.0f });

		static void DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, const glm::vec2& quantity = { 1.0f,1.0f });
		static void DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity = { 1.0f,1.0f });

		//static void DrawInstancedCircle();
		static void DrawInstancedCircle_glm(const Rectangle& rectangle, const glm::vec2& quantity = { 1.0f,1.0f });
		static void DrawInstancedCircle_glm(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity = { 1.0f,1.0f });

		static void InitInstancedCircles_glm(unsigned int amount, float scale, const glm::vec4& color);
		static void UpdateInstancedCircleMatrices_glm(unsigned int amount);
		static void EndInstancedCircles_glm();		
		static void ShutdownInstancedCircles_glm();
		static void ShutdownInstancedCircles();

		static void RegisterCudaGLDevice();
		static void OnReset(unsigned int amount);
		static void InitInteropInstancedCircles_glm(unsigned int amount, float scale, const glm::vec4& color);			

		static void EndInteropInstancedCircles();		

		static void InitInstancedSquares(unsigned int amount, float scale, CudaMath::Vector3f* locations, float* densityVals, const CudaMath::Vector4f& backgroundColor, const CudaMath::Vector4f& colorMask);
		static void UpdateLocationMatrices(CudaMath::Vector3f* locations, float scale, int n);
		static void EndInstancedSquares();
		static void ShutDownInstancedSquares();
		static void UpdateInstancedColors(const CudaMath::Vector4f& backgroundColor, const CudaMath::Vector4f& colorMask, float* densityVals, int n);
		//static void BeginSceneInstanced(const Camera& camera);

	};
}