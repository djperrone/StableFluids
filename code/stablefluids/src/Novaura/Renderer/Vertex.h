#pragma once

#include <glm/glm.hpp>
#include "CudaSrc/CudaMath.cuh"

namespace Novaura{
	struct VertexData
	{
		VertexData(const glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord, const glm::vec2& quantity, float slot)
			: Position(position), Color(color), TexCoord(texCoord), Quantity(quantity), TextureSlot(slot){}
		VertexData(const glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord, const glm::vec2& quantity) : Position(position), Color(color), TexCoord(texCoord), Quantity(quantity) {}
		VertexData(const glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord) : Position(position), Color(color), TexCoord(texCoord) {}
		glm::vec3 Position;
		glm::vec4 Color;
		glm::vec2 TexCoord;
		glm::vec2 Quantity = glm::vec2(1.0f, 1.0f);
		float TextureSlot = 0.0f;
	};

	struct InstancedVertexData_glm
	{
		InstancedVertexData_glm(const glm::vec4& position, const glm::vec4& color)
			: Position(position), Color(color) {}

		glm::vec4 Position;
		glm::vec4 Color;		
	};

	//4th pos is scale
	struct InstancedVertexData
	{
		InstancedVertexData(const CudaMath::Vector4f& position, const CudaMath::Vector4f& color) 
			: Position(position), Color(color) {}

		CudaMath::Vector4f Position;		
		CudaMath::Vector4f Color;
	};

	struct InstancedInteropVertexData
	{
		InstancedInteropVertexData(const CudaMath::Vector4f& position)
			: Position(position){}

		CudaMath::Vector4f Position;		
	};


}

