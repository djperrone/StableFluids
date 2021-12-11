#pragma once
#include "VertexBuffer.h"
#include <glad/glad.h>

namespace Novaura {

	class VertexArray
	{
	public:
		VertexArray();
		VertexArray(const VertexArray& other) = default;
		VertexArray(VertexArray&& other) = default;
		~VertexArray();

		//void Delete()const;

		void Bind() const;
		void UnBind() const;

		inline unsigned int GetID() const { return m_VertexArrayID; }
		
		void AddBuffer(const VertexBuffer& vb, unsigned int location, unsigned int size, GLenum type, unsigned char normalized, unsigned int stride, unsigned int offset);
	private:
		unsigned int m_VertexArrayID;
	};
}