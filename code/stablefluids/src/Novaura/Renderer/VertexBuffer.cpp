#include "sapch.h"
#include "VertexBuffer.h"
#include <glad/glad.h>

namespace Novaura {
	VertexBuffer::VertexBuffer()
	{
		glGenBuffers(1, &m_VertexBufferID);
	}
	VertexBuffer::VertexBuffer(float* vertices, unsigned int size)
	{
		glGenBuffers(1, &m_VertexBufferID);
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferID);
		glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_DYNAMIC_DRAW);

	}

	VertexBuffer::~VertexBuffer()
	{
		glDeleteBuffers(1, &m_VertexBufferID);
	}

	void VertexBuffer::Bind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferID);
	}

	void VertexBuffer::UnBind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	void VertexBuffer::SetData(const std::vector<VertexData>& vertices)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferID);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);
	}
	void VertexBuffer::SetData(const std::vector<VertexData>& vertices, uint32_t start, uint32_t end)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferID);
		glBufferData(GL_ARRAY_BUFFER, (end - start) * sizeof(VertexData), &vertices[start], GL_STATIC_DRAW);
		//glBufferData(GL_ARRAY_BUFFER, end - start * sizeof(VertexData), &vertices[start], GL_STATIC_DRAW);
	}
	void VertexBuffer::SetData(float* vertices, unsigned int size)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferID);
		glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);
	}
}