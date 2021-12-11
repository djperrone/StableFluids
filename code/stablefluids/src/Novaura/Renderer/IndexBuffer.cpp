#include "sapch.h"
#include "IndexBuffer.h"
#include <glad/glad.h>

namespace Novaura {


	IndexBuffer::IndexBuffer(unsigned int* indicesData, unsigned int numIndices)
		: m_Count(numIndices)
	{
		glGenBuffers(1, &m_IndexBufferID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBufferID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Count * sizeof(unsigned int), indicesData, GL_STATIC_DRAW);
	}

	//IndexBuffer::IndexBuffer(const std::vector<unsigned int>& indices)
	//	: m_Count(indices.size())
	//{
	//	glGenBuffers(1, &m_IndexBufferID);
	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBufferID);
	//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
	//}

	IndexBuffer::~IndexBuffer()
	{
		glDeleteBuffers(1, &m_IndexBufferID);
	}

	//void IndexBuffer::ReDo()
	//{
	//	glGenBuffers(1, &m_IndexBufferID);
	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBufferID);
	//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Indices.size() * sizeof(unsigned int), &m_Indices[0], GL_STATIC_DRAW);
	//}

	void IndexBuffer::Bind()
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBufferID);
	}

	void IndexBuffer::UnBind()
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	}
}
