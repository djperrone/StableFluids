#include "sapch.h"


#include "VertexArray.h"

#include <glad/glad.h>
#include <spdlog/spdlog.h>
namespace Novaura {

	VertexArray::VertexArray()
	{
		glGenVertexArrays(1, &m_VertexArrayID);
	}

	VertexArray::~VertexArray()
	{
		//spdlog::error("va dest\n");
		glDeleteVertexArrays(1, &m_VertexArrayID);
	}	

	void VertexArray::AddBuffer(const VertexBuffer& vb, unsigned int location, unsigned int size, GLenum type, unsigned char normalized, unsigned int stride, unsigned int offset)
	{
		Bind();
		vb.Bind();
		glEnableVertexAttribArray(location);
		glVertexAttribPointer(location, size, type, normalized, stride, (void*)offset);

	}



	void VertexArray::Bind() const
	{
		glBindVertexArray(m_VertexArrayID);
	}

	void VertexArray::UnBind() const
	{
		glBindVertexArray(0);
	}
}