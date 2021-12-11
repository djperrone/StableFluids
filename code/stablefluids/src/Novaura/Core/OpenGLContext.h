#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Window.h"

namespace Novaura {

	class OpenGLContext
	{
	public:
		OpenGLContext() = default;
		OpenGLContext(std::string_view title, float width, float height);
		~OpenGLContext();
	
		void PollEvents();
	
		std::shared_ptr<Window> GetWindow() { return m_Window; }
		const std::shared_ptr<Window> GetWindow() const { return m_Window; }
		
		void SwapBuffers() const;
		bool OpenGLContext::IsRunning() const;	
	
	private:
		std::shared_ptr<Window> m_Window;	
	};
}
