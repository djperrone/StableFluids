#pragma once
#include <GLFW/glfw3.h>

namespace Novaura {

	class Application;

	class GLFWCallbackWrapper
	{
	public:
		GLFWCallbackWrapper() = delete;
		GLFWCallbackWrapper(const GLFWCallbackWrapper&) = delete;
		GLFWCallbackWrapper(GLFWCallbackWrapper&&) = delete;
		~GLFWCallbackWrapper() = delete;

		static void WindowResizeCallBack(GLFWwindow* window, int width, int height);
		static void MousePositionCallBack(GLFWwindow* window, double positionX, double positionY);
		static void MouseScrollCallBack(GLFWwindow* window, double xOffset, double yOffset);
		static void KeyBoardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void SetApplication(Application* application);
	private:
		static Application* s_Application;
	};
}