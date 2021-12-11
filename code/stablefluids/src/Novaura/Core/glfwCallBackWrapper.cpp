#include "sapch.h"
#include "glfwCallBackWrapper.h"
#include "Application.h"
namespace Novaura {

    Application* GLFWCallbackWrapper::s_Application = nullptr;

    void GLFWCallbackWrapper::SetApplication(Application* application)
    {
        GLFWCallbackWrapper::s_Application = application;
    }

    void GLFWCallbackWrapper::MouseScrollCallBack(GLFWwindow* window, double xOffset, double yOffset)
    {
        s_Application->MouseScrollCallBack(xOffset, yOffset);
    }

    void GLFWCallbackWrapper::KeyBoardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        s_Application->KeyboardCallback(key, scancode, action, mods);
    }

    void GLFWCallbackWrapper::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        s_Application->MouseButtonCallBack(button, action, mods);

    }

    void GLFWCallbackWrapper::WindowResizeCallBack(GLFWwindow* window, int width, int height)
    {
        s_Application->WindowResizeCallBack(width, height);

    }
    void GLFWCallbackWrapper::MousePositionCallBack(GLFWwindow* window, double positionX, double positionY)
    {
        s_Application->MousePositionCallBack(positionX, positionY);
    }
}