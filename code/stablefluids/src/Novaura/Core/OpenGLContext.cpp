#include "sapch.h"
#include "OpenGLContext.h"

#include <spdlog/spdlog.h>



namespace Novaura {

    OpenGLContext::OpenGLContext(std::string_view title, float width, float height)
    {
        // glfw
        m_Window = std::make_shared<Window>();
        glfwInit();
        //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        m_Window->Width = width;
        m_Window->Height = height;
        m_Window->AspectRatio = m_Window->Width / m_Window->Height;
        m_Window->Window = glfwCreateWindow(m_Window->Width, m_Window->Height, title.data(), NULL, NULL);
        if (m_Window->Window == NULL)
        {
            spdlog::error("Failed to create GLFW window");
            glfwTerminate();

            exit(0);
        }
        glfwMakeContextCurrent(m_Window->Window);

        // glad
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            spdlog::error("Failed to initialize GLAD");
            exit(0);
        }

        glViewport(0, 0, m_Window->Width, m_Window->Height);      

        spdlog::info("glfw initialized");

       spdlog::info("OpenGL Info:");
       spdlog::info("  Vendor: {0}", glGetString(GL_VENDOR));
       spdlog::info("  Renderer: {0}", glGetString(GL_RENDERER));
       spdlog::info("  Version: {0}", glGetString(GL_VERSION));       

     
       
    }
    void OpenGLContext::SwapBuffers() const
    {
        glfwSwapBuffers(m_Window->Window);
    }

    bool OpenGLContext::IsRunning() const
    {
        return !glfwWindowShouldClose(m_Window->Window);
    }

    OpenGLContext::~OpenGLContext()
    {
        glfwTerminate();

    }

    void OpenGLContext::PollEvents()
    {
        glfwPollEvents();
    }
}
