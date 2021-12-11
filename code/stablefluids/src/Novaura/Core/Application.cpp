#include "sapch.h"
#include "Application.h"
#include "Novaura/Renderer/Renderer.h"
#include "Novaura/Renderer/BatchRenderer.h"
#include <spdlog/spdlog.h>
#include "Novaura/Renderer/Texture.h"
#include "glfwCallBackWrapper.h"
#include "Novaura/Input/InputHandler.h"
#include "Novaura/StateMachine/State.h"
#include "Novaura/Random.h"
namespace test {

	glm::vec3 velocity(0.5f, 0.5f, 0.0f);
	
}

namespace Novaura {
	std::shared_ptr<CameraController> Application::m_CameraController;
	std::shared_ptr <StateMachine> Application::m_StateMachine;

	Application::Application()
		: m_Context("Space Adventures", 1280.0f,720.0f)
	{		
		m_StateMachine = std::make_shared<StateMachine>();
		m_CameraController = std::make_shared<CameraController>(m_Context.GetWindow()->Width, m_Context.GetWindow()->Height);
		Novaura::BatchRenderer::Init();
		Novaura::Renderer::Init();

		Novaura::InputHandler::Init();
		Novaura::InputHandler::SetCurrentWindow(m_Context.GetWindow());
		
		SetCallBackFunctions();				
	}

	Application::Application(std::string_view title, float width, float height)
		: m_Context(title, width, height) 
	{
		m_StateMachine = std::make_shared<StateMachine>();
		m_CameraController = std::make_shared<CameraController>(width, height);
		InputHandler::Init();
		InputHandler::SetCurrentWindow(m_Context.GetWindow());
		Novaura::BatchRenderer::Init();
		Novaura::Renderer::Init();

		Random::Init();
		SetCallBackFunctions();
	}

	Application::~Application()
	{
		
	}

	void Application::Update()
	{
		float currentFrame = glfwGetTime();
		m_DeltaTime = currentFrame - m_LastFrame;
		m_LastFrame = currentFrame;
		m_Context.PollEvents();


		for (auto& [key, command] : InputHandler::GetCurrentController().GetAxisInputBindings())
		{
			if (InputHandler::IsPressed(key))
			{
				command.Execute();
			}
		}

		
		if (m_StateMachine->GetCurrentState().IsPaused())
		{
			auto currentState = std::move(m_StateMachine->GetStateStack().top());

			m_StateMachine->GetStateStack().pop();			
			m_StateMachine->GetCurrentState().Draw(m_DeltaTime);

			currentState->Update(m_DeltaTime);
			currentState->Draw(m_DeltaTime);
			m_StateMachine->PushState(std::move(currentState));

		}
		else
		{
			m_StateMachine->GetCurrentState().Update(m_DeltaTime);
			m_StateMachine->GetCurrentState().Draw(m_DeltaTime);
		}

		m_Context.SwapBuffers();		
	}	

	void Application::WindowResizeCallBack(int width, int height)
	{
		glViewport(0, 0, width, height);
		GetWindow()->Width = width;
		GetWindow()->Height = height;
		GetWindow()->AspectRatio = (float)width / (float)height;

		m_CameraController->GetCamera().SetProjectionMatrix(width, height);
	}

	void Application::MouseScrollCallBack(double xoffset, double yoffset)
	{
		m_CameraController->ProcessMouseScroll(yoffset);
	}

	void Application::KeyboardCallback(int key, int scancode, int action, int mods)
	{
		InputController controller = InputHandler::GetCurrentController();

		if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		{
			spdlog::info("space");
		}

		

		if (controller.GetActionInputBindings().find(action) != controller.GetActionInputBindings().end())
		{
			if (controller.GetActionInputBindings()[action].find(key) != controller.GetActionInputBindings()[action].end())
			{
				controller.GetActionInputBindings()[action][key].Execute();
			}
		}		
	}	

	void Application::MouseButtonCallBack(int button, int action, int mods)
	{
		InputController controller = InputHandler::GetCurrentController();

		if (controller.GetAxisInputBindings().find(button) != controller.GetAxisInputBindings().end())
		{
			controller.GetAxisInputBindings()[button].Execute();
		}

		if (controller.GetActionInputBindings().find(action) != controller.GetActionInputBindings().end())
		{
			if (controller.GetActionInputBindings()[action].find(button) != controller.GetActionInputBindings()[action].end())
			{
				controller.GetActionInputBindings()[action][button].Execute();
			}
		}
	}

	void Application::MousePositionCallBack(double positionX, double positionY)
	{
		//spdlog::info("{0}, {1}", positionX, positionY);
	}


	void Application::ScreenSaver()
	{		
			//float aspectRatio = m_Context.GetWindow().Width / m_Context.GetWindow().Height;

			//BeginFrame();
			////m_Camera.SetProjectionMatrix(m_Context.GetWindow().Width, m_Context.GetWindow().Height);
			//m_CameraController.Update(GetWindow(), m_DeltaTime);

			//Renderer::BeginScene(*m_Shader, m_CameraController.GetCamera());
			////Renderer::BeginScene(*m_Shader, m_Camera);
			//m_Rect->m_Rotation += 1.0f * m_DeltaTime;
			//m_Rect->m_Position.y -= test::velocity.y * m_DeltaTime;
			//m_Rect->m_Position.x -= test::velocity.x * m_DeltaTime;
			//if (m_Rect->m_Position.y < -1.0f || m_Rect->m_Position.y > 1.0f)
			//{
			//	test::velocity.y = -test::velocity.y;
			//}
			//if (m_Rect->m_Position.x < -aspectRatio || m_Rect->m_Position.x > aspectRatio)
			//{
			//	test::velocity.x = -test::velocity.x;
			//}

			//Novaura::Renderer::DrawRotatedRectangle(*m_Rect);


			//EndFrame();
	}

	void Application::SetCallBackFunctions()
	{
	    GLFWCallbackWrapper::SetApplication(this);
	    glfwSetCursorPosCallback(GetWindow()->Window, GLFWCallbackWrapper::MousePositionCallBack);
	    glfwSetScrollCallback(GetWindow()->Window, GLFWCallbackWrapper::MouseScrollCallBack);
	    glfwSetFramebufferSizeCallback(GetWindow()->Window, GLFWCallbackWrapper::WindowResizeCallBack);
		glfwSetKeyCallback(GetWindow()->Window, GLFWCallbackWrapper::KeyBoardCallback);
		glfwSetMouseButtonCallback(GetWindow()->Window, GLFWCallbackWrapper::MouseButtonCallback);
	}
}