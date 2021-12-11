#pragma once
#include "Novaura/Core/OpenGLContext.h"
#include "Novaura/Primitives/Rectangle.h"
#include "Novaura/Renderer/Shader.h"
#include "Novaura/Camera/Camera.h"
#include "Novaura/Camera/CameraController.h"

#include "Novaura/Input/InputController.h"

#include "Novaura/StateMachine/StateMachine.h"

namespace Novaura {

	class Application
	{
	public:
		Application();
		Application(std::string_view title, float width, float height);
		~Application();

		void Update();
		void BeginFrame();
		void EndFrame();
		const std::shared_ptr<Window> GetWindow() const { return (m_Context.GetWindow()); }
		std::shared_ptr<Window> GetWindow() { return (m_Context.GetWindow()); }
		inline bool IsRunning() const { return m_Context.IsRunning(); }

	protected:
		static std::shared_ptr <StateMachine> m_StateMachine;

	private:
		void ScreenSaver();
	public:
		void SetCallBackFunctions();

		void WindowResizeCallBack(int width, int height);
		void KeyboardCallback(int key, int scancode, int action, int mods);
		
		void MousePositionCallBack(double positionX, double positionY);
		void MouseButtonCallBack(int button, int action, int mods);
		void MouseScrollCallBack(double xoffset, double yoffset);

		static std::shared_ptr<CameraController> GetCameraController() { return m_CameraController; }
		static std::shared_ptr<StateMachine> GetStateMachine() { return m_StateMachine; }

	private:
		OpenGLContext m_Context;
		
	public:
		static std::shared_ptr<CameraController> m_CameraController;	

	private:
		float m_DeltaTime = 0.0f;
		float m_LastFrame = 0.0f;

	};
}