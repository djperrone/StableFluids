#pragma once
#include <GLFW/glfw3.h>
//#include <queue>
//#include "Command.h"
#include "InputController.h"
#include "Novaura/Core/Window.h"

#include <glm/glm.hpp>

namespace Novaura {	
	class Rectangle;
	using EventType = int;
	using KeyCode = int;

	struct MousePosition
	{
		double x, y;
	};

	struct Event
	{
		EventType Type;
		KeyCode Key;
	};

	class InputHandler
	{
	public:
		
		static void Init();	
		static bool IsPressed(GLFWwindow* window, int keyCode);
		static bool IsPressed(int keyCode);

		static std::shared_ptr<InputController> CreateNewInputController();

		static void SetCurrentWindow(std::shared_ptr<Window> window) { s_CurrentWindow = window; }
		static std::shared_ptr<Window> GetCurrentWindow() { return s_CurrentWindow; }

		static void SetCurrentController(std::shared_ptr<InputController> controller) { s_InputController = controller; }
		static InputController& GetCurrentController() { return *s_InputController; }
		
		static MousePosition GetMousePosition();
		static MousePosition GetMouseDeviceCoordinates();
		static bool IsRectHovered(const Rectangle& rectangle);

		//static std::queue<Event> EventQueue;
	private:
		static std::shared_ptr<InputController> s_InputController;
		static std::shared_ptr<Window>s_CurrentWindow;
	private:
		InputHandler() = default;
		InputHandler(const InputHandler&) = delete;		
		InputHandler(std::shared_ptr<InputController> controller);
		
	};
}
