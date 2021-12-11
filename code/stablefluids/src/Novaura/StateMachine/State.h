#pragma once

#include "Novaura/Input/InputController.h"
#include "Novaura/Core/Window.h"
#include "Novaura/Camera/CameraController.h"
#include "StateMachine.h"

namespace Novaura {

	class State
	{
	public:
		State() = default;		

		virtual void HandleInput() = 0;
		virtual void Update(float deltaTime) = 0;
		virtual void Draw(float deltaTime) = 0;

		virtual void OnEnter() = 0;
		virtual void OnExit() = 0;

		virtual void Pause() = 0;
		virtual void Resume() = 0;

		InputController& GetInputController(){return *m_InputController;}
		const InputController& GetInputController() const { return *m_InputController; }

		void SetInputController(std::shared_ptr<InputController> inputController) { m_InputController = inputController; }

		CameraController& GetCameraController() { return *m_CameraController; }
		const CameraController& GetCameraController() const { return *m_CameraController; }

		StateMachine& GetStateMachine() { return *m_StateMachine; }
		const StateMachine& GetStateMachine() const { return *m_StateMachine; }

		void SetCameraController(std::shared_ptr<CameraController> cameraController) { m_CameraController = cameraController; }

		Novaura::Window& GetWindow() { return *m_Window; }
		const Novaura::Window& GetWindow() const { return *m_Window; }

		bool IsOverLay() { return m_IsOverlay; }
		bool SetIsOverLay(bool isOverlay) { m_IsOverlay = isOverlay; }		

		inline bool IsPaused() { return m_IsPaused; }

	protected:
		std::shared_ptr<InputController> m_InputController;
		std::shared_ptr<CameraController> m_CameraController;
		std::shared_ptr<Window> m_Window;
		std::shared_ptr<StateMachine> m_StateMachine;
		bool m_IsOverlay = false;
		bool m_IsPaused = false;
		
	};
}