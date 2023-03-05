#include "sapch.h"
#include "TestState.h"

#include <glm/glm.hpp>
#include "Novaura/Renderer/Renderer.h"

#include <spdlog/spdlog.h>

#include "Novaura/Renderer/Texture.h"
#include "Novaura/Renderer/TextureLoader.h"
#include "Novaura/Input/InputHandler.h"

namespace Simulation {	

	TestState::TestState(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
	{
		m_Window = window;
		m_CameraController = cameraController;
		m_StateMachine = stateMachine;
		OnEnter();
	}

	void TestState::OnEnter()
	{

		Novaura::Renderer::Init();

		m_Rect = std::make_unique<Novaura::Rectangle>(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 0.5f), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		Novaura::InputHandler::GetCurrentController().BindAxisInputEvent(GLFW_KEY_T, []() {spdlog::info("test axis event T"); });
		//Novaura::Texture tex = Novaura::TextureLoader::LoadTexture("Assets/Textures/Spaceship.png");
		/*Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_MOUSE_BUTTON_1, [m_Rect = &m_Rect, m_Window = &m_Window]()
			{
				if (Novaura::InputHandler::IsRectHovered(*m_Rect->get()))
				{
					spdlog::info("collision deteced!");
				}

			});*/


	}


	void TestState::HandleInput()
	{
	}
	namespace testState {
		float velocity = 0.01f;
	}
	void TestState::Update(float deltaTime)
	{
		m_CameraController->Update(GetWindow(), deltaTime);		
		float aspectRatio = m_Window->Width / m_Window->Height;
		
		m_Rect->GetPosition().x += testState::velocity;
		if (m_Rect->GetPosition().x > aspectRatio)
		{
			testState::velocity *= -1;
		}

		if (m_Rect->GetPosition().x < -aspectRatio)
		{
			testState::velocity *= -1;
		}


		//Draw(deltaTime);		

	}

	void TestState::Draw(float deltaTime)
	{
		Novaura::Renderer::SetClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		Novaura::Renderer::Clear();
		m_Rect->SetRotation(45.0f);
		Novaura::Renderer::BeginScene(m_CameraController->GetCamera());
		//Novaura::Renderer::DrawRotatedRectangle(*m_Rect, "Assets/Textures/Spaceship.png");		
		Novaura::Renderer::DrawRectangle(glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.5f, 0.8f, 0.8f, 1.0f));

		Novaura::Renderer::DrawRectangle(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.2f, 0.8f, 0.2f, 1.0f), "Assets/Textures/UFO.png");
		Novaura::Renderer::DrawRectangle(glm::vec3(-0.50f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.2f), glm::vec4(0.5f, 0.5f, 0.8f, 1.0f), "Assets/Textures/Spaceship.png");
		Novaura::Renderer::DrawRectangle(glm::vec3(0.75f, 0.75f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.7f, 0.3f, 0.5f, 1.0f));
		Novaura::Renderer::DrawRectangle(glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.2f), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), "Assets/Textures/UFO.png");

		//Novaura::Renderer::EndScene();
	}

	

	void TestState::OnExit()
	{
	}

	void TestState::Pause()
	{
	}

	void TestState::Resume()
	{
	}
}
