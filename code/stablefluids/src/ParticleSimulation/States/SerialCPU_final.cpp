#include "sapch.h"
#include "SerialCPU_final.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"
#include "Novaura/Random.h"

namespace ParticleSimulation {
	

	SerialCPU_final::SerialCPU_final()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	SerialCPU_final::SerialCPU_final(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
		: m_StateInfo()
	{
		m_Window = window;
		m_CameraController = cameraController;
		m_StateMachine = stateMachine;
		m_Gui = std::make_unique<Pgui::Gui>(Novaura::InputHandler::GetCurrentWindow());
		m_InputController = Novaura::InputHandler::CreateNewInputController();
		Novaura::InputHandler::SetCurrentController(m_InputController);
		OnEnter();
	}

	void SerialCPU_final::OnEnter()
	{
				
		n_per_side = (int)sqrt(n);
		
		sq = StableFluids::FluidSquareCreate(n_per_side, d, v, dt);

		StableFluids::FluidSquareAddDensity(sq, n_per_side / 2, n_per_side / 2, 50);
		StableFluids::FluidSquareAddVelocity(sq, n_per_side / 2, n_per_side / 2, 3, 3);		
	

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void SerialCPU_final::HandleInput()
	{
	}

	void SerialCPU_final::Update(float deltaTime)
	{		
		if (m_StateInfo.RESET)
		{			
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
		

			FluidSquareStep(sq);
			
			float addDensity = 5.0f;
			float addVelocityx = glm::sin(glfwGetTime()) * Novaura::Random::Float(-0.2f, 0.2f);
			float addVelocityy = -glm::sin(glfwGetTime());
			StableFluids::FluidSquareAddDensity(sq, n_per_side / 2, n_per_side / 2, addDensity);
			StableFluids::FluidSquareAddVelocity(sq, n_per_side / 2, n_per_side / 2, addVelocityx, addVelocityy);
			
		}
	}

	void SerialCPU_final::Draw(float deltaTime)
	{
		
		Novaura::BatchRenderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::BatchRenderer::Clear();
		Novaura::BatchRenderer::BeginScene(m_CameraController->GetCamera());	
		m_Gui->BeginFrame();

		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;		
	
		for (int i = 0; i < n_per_side; i++)
		{
			for (int j = 0; j < n_per_side; j++)
			{
				int scale = n / 150;
				float x = i * scale / width;
				float y = j * scale / height;
				int N = n_per_side;
				float d = sq->density[IX(i, j)];			
			
				Novaura::BatchRenderer::DrawRectangle(glm::vec3(x, y, 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f,0.1f, 0.1f, glm::clamp(d,0.0f,1.0f )), glm::vec2(1.0f, 1.0f));
			}		
		}		
	
		Novaura::BatchRenderer::EndScene();
		m_Gui->DrawStateButtons(m_StateInfo, particleScale);		
		m_Gui->EndFrame();
	}
	

	void SerialCPU_final::OnExit()
	{
		
		StableFluids::FluidSquareFree(sq);
	}

	void SerialCPU_final::Pause()
	{
		
		
	}

	void SerialCPU_final::Resume()
	{
		
	}

}
