#include "sapch.h"
#include "StableFluidsGPU_test.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "Novaura/Random.h"

namespace Simulation {


	StableFluidsGPU_test::StableFluidsGPU_test()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}

	StableFluidsGPU_test::StableFluidsGPU_test(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void StableFluidsGPU_test::OnEnter()
	{
		n_per_side = (int)sqrt(n);

		StableFluidsCuda::FluidSquareCreate(&sq, n_per_side, d, v, dt);		

		StableFluidsCuda::FluidSquareCreate_cpu(&sq_cpu, n_per_side, d, v, dt);

		StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, 50);
		StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, 3, 3);

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void StableFluidsGPU_test::HandleInput()
	{
	}

	void StableFluidsGPU_test::Update(float deltaTime)
	{
		if (m_StateInfo.RESET)
		{			
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			FluidSquareStep(&sq);
			float addDensity = 5.0f;
			float addVelocityx = glm::sin(glfwGetTime()) * Novaura::Random::Float(-0.2f, 0.2f);
			float addVelocityy = -glm::sin(glfwGetTime());// *Novaura::Random::Float(-0.2f, 0.2f);			
			StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, addDensity);
			StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, addVelocityx, addVelocityy);
			StableFluidsCuda::CopyToCPU(&sq_cpu, &sq, sq_cpu.data.size);
		}
	}

	void StableFluidsGPU_test::Draw(float deltaTime)
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
				float d = sq_cpu.density[IX(i, j)];



				Novaura::BatchRenderer::DrawRectangle(glm::vec3(x, y, 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.1f, 0.1f, glm::clamp(d, 0.0f, 1.0f)), glm::vec2(1.0f, 1.0f));

				//Novaura::BatchRenderer::DrawCircle(glm::vec3(sq->Vx[i + j * n_per_side], sq->Vy[i + j * n_per_side], 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
				//Novaura::BatchRenderer::DrawRectangle(glm::vec3(x, y, 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(glm::clamp(d, 0.2f, 1.0f), 0.1f, 0.1f, glm::clamp(d, 0.2f, 1.0f)), glm::vec2(1.0f, 1.0f));
			}
			// cool
			//float addDensity = Novaura::Random::Float(0.0f, 25.f);
			//float addVelocityx = glm::sin(glfwGetTime()) * Novaura::Random::Float(-0.2f, 0.2f);
			//float addVelocityy = glm::sin(glfwGetTime());// *Novaura::Random::Float(-0.2f, 0.2f);
		}

	

		Novaura::BatchRenderer::EndScene();
		m_Gui->DrawStateButtons(m_StateInfo, particleScale);

		m_Gui->EndFrame();
	}



	void StableFluidsGPU_test::OnExit()
	{
		StableFluidsCuda::FluidSquareFree(&sq);
	}

	void StableFluidsGPU_test::Pause()
	{

	}

	void StableFluidsGPU_test::Resume()
	{

	}

}