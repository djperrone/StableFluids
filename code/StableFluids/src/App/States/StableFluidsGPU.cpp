#include "sapch.h"
#include "StableFluidsGPU.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "Novaura/Random.h"

namespace Simulation {


	StableFluidsGPU::StableFluidsGPU()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}

	StableFluidsGPU::StableFluidsGPU(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void StableFluidsGPU::OnEnter()
	{
		spdlog::info(__FUNCTION__);

		StableFluidsCuda::FluidSquareCreate(&sq, n_per_side, d, v, dt);

		StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, 50);
		StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, 3, 3);

		spacing = n_per_side * n_per_side / 100;

		
		m_PreviousTime = glfwGetTime();
		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;


		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;
		m_Locations = (CudaMath::Vector3f*)malloc(sizeof(CudaMath::Vector3f) * n_per_side * n_per_side);
		cudaMalloc((void**)&m_Locations_gpu, sizeof(CudaMath::Vector3f) * n_per_side * n_per_side);

		for (int i = 0; i < n_per_side; i++)
		{
			for (int j = 0; j < n_per_side; j++)
			{
				//spdlog::info("i j, {}, {}", i, j);
				//spacing = scale;
				float scale = n / 50;
				float x = i * scale / width;
				float y = j * scale / width;
				
				int N = n_per_side;
				m_Locations[i + j * n_per_side] = { x,y,0 };
				//spdlog::info("x: {} y: {}", m_Locations[i].x, m_Locations[i].y);
				//float color = 1.0f - (d > 1.0f ? 1.0f : d);
				//Novaura::BatchRenderer::DrawRectangle(glm::vec3(x, y, 0.0f), glm::vec3(squareScale, squareScale, 0), glm::vec4(color, color, color, 1.0f), glm::vec2(1.0f, 1.0f));

			}
		}

		cudaMemcpy(m_Locations_gpu, (void*)m_Locations, sizeof(CudaMath::Vector3f) * n_per_side * n_per_side, cudaMemcpyHostToDevice);

		free(m_Locations);

		Novaura::Renderer::InitInstancedSquares(n_per_side * n_per_side, squareScale, m_Locations_gpu, sq.density, backgroundColor, colorMask);
		//Novaura::Renderer::UpdateLocationMatrices(m_Locations_gpu, squareScale, n_per_side * n_per_side);
		
	}

	void StableFluidsGPU::HandleInput()
	{
	}

	void StableFluidsGPU::Update(float deltaTime)
	{
		if (m_StateInfo.RESET)
		{			
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			
			
		
		
		
			StableFluidsCuda::FluidSquareStep(&sq);

			Novaura::Renderer::UpdateInstancedColors(backgroundColor, colorMask, sq.density, n_per_side * n_per_side);
			

			double currentTime = glfwGetTime();
			//if (currentTime - m_PreviousTime >0.05)
			{
				m_PreviousTime = currentTime;
				float addDensity = 5.0f;
				float addVelocityx = (10 + glm::sin(glfwGetTime() * 2.0f)) * 0.5f;// *Novaura::Random::Float(-0.2f, 0.2f);
				float addVelocityy = glm::sin(glfwGetTime() / 2.0f) * 0.5f;
								
				
				StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, addDensity);
				StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, addVelocityx, addVelocityy);	
			}					
		
		}
	}

	void StableFluidsGPU::Draw(float deltaTime)
	{
		Novaura::Renderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::Renderer::Clear();
		Novaura::Renderer::BeginSceneInstanced(m_CameraController->GetCamera());
		m_Gui->BeginFrame();

		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;
		
	
	
	

		Novaura::Renderer::EndInstancedSquares();
		// spacing, fluiddata, color, vx, vy, color channel bool
		m_Gui->DrawStateButtons(m_StateInfo,sq.data, n_per_side, squareScale, spacing, backgroundColor, colorMask);

		m_Gui->EndFrame();
	}



	void StableFluidsGPU::OnExit()
	{		
		StableFluidsCuda::FluidSquareFree(&sq);	
		cudaFree(m_Locations_gpu);
		Novaura::Renderer::ShutDownInstancedSquares();
	}

	void StableFluidsGPU::Pause()
	{

	}

	void StableFluidsGPU::Resume()
	{

	}

	

}