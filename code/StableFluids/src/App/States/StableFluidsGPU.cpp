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

		StableFluidsCuda::FluidSquareAddDensity(&sq, m_AddPos.x, m_AddPos.y, m_Add.z);
		StableFluidsCuda::FluidSquareAddVelocity(&sq, m_AddPos.x, m_AddPos.y, m_Add.x, m_Add.y);

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
				float scale = n / 50;
				float x = i * spacing / width;
				float y = j * spacing / width;
				
				int N = n_per_side;
				m_Locations[i + j * n_per_side] = { x - 1.5f,y - 1.5f,0 };				
			}
		}

		cudaMemcpy(m_Locations_gpu, (void*)m_Locations, sizeof(CudaMath::Vector3f) * n_per_side * n_per_side, cudaMemcpyHostToDevice);

		free(m_Locations);

		Novaura::Renderer::InitInstancedSquares(n_per_side * n_per_side, squareScale, m_Locations_gpu, sq.density, backgroundColor, colorMask);		
		cudaFree(m_Locations_gpu);
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
			
			if(addForce)
			{						
				StableFluidsCuda::FluidSquareAddDensity(&sq, m_AddPos.x, m_AddPos.y, m_Add.z);
				StableFluidsCuda::FluidSquareAddVelocity(&sq,m_AddPos.x, m_AddPos.y, m_Add.x, m_Add.y);
			}							
		}
	}

	void StableFluidsGPU::Draw(float deltaTime)
	{
		Novaura::Renderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::Renderer::Clear();
		Novaura::Renderer::BeginSceneInstanced(m_CameraController->GetCamera());
		m_Gui->BeginFrame();		

		Novaura::Renderer::EndInstancedSquares();
		
		m_Gui->DrawStateButtons(m_StateInfo,sq.data, n_per_side, squareScale, spacing,m_AddPos, m_Add, backgroundColor, colorMask, addForce);

		m_Gui->EndFrame();
	}

	void StableFluidsGPU::OnExit()
	{		
		StableFluidsCuda::FluidSquareFree(&sq);			
		Novaura::Renderer::ShutDownInstancedSquares();
	}

	void StableFluidsGPU::Pause()
	{

	}

	void StableFluidsGPU::Resume()
	{

	}

	

}