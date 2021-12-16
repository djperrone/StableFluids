#include "sapch.h"
#include "StableFluidsGPU_benchmark.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "Novaura/Random.h"

namespace Simulation {

	StableFluidsGPU_benchmark::StableFluidsGPU_benchmark()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}

	StableFluidsGPU_benchmark::StableFluidsGPU_benchmark(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void StableFluidsGPU_benchmark::OnEnter()
	{
		spdlog::info(__FUNCTION__);
		spacing = n_per_side * n_per_side / 50;

		StableFluidsCuda::FluidSquareCreate(&sq, n_per_side, d, v, dt);
		

		sq_test = StableFluids::FluidSquareCreate(n_per_side, d, v, dt);
		StableFluids::FluidSquareAddDensity(sq_test, n_per_side / 2, n_per_side / 2, 50);
		StableFluids::FluidSquareAddVelocity(sq_test, n_per_side / 2, n_per_side / 2, 3, 3);

		StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, 50);
		StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, 3, 3);

	
		//CompareResults();
		m_PreviousTime = glfwGetTime();
		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;

		timer.SetSquareSize(n_per_side);
		std::string test = "func_results_";
		test.append(std::to_string(n_per_side));
		test.append("_");
		test.append(std::to_string(glfwGetTime()));
		spdlog::info(test);
		timer.SetOutFile(test);
		//cudaTimer.SetOutFile(test);
		timer.WriteHeader();
		//cudaTimer.OpenAppend();
		//cudaTimer.WriteHeader();
	}

	void StableFluidsGPU_benchmark::HandleInput()
	{
	}

	void StableFluidsGPU_benchmark::Update(float deltaTime)
	{
		if (m_StateInfo.RESET)
		{			
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			//timer.SetFunctionName("sqstep");
			//timer.Start();
			//timer.WriteSeparator("CPU");
			timer.CPU = false;
			//timer.BeginTimeFunction("cpu_step");

			StableFluids::FluidSquareStep(sq_test);
			//StableFluids::FluidSquareStep(sq_test, timer);
			//timer.EndTimeFunction();
			//timer.WriteSeparator("GPU");

			//StableFluidsCuda::FluidSquareStep(&sq, timer);
			//timer.Flush();
			//timer.WriteCSV();
			timer.GPU = false;
			//timer.BeginTimeFunction("gpu_step vx");
			StableFluidsCuda::FluidSquareStep(&sq, timer);
			//StableFluidsCuda::FluidSquareStep(&sq);
			//timer.EndTimeFunction();

			double currentTime = glfwGetTime();
			//if (currentTime - m_PreviousTime >0.05)
			{
				m_PreviousTime = currentTime;
				float addDensity = 5.0f;
				float addVelocityx = (10 + glm::sin(glfwGetTime() * 2.0f)) * 0.5f;// *Novaura::Random::Float(-0.2f, 0.2f);
				float addVelocityy = glm::sin(glfwGetTime() / 2.0f) * 0.5f;
								
				StableFluids::FluidSquareAddDensity(sq_test, n_per_side / 2, n_per_side / 2, addDensity);
				StableFluids::FluidSquareAddVelocity(sq_test, n_per_side / 2, n_per_side / 2, addVelocityx, addVelocityy);
				StableFluidsCuda::FluidSquareAddDensity(&sq, n_per_side / 2, n_per_side / 2, addDensity);
				StableFluidsCuda::FluidSquareAddVelocity(&sq, n_per_side / 2, n_per_side / 2, addVelocityx, addVelocityy);
			}			
			
			
		
			counter++;
	
		}
	}

	void StableFluidsGPU_benchmark::Draw(float deltaTime)
	{
		
	}



	void StableFluidsGPU_benchmark::OnExit()
	{
		timer.Flush();		
		StableFluidsCuda::FluidSquareFree(&sq);		
		StableFluids::FluidSquareFree(sq_test);
	}

	void StableFluidsGPU_benchmark::Pause()
	{

	}

	void StableFluidsGPU_benchmark::Resume()
	{

	}

	void StableFluidsGPU_benchmark::DrawCPU()
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
				float scale = n / 100;
				//spdlog::info("i j, {}, {}", i, j);
				float x = i * spacing / width;
				float y = j * spacing / width;
				int N = n_per_side;
				float d = sq_test->density[IX(i, j)];

				float color = 1.0f - (d > 1.0f ? 1.0f : d);
				Novaura::BatchRenderer::DrawRectangle(glm::vec3(x - 1.8, y, 0.0f), glm::vec3(squareScale, squareScale, 0), glm::vec4(color, color, color, 1.0f), glm::vec2(1.0f, 1.0f));
			}

		}

		Novaura::BatchRenderer::EndScene();
		// spacing, fluiddata, color, vx, vy, color channel bool
		m_Gui->DrawStateButtons(m_StateInfo, sq.data, n_per_side, squareScale, spacing);

		m_Gui->EndFrame();
	}

	void StableFluidsGPU_benchmark::DrawGPU()
	{
		Novaura::Renderer::UpdateInstancedColors(backgroundColor, colorMask, sq.density, n_per_side * n_per_side);		
		StableFluidsCuda::FluidSquareAddDensity(&sq, m_AddPos.x, m_AddPos.y, m_Add.z);
		StableFluidsCuda::FluidSquareAddVelocity(&sq, m_AddPos.x, m_AddPos.y, m_Add.x, m_Add.y);
		
	}


}