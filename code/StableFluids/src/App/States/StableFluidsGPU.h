#pragma once
#include "Novaura/StateMachine/State.h"
#include "Novaura/Camera/CameraController.h"
//#include "Novaura/Primitives/Rectangle.h"
#include "StateInfo.h"
#include "../gui.h"
#include "FinalCode/fluid.h"
#include "FinalCode/utilities.h"

#include "CudaSrc/Fluid.cuh"

#include "Benchmark/timer.h"
#include "Benchmark/CudaTimer.cuh"

#include "CudaSrc/CudaMath.cuh"

namespace Simulation {

	class StableFluidsGPU : public Novaura::State
	{
	public:
		StableFluidsGPU();
		StableFluidsGPU(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine);

		virtual void OnEnter() override;

		virtual void HandleInput() override;
		virtual void Update(float deltaTime)override;
		virtual void Draw(float deltaTime) override;

		virtual void OnExit() override;

		virtual void Pause() override;
		virtual void Resume() override;


	private:
		//long m_PreviousTime;
		double m_CurrentTime = 0.0;
		double m_PreviousTime = 0.0;
		float particleScale = 0.08f;
		float spacing;
		float squareScale = 0.08f;

		CudaMath::Vector4f colorMask = { 1.0f,1.0f,1.0f,1.0f };
		CudaMath::Vector4f backgroundColor = {1.0f,1.0f,1.0f,1.0f};

		StateInfo m_StateInfo;
		int counter = 0;

		std::unique_ptr<Pgui::Gui> m_Gui;

		// final
		StableFluidsCuda::FluidSquare sq;
		


		int n = 5000;
		float d = 0;
		float v = .00001;
		float dt = .005;
		int n_per_side = sqrt(n);

		CudaMath::Vector3f* m_Locations;
		CudaMath::Vector3f* m_Locations_gpu;
		
	};
}