#pragma once
#include "Novaura/StateMachine/State.h"
#include "Novaura/Camera/CameraController.h"
//#include "Novaura/Primitives/Rectangle.h"
#include "StateInfo.h"


#include "../gui.h"

#include "FinalCode/fluid.h"
#include "FinalCode/utilities.h"

namespace Simulation {


	class SerialCPU_final : public Novaura::State
	{
	public:
		SerialCPU_final();		
		SerialCPU_final(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine);
		
		virtual void OnEnter() override;

		virtual void HandleInput() override;
		virtual void Update(float deltaTime)override;
		virtual void Draw(float deltaTime) override;

		virtual void OnExit() override;

		virtual void Pause() override;
		virtual void Resume() override;

	
	private:		

		double m_CurrentTime = 0.0;
		double m_PreviousTime = 0.0;
		float particleScale = 0.03f;

		StateInfo m_StateInfo;


		
		//int n;
		double simulation_time;
		int navg, nabsavg = 0;
		double davg, dmin, absmin = 1.0, absavg = 0.0;

		std::unique_ptr<Pgui::Gui> m_Gui;

		// final
		StableFluids::FluidSquare* sq;

		int n = 2000;
		float d = 0;
		float v = .00001;
		float dt = .005;
		int n_per_side;

	};
}
