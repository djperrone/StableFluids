#include "sapch.h"
#include "SimulationApp.h"



#include "States/SerialCPU_final.h"
#include "States/StableFluidsGPU_test.h"
#include "States/StableFluidsGPU.h"


namespace Simulation {
	SimulationApp::SimulationApp()
	{
		
	}
	SimulationApp::SimulationApp(std::string_view title, float width, float height)
		:Application(title, width, height)
	{	
	
	
		//m_StateMachine->PushState(std::make_unique<SerialCPU_final>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<StableFluidsGPU_test>(GetWindow(), m_CameraController, m_StateMachine));
		m_StateMachine->PushState(std::make_unique<StableFluidsGPU>(GetWindow(), m_CameraController, m_StateMachine));
	
	}
}