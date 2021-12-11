#include "sapch.h"
#include "Novaura/Core/Application.h"
#include "ParticleSimulation/ParticleSimulationApp.h"

int main()
{
	Novaura::Application* app = new ParticleSimulation::ParticleSimulationApp("Particle Simulation2D", 1280.0f, 720.0f);
	
	while (app->IsRunning())
	{
		app->Update();
	}
	app->GetStateMachine()->ShutDown();
	delete app;
	return 0;
}