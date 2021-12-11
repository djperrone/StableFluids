#include "sapch.h"
#include "Novaura/Core/Application.h"
#include "App/SimulationApp.h"

int main()
{
	Novaura::Application* app = new Simulation::SimulationApp("Stable Fluids", 1280.0f, 720.0f);
	
	while (app->IsRunning())
	{
		app->Update();
	}
	app->GetStateMachine()->ShutDown();
	delete app;
	return 0;
}