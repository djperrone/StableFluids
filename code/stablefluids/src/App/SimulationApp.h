#pragma once
#include "Novaura/Core/Application.h"

namespace Simulation {

	class SimulationApp : public Novaura::Application
	{
	public:
		SimulationApp();
		SimulationApp(std::string_view title, float width, float height);
		
	private:


	};
}