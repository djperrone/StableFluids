#pragma once
#include <GLFW/glfw3.h>
#include "Novaura/Core/Window.h"
#include "States/StateInfo.h"


namespace Pgui {

	class Gui
	{
	public:
		//Gui() = default;
		Gui(std::shared_ptr<Novaura::Window>& window);
		
		~Gui();

		void Draw();		
		void DrawStateButtons(Simulation::StateInfo& stateInfo, float& pscale);
		void DrawDockSpace(Simulation::StateInfo& stateInfo);

		void BeginFrame();
		void EndFrame();

	private:
		//void DrawBackPackUI(std::vector<std::unique_ptr<nova::Model>>& actors);
		std::vector<bool> m_ActiveSlots;
		bool test = false;
		bool m_Changed = false;

	private:
		bool rotationReset = false;
		bool translationReset = false;


	private:
		std::shared_ptr<Novaura::Window> m_Window;

	private:
		bool show_demo_window = true;
		bool show_another_window = false;
		//ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	};
}