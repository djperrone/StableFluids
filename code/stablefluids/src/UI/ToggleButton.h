#pragma once
#include "Button.h"

namespace UI {

	class ToggleButton : public Button 
	{
	public:
		ToggleButton() = default;
		//ToggleButton(std::string_view texFile, const Novaura::Rectangle& rect, Novaura::Command&& command);
		ToggleButton(std::string_view texFile, glm::vec3&& pos, glm::vec3&& scale, Novaura::Command&& command);
		ToggleButton(std::string_view texFile, const glm::vec3& pos, const glm::vec3& scale, Novaura::Command&& command);
	
		virtual void Update() override;
		virtual void Execute() override;

	private:


	};
}