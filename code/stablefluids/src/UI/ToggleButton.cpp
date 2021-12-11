#include "sapch.h"
#include "ToggleButton.h"
#include "Novaura/Input/InputHandler.h"

namespace UI {

	ToggleButton::ToggleButton(std::string_view texFile, glm::vec3&& pos, glm::vec3&& scale, Novaura::Command&& command)		
	{
		m_TextureFile = texFile;
		m_Rect = std::make_unique<Novaura::Rectangle>(pos, scale);
		m_Command = std::move(command);
	}
	ToggleButton::ToggleButton(std::string_view texFile, const glm::vec3& pos, const glm::vec3& scale, Novaura::Command&& command)
	{
		m_TextureFile = texFile;
		m_Rect = std::make_unique<Novaura::Rectangle>(pos, scale);
		m_Command = std::move(command);
	}

	
	void ToggleButton::Update()
	{
		const bool state = Novaura::InputHandler::IsRectHovered(*m_Rect);
		if (state && !m_IsHovered)
		{
			m_Rect->SetColor(glm::vec4(1.25f, 1.25f, 1.25f, 1.0f));
			m_IsHovered = true;
		}
		else if (!state && m_IsHovered)
		{
			m_Rect->SetColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
			m_IsHovered = false;
		}
	}
	void ToggleButton::Execute()
	{
		m_Command.Execute();
	}
}