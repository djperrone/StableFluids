#pragma once
#include "Actor.h"
#include <glm/glm.hpp>

namespace Simulation {

	enum class ButtonType
	{
		None = 0,
		Play,
		Resume,
		PauseMenu,
		MainMenu,
		Exit
	};

	class Button : public Actor
	{
	public:

		Button() = default;
		Button(std::string_view fileName, ButtonType type, const glm::vec2& position, const glm::vec2& scale);	

		virtual void Update(float dt) override;

		inline bool IsHovered() { return m_IsHovered; }
		inline void SetHovered(bool isHovered) {m_IsHovered = isHovered; }
		inline bool WasHovered() { return m_WasHovered; }
		inline void SetWasHovered(bool wasHovered) { m_WasHovered = wasHovered; }

		inline ButtonType GetButtonType() { return m_ButtonType; }
		const inline ButtonType GetButtonType() const { return m_ButtonType; }


	private:

		bool m_IsHovered = false;
		bool m_WasHovered = false;
		ButtonType m_ButtonType;
	};
}