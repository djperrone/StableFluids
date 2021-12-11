#pragma once

#include "Novaura/Input/Command.h"
#include "Novaura/Primitives/Rectangle.h"
#include <glm/glm.hpp>
#include <string_view>

namespace UI {

	class Button
	{
	public:
		Button() = default;		
		virtual ~Button() = default;

		virtual void Update() = 0;
		virtual void Execute() = 0;

		inline bool IsHovered() { return m_IsHovered; }
		inline void SetHovered(bool isHovered) { m_IsHovered = isHovered; }

		inline const Novaura::Rectangle& GetRectangle() const { return *m_Rect; }
		inline Novaura::Rectangle& GetRectangle() { return *m_Rect; }

		inline std::string_view GetTextureFile() { return m_TextureFile; }
		inline const std::string_view GetTextureFile() const { return m_TextureFile; }

	protected:
		std::string_view m_TextureFile;
		bool m_IsHovered = false;
		std::unique_ptr<Novaura::Rectangle> m_Rect;
		Novaura::Command m_Command;
	};	
}