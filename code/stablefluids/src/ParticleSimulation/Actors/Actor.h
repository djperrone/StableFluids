#pragma once

#include "Novaura/Primitives/Rectangle.h"

namespace ParticleSimulation {

	struct Bounds
	{
		glm::vec2 BottomLeft, BottomRight, TopLeft, TopRight;
	};

	class Actor
	{
	public:
		virtual void Update(float dt) = 0;

		inline const Novaura::Rectangle& GetRectangle() const { return *m_Rect; }
		inline Novaura::Rectangle& GetRectangle() { return *m_Rect; }

		inline void SetTextureFile(std::string_view texFile) { m_TextureFile = texFile; }
		inline std::string_view GetTextureFile() { return m_TextureFile; }
		inline const std::string_view GetTextureFile() const { return m_TextureFile; }

		Bounds GetBounds() const;

	protected:
		std::unique_ptr<Novaura::Rectangle> m_Rect;
		std::string_view m_TextureFile;
	};
}