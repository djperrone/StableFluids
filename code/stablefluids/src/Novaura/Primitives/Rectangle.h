#pragma once
#include <glm/glm.hpp>

namespace Novaura {	

	struct Bounds
	{
		glm::vec2 BottomLeft, BottomRight, TopLeft, TopRight;
	};
	class Rectangle
	{
	public:
		Rectangle() = default;
		//Rectangle(const glm::vec2& position, const glm::vec2& size, const glm::vec4& color, const glm::vec2& scale = glm::vec2(1.0f, 1.0f));
		Rectangle(const glm::vec2& position, const glm::vec2& scale, const glm::vec4& color);
		Rectangle(const glm::vec2& position, const glm::vec2& scale);
		Rectangle(const glm::vec2& position, const glm::vec2& scale, float angle);
		Rectangle(const glm::vec2& position);
		~Rectangle() = default;	

		Bounds GetBounds() const;


	public:
		inline glm::vec3& GetPosition()  { return m_Position; }
		inline glm::vec3& GetScale()		{ return m_Scale; }
		inline glm::vec4& GetColor()		{ return m_Color; }
		inline float GetRotation()	const	{ return m_Rotation; }

		inline const glm::vec3& GetPosition() const { return m_Position; }
		inline const glm::vec3& GetScale()    const { return m_Scale; }
		inline const glm::vec4& GetColor()    const { return m_Color; }
		//inline const float GetRotation()     const { return m_Rotation; }


		inline void SetPosition(const glm::vec3& position) { m_Position = position; }
		inline void SetScale(const glm::vec3& scale) { m_Scale = scale; }
		inline void SetColor(const glm::vec4& color) {  m_Color = color; }
		inline void SetRotation(float rotation) { m_Rotation = rotation; }

	private:
		glm::vec3 m_Position = glm::vec3(0.0f,0.0f,0.0f);	
		glm::vec3 m_Scale = glm::vec3(1.0f,1.0f,1.0f);
		glm::vec4 m_Color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);;
		float m_Rotation = 0.0f;
	};
}