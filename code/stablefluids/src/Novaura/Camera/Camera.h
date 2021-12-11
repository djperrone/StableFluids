#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Novaura {

	class Camera
	{
	public:
		Camera(float width, float height);

		const glm::mat4& GetViewMatrix() const { return m_ViewMatrix; }
		const glm::mat4& GetProjectionMatrix() const { return m_ProjectionMatrix; }
		const glm::mat4& GetViewProjectionMatrix() const { return m_ViewProjectionMatrix; }

		void SetProjectionMatrix(float width, float height);
		void SetProjectionMatrix(float left, float right, float bottom, float top);
		void SetViewMatrix(const glm::vec3& position, float rotation);
		void SetViewMatrix(const glm::vec3& position);
		void SetRotation(float rotation) { m_Rotation = rotation; CalcViewMatrix();
		}
		void SetPosition(const glm::vec3& position) { m_Position = position; CalcViewMatrix(); }

		inline float GetAspectRatio() { return m_AspectRatio; }

	private:		
		void CalcProjectionMatrix();
		void CalcViewProjectionMatrix();
		void CalcViewMatrix();
	private:
		float m_AspectRatio;
		float m_Rotation;
		glm::vec3 m_Position = glm::vec3(0.0f,0.0f,0.0f);
		float m_Width, m_Height;
		glm::mat4 m_ProjectionMatrix;
		glm::mat4 m_ViewMatrix;
		glm::mat4 m_ViewProjectionMatrix;


	};
}