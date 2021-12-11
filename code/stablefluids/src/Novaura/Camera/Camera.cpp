#include "sapch.h"
#include "Camera.h"

namespace Novaura {

	Camera::Camera(float width, float height)
		: m_AspectRatio(width / height), m_Width(width), m_Height(height), m_ProjectionMatrix(glm::ortho(-m_AspectRatio, m_AspectRatio, -1.0f, 1.0f, -1.0f, 1.0f)), m_ViewMatrix(glm::mat4(1.0f))
		//: m_AspectRatio(width / height), m_Width(width), m_Height(height), m_ProjectionMatrix(glm::ortho(0.0f, m_AspectRatio, 0.0f, m_AspectRatio, -1.0f, 1.0f)), m_ViewMatrix(glm::mat4(1.0f))
		//: m_AspectRatio(width / height), m_Width(width), m_Height(height), m_ProjectionMatrix(glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f)), m_ViewMatrix(glm::mat4(1.0f))
	{		
		SetViewMatrix(glm::vec3(0.0f,0.0f,0.0f), 0.0f);
		CalcProjectionMatrix();
		CalcViewProjectionMatrix();
	}
	void Camera::SetProjectionMatrix(float width, float height)
	{
		m_AspectRatio = (width / height);
		m_Width = width;
		m_Height = height;
		//m_ProjectionMatrix = glm::ortho(-m_AspectRatio, m_AspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);
		//m_ProjectionMatrix = glm::ortho(0.0f, m_AspectRatio, 0.0f, m_AspectRatio, -1.0f, 1.0f);
		//m_ProjectionMatrix = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
		m_ProjectionMatrix = glm::ortho(-m_AspectRatio, m_AspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);

		CalcViewProjectionMatrix();
	}
	void Camera::SetProjectionMatrix(float left, float right, float bottom, float top)
	{
		//m_AspectRatio = (width / height);
		//m_Width = width;
		//m_Height = height;
		//m_ProjectionMatrix = glm::ortho(-m_AspectRatio, m_AspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);
		//m_ProjectionMatrix = glm::ortho(0.0f, m_AspectRatio, 0.0f, m_AspectRatio, -1.0f, 1.0f);
		//m_ProjectionMatrix = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
		m_ProjectionMatrix = glm::ortho(left, right, bottom, top, -1.0f, 1.0f);
	}
	void Camera::SetViewMatrix(const glm::vec3& position, float rotation)
	{
		m_ViewMatrix = glm::inverse(glm::translate(glm::mat4(1.0f), position) * glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0.0f,0.0f,1.0f)));
		//m_ViewMatrix = glm::translate(glm::mat4(1.0f), position);
		CalcViewProjectionMatrix();
	}

	void Camera::SetViewMatrix(const glm::vec3& position)
	{
		m_ViewMatrix = glm::inverse(glm::translate(glm::mat4(1.0f), position));
		//m_ViewMatrix = glm::translate(glm::mat4(1.0f), position);
		CalcViewProjectionMatrix();
	}
	
	void Camera::CalcProjectionMatrix()
	{
		m_ProjectionMatrix = glm::ortho(-m_AspectRatio, m_AspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);
		//m_ProjectionMatrix = glm::ortho(0.0f, m_Width, 0.0f, m_Height, -1.0f, 1.0f);

	}	
	void Camera::CalcViewProjectionMatrix()
	{		
		m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
	}
	void Camera::CalcViewMatrix()
	{
		//glm::mat4 transform = glm::translate(glm::mat4(1.0f), m_Position) * glm::rotate(glm::mat4(1.0f), glm::radians(m_Rotation), glm::vec3(0, 0, 1));

		//m_ViewMatrix = glm::inverse(transform);
		m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
	}
}