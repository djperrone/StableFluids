#include "sapch.h"
#include "CameraController.h"
#include <spdlog/spdlog.h>

namespace Novaura {
	CameraController::CameraController(float width, float height)
		: m_Camera(width, height)
	{
	}
	void CameraController::Update(Window& window, float deltaTime)
	{
		if (glfwGetKey(window.Window, GLFW_KEY_LEFT) == GLFW_PRESS)
		{
			//spdlog::info("a pressed");
			m_Position.x -= m_CameraSpeed * deltaTime;

			/*m_Position.x -= cos(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;
			m_Position.y -= sin(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;*/
			//spdlog::info("camera pos {0}, {1}", m_Position.x, m_Position.y);
		}
		if (glfwGetKey(window.Window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		{
			//spdlog::info("d pressed");
			m_Position.x+= m_CameraSpeed * deltaTime;
			/*m_Position.x += cos(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;
			m_Position.y += sin(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;*/
			//spdlog::info("camera pos {0}, {1}", m_Position.x, m_Position.y);
		}
		if (glfwGetKey(window.Window, GLFW_KEY_UP) == GLFW_PRESS)
		{
		//	spdlog::info("w pressed");
			m_Position.y+= m_CameraSpeed * deltaTime;
		/*	m_Position.x += -sin(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;
			m_Position.y += cos(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;*/
			//spdlog::info("camera pos {0}, {1}",m_Position.x, m_Position.y);
		}
		if (glfwGetKey(window.Window, GLFW_KEY_DOWN) == GLFW_PRESS)
		{
			//spdlog::info("s pressed");
		//	spdlog::info("camera pos {0}, {1}", m_Position.x, m_Position.y);
			m_Position.y-= m_CameraSpeed * deltaTime;
			/*m_Position.x -= -sin(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;
			m_Position.y -= cos(glm::radians(m_Rotation)) * m_CameraSpeed * deltaTime;*/
		}
		
		m_Camera.SetViewMatrix(m_Position);
	
	}
	void CameraController::ProcessMouseScroll(double yoffset)
	{
		//m_Rotation += yoffset;
		m_Zoom -= yoffset * 0.25f;
		m_Zoom = std::max(m_Zoom, 0.25f);
		//m_Bounds = { -m_Camera.GetAspectRatio() * m_Zoom, m_Camera.GetAspectRatio() * m_Zoom, -m_Zoom, m_Zoom };
		m_Camera.SetProjectionMatrix(-m_Camera.GetAspectRatio() * m_Zoom, m_Camera.GetAspectRatio() * m_Zoom, -m_Zoom, m_Zoom);
	}
}