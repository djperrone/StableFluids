#include "sapch.h"
#include "Actor.h"

namespace Simulation {

	Bounds Simulation::Actor::GetBounds() const
	{
		Bounds bounds;
		bounds.BottomLeft = { m_Rect->GetPosition().x - m_Rect->GetScale().x * 0.5f, m_Rect->GetPosition().y - m_Rect->GetScale().y * 0.5f };
		bounds.BottomRight = { m_Rect->GetPosition().x + m_Rect->GetScale().x * 0.5f, m_Rect->GetPosition().y - m_Rect->GetScale().y * 0.5f };
		bounds.TopLeft = { m_Rect->GetPosition().x - m_Rect->GetScale().x * 0.5f, m_Rect->GetPosition().y + m_Rect->GetScale().y * 0.5f };
		bounds.TopRight = { m_Rect->GetPosition().x + m_Rect->GetScale().x * 0.5f, m_Rect->GetPosition().y + m_Rect->GetScale().y * 0.5f };
		return bounds;
	}

}