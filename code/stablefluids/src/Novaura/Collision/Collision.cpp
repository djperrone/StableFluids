#include "sapch.h"
#include "Collision.h"
#include "Novaura/Primitives/Rectangle.h"

namespace Novaura {
	bool IsCollidingAABB(const Rectangle& rectA, const Rectangle& rectB)
	{
		Bounds boundsA = rectA.GetBounds();
		Bounds boundsB = rectB.GetBounds();

		if (boundsA.BottomRight.x >= boundsB.BottomLeft.x &&
			boundsB.BottomRight.x >= boundsA.BottomLeft.x &&
			boundsA.TopLeft.y >= boundsB.BottomRight.y &&
			boundsB.TopRight.y >= boundsA.BottomLeft.y)
		{
			return true;
		}
		return false;
	}
}