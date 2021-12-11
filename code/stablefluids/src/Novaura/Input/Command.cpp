#include "sapch.h"
#include "Command.h"
namespace Novaura {

	Command::Command(std::function<void()>&& func)
		: m_Command(func)
	{
	}

	void Command::Execute()
	{
		m_Command();
	}
}