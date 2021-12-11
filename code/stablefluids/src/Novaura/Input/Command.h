#pragma once

namespace Novaura {
	class Command
	{
	public:
		Command() = default;
		Command(std::function<void()>&& func);
		void Execute();

	private:
		std::function<void()> m_Command;
	};
}
