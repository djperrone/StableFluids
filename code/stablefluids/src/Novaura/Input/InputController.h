#pragma once
#include "Command.h"

namespace Novaura {

	using EventType = int;
	using KeyCode = int;

	class InputController
	{
	public:
		InputController() = default;

		template <typename... T>
		void BindAxisInputEvent(KeyCode code, T&&... args)
		{
			m_AxisInputBindings[code] = Command(std::bind(args...));
		}
		template <typename... T>
		void BindActionInputEvent(EventType action, KeyCode code, T&&... args)
		{
			m_ActionInputBindings[action][code] = Command(std::bind(args...));
		}

		inline std::unordered_map<int, Command>& GetAxisInputBindings() { return m_AxisInputBindings; }
		inline const std::unordered_map<int, Command>&  GetAxisInputBindings() const{ return m_AxisInputBindings; }

		inline std::unordered_map<EventType, std::unordered_map<KeyCode, Command>>& GetActionInputBindings(){ return m_ActionInputBindings; }
		inline const std::unordered_map<EventType, std::unordered_map<KeyCode, Command>>& GetActionInputBindings() const { return m_ActionInputBindings; }

	private:
		std::unordered_map<int, Command> m_AxisInputBindings;
		std::unordered_map<EventType, std::unordered_map<KeyCode, Command>> m_ActionInputBindings;
	};
}