#pragma once
namespace Novaura {
	class State;
	class StateMachine
	{
	public:
		StateMachine() = default;

		void PushState(std::unique_ptr<State> state);
		void ReplaceCurrentState(std::unique_ptr<State> state);
		void PopState();
		void ShutDown();
		void ClearPastStates();


		inline State& GetCurrentState() { return *m_States.top(); }
		inline const State& GetCurrentState() const { return *m_States.top(); }


		std::stack<std::unique_ptr<State>>& GetStateStack() { return m_States; }

	private:
		std::stack<std::unique_ptr<State>> m_States;

	};

}