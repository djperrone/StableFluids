#include "sapch.h"
#include "StateMachine.h"
#include "State.h"

namespace Novaura {
 
    void StateMachine::PushState(std::unique_ptr<State> state)
    {
        m_States.push(std::move(state));
    }

    void StateMachine::ReplaceCurrentState(std::unique_ptr<State> state)
    {
        m_States.top()->OnExit();
        m_States.pop();
        m_States.push(std::move(state));
    }

    void StateMachine::PopState()
    {
        m_States.pop();
        if (!m_States.empty())
        {
            m_States.top()->Resume();
        }
    }
    void StateMachine::ShutDown()
    {
        while (!m_States.empty())
        {
            m_States.pop();
        }
        glfwTerminate();
        exit(0);
    }
    void StateMachine::ClearPastStates()
    {
        auto state = std::move(m_States.top());
        while (!m_States.empty())
        {
            m_States.pop();
        }
        m_States.push(std::move(state));
    }
}