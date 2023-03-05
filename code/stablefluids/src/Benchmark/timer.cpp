#include "sapch.h"
#include "timer.h"

void Timer::Flush()
{
	//m_EndTimepoint = std::chrono::steady_clock::now(); Calc(); WriteCSV();  Reset();
	for (auto& pair : m_Data)
	{
		m_Stream << pair.first << ',';
		long double sum = 0.0;
		for (auto& value : pair.second)
		{
			sum += value;
			m_Stream << value << ',';
		}
		m_Averages[pair.first] = sum / pair.second.size();
		m_Stream << '\n';
	}
	m_Stream << '\n';
	for (auto& pair : m_Averages)
	{
		m_Stream << pair.first << " average:," << pair.second << '\n';
	}

	m_Data.clear();
	m_Stream.close();
}

void Timer::PushValue()
{
	m_Data[m_FunctionName].push_back(m_Duration);
}
