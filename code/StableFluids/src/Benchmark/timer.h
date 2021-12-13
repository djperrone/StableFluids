#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#define TIME_FUNC func(...)

class Timer{

private:
	std::chrono::time_point<std::chrono::steady_clock> m_StartTimePoint;
	std::chrono::time_point<std::chrono::steady_clock> m_EndTimepoint;
	std::string m_OutFile, m_FunctionName;
	long double start, end, duration;
	unsigned int m_SquareSize;
	std::ofstream m_Stream;


public:
	unsigned int Counter = 0;
	bool CPU = false, GPU = false;
	// Default Constructor
	Timer()
		:m_OutFile("timer.csv"), start(0), end(0), duration(0) {
		//WriteHeader();
	}

	// Standard Constructor for testing - takes in outfile, code, and mod
	Timer(std::string& outFile,unsigned int squareSize, std::string& code, int mod)
		: m_OutFile(outFile.append(".csv")), m_SquareSize(squareSize), start(0), end(0), duration(0) {
		//WriteHeader();
	}	

	void Calc() {
		start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimePoint).time_since_epoch().count();
		end = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_EndTimepoint).time_since_epoch().count();
		duration = (end - start) * 0.000001;
	}

	void Reset() {
		start = end = duration = 0;
	}

	void WriteHeader() {
		
		m_Stream.open(m_OutFile);
		m_Stream << "size,function,duration" << '\n';
		//oStream.close();
	}

	void WriteSeparator(const std::string& separator)
	{
		//oStream.open(m_outFile, std::ios::app);
		m_Stream << separator << '\n';
		//oStream.close();
	}

	void WriteCSV() {
		//std::ofstream oStream;
		//oStream.open(m_outFile, std::ios::app);
		m_Stream << m_SquareSize <<","<< m_FunctionName << "," << duration<< std::endl;
		//oStream.close();
	}	

	inline void Start() { m_StartTimePoint = std::chrono::steady_clock::now(); }
	inline void Stop() { m_EndTimepoint = std::chrono::steady_clock::now(); }
	inline void SetSquareSize(unsigned int size) { m_SquareSize = size; }
	inline void SetFunctionName(const std::string& name) { m_FunctionName = name; }
	inline void Flush() { m_EndTimepoint = std::chrono::steady_clock::now(); Calc(); WriteCSV();  Reset(); }
	inline double GetDuration() const { return duration; }
	inline void SetOutFile(const std::string& outFile) { m_OutFile = outFile; m_OutFile.append(".csv"); }
};