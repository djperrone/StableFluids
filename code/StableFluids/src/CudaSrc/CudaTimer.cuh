#pragma once

#include <chrono>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include "CudaSrc/CUDA_KERNEL.h"

class CudaTimer {

private:
	
	std::string m_OutFile, m_FunctionName;
	long double m_Duration;
	unsigned int m_SquareSize;
	std::ofstream m_Stream;

	std::map<std::string, std::vector<float>> m_Data;
	std::map<std::string, float > m_Averages;

	cudaEvent_t start, stop;


public:
	unsigned int Counter = 0;
	bool CPU = false, GPU = false;
	// Default Constructor
	CudaTimer()
		:m_OutFile("timer.csv"), m_Duration(0) {
		//WriteHeader();
	}

	// Standard Constructor for testing - takes in outfile, code, and mod
	CudaTimer(std::string& outFile, unsigned int squareSize, std::string& code, int mod)
		: m_OutFile(outFile.append(".csv")), m_SquareSize(squareSize), m_Duration(0) {
		//WriteHeader();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	void Calc() {
		
	}

	void Reset() {
		m_Duration = 0;
	}

	void OpenAppend()
	{
		m_Stream.open(m_OutFile, std::ios::app);
	}

	void WriteHeader() {

		m_Stream.open(m_OutFile);
		m_Stream << "size: " << std::to_string(m_SquareSize) << '\n' << "function,m_Duration (ms)" << '\n';
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
		m_Stream << m_SquareSize << "," << m_FunctionName << "," << m_Duration << std::endl;
		//oStream.close();
	}

	inline void BeginTimeFunction(const std::string& funcName)
	{
		m_FunctionName = funcName;
		Start();
	}

	inline void EndTimeFunction()
	{
		Stop();
		float milliseconds = 0.0f;
		cudaEventElapsedTime(&milliseconds, start, stop);
		PushValue(milliseconds); Reset();
	}

	inline void PushValue(float value)
	{
		printf("value: %f\n", value);
		m_Data[m_FunctionName].push_back(value);
	}

	void Flush();


	inline void Start() { cudaEventRecord(start); }
	inline void Stop() {
		cudaEventSynchronize(stop);
		
		cudaEventRecord(stop); }
	inline void SetSquareSize(unsigned int size) { m_SquareSize = size; }
	inline void SetFunctionName(const std::string& name) { m_FunctionName = name; }
	//inline double GetDuration() const { return m_Duration; }
	inline void SetOutFile(const std::string& outFile) { m_OutFile = outFile; m_OutFile.append(".csv"); }
};