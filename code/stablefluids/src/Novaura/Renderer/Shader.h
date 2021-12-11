#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h> 

namespace Novaura {

	struct ShaderProgramSource
	{
		std::string VertexSource;
		std::string FragmentSource;
	};


	class Shader
	{
	public:
		Shader(const std::string& vertexPath, const std::string& fragmentPath);
		Shader(const std::string& shaderPath);

		void Bind();

		void SetBool(const std::string& name, bool vlaue) const;
		void SetInt(const std::string& name, int value) const;
		void SetFloat(const std::string& name, float value) const;

		void SetUniform1i(const std::string& name, int value);
		void SetUniform1f(const std::string& name, float value);
		void SetUniform2f(const std::string& name, float value1, float value2);

		void SetUniform3f(const std::string& name, const glm::vec3& value);
		void SetUniform4f(const std::string& name, const glm::vec4& value);
		void SetUniform4f(const std::string& name, const glm::vec4&& value);

		void SetUniformMat3f(const std::string& name, const glm::mat3& matrix);
		void SetUniformMat4f(const std::string& name, const glm::mat4& matrix);

		void SetIntArray(const std::string& name, int* values, uint32_t count);


		inline unsigned int GetID() const { return m_ShaderID; }

	private:
		ShaderProgramSource ParseShader(const std::string& vertexPath, const std::string& fragmentPath);
		ShaderProgramSource ParseShader(const std::string& shaderPath);
		unsigned int CompileShader(unsigned int type, const std::string& source);
		unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);
		GLint GetUniformLocation(const std::string& name) const;


	private:
		unsigned int m_ShaderID;
		// mutable to modify map in a const function
		mutable std::unordered_map<std::string, GLint> m_UniformLocationCache;
	};
}