#include "sapch.h"
#include "Shader.h"

namespace Novaura {


    Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
    {
        ShaderProgramSource source = ParseShader(vertexPath, fragmentPath);
        m_ShaderID = CreateShader(source.VertexSource, source.FragmentSource);
    }

    Shader::Shader(const std::string& shaderPath)
    {
        ShaderProgramSource source = ParseShader(shaderPath);
        m_ShaderID = CreateShader(source.VertexSource, source.FragmentSource);
    }

    ShaderProgramSource Shader::ParseShader(const std::string& vertexPath, const std::string& fragmentPath)
    {
        std::string vertexCode;
        std::string fragmentCode;

        std::ifstream vShaderFile;
        std::ifstream fShaderFile;

        std::string line;
        std::stringstream vss;
        std::stringstream fss;

        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);

        while (std::getline(vShaderFile, line))
        {
            vss << line << '\n';
        }

        while (std::getline(fShaderFile, line))
        {
            fss << line << '\n';
        }
        fragmentCode = fss.str();

        vertexCode = vss.str();
        vShaderFile.close();
        fShaderFile.close();
        return { vertexCode, fragmentCode };
    }

    ShaderProgramSource Shader::ParseShader(const std::string& shaderPath)
    {
        std::ifstream shaderFile;

        std::string line;
        std::stringstream vss;
        std::stringstream fss;

        shaderFile.open(shaderPath);
        bool isVertex = false, isFragment = false;

        while (std::getline(shaderFile, line))
        {
            if (line.find("#shader") != std::string::npos)
            {
                if (line.find("vertex") != std::string::npos)
                {
                    isVertex = true;
                    isFragment = false;
                }
                else if (line.find("fragment") != std::string::npos)
                {
                    isFragment = true;
                    isVertex = false;
                }
            }
            else
            {
                if (isFragment)
                {
                    fss << line << '\n';
                }
                else if (isVertex)
                {
                    vss << line << '\n';
                }
            }
        }
        shaderFile.close();
        return { vss.str(), fss.str() };
    }

    unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
    {
        unsigned int id = glCreateShader(type);

        const char* src = source.c_str();

        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);

        int success;
        char infoLog[512];

        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(id, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << "VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
            glDeleteShader(id);
            return 0;
        };

        return id;
    }

    unsigned int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
    {
        unsigned int program = glCreateProgram();
        unsigned int vertex = CompileShader(GL_VERTEX_SHADER, vertexShader);
        unsigned int fragment = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);


        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);

        int success;
        char infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            return 0;
        }

        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);

        return program;
    }

    GLint Shader::GetUniformLocation(const std::string& name) const
    {
        if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
        {
            return m_UniformLocationCache[name];
        }
        else
        {
            m_UniformLocationCache[name] = glGetUniformLocation(m_ShaderID, name.c_str());
            return m_UniformLocationCache[name];
        }
    }

    void Shader::Bind()
    {
        glUseProgram(m_ShaderID);
    }

    void Shader::SetUniform1i(const std::string& name, int value)
    {
        glUniform1i(GetUniformLocation(name), value);
    }

    void Shader::SetUniform1f(const std::string& name, float value)
    {
        glUniform1f(GetUniformLocation(name), value);
    }

    void Shader::SetUniform2f(const std::string& name, float value1, float value2)
    {
        GLint location = GetUniformLocation(name);
        glUniform2f(location, value1, value2);
    }

    void Shader::SetUniform3f(const std::string& name, const glm::vec3& value)
    {
        GLint location = GetUniformLocation(name);
        glUniform3f(location, value.x, value.y, value.z);
    }

    void Shader::SetUniform4f(const std::string& name, const glm::vec4& value)
    {
        GLint location = GetUniformLocation(name);
        glUniform4f(location, value.x, value.y, value.z, value.w);
    }

    void Shader::SetUniform4f(const std::string& name, const glm::vec4&& value)
    {
        GLint location = GetUniformLocation(name);
        glUniform4f(location, value.x, value.y, value.z, value.w);
    }

    void Shader::SetUniformMat3f(const std::string& name, const glm::mat3& matrix)
    {
        GLint location = GetUniformLocation(name);
        glUniformMatrix3fv(location, 1, GL_FALSE, &matrix[0][0]);
    }

    void Shader::SetUniformMat4f(const std::string& name, const glm::mat4& matrix)
    {
        GLint location = GetUniformLocation(name);
        glUniformMatrix4fv(location, 1, GL_FALSE, &matrix[0][0]);
    }



    void Shader::SetBool(const std::string& name, bool value) const
    {
        glUniform1i(GetUniformLocation(name), (int)value);
    }
    void Shader::SetInt(const std::string& name, int value) const
    {
        glUniform1i(GetUniformLocation(name), value);
    }
    void Shader::SetFloat(const std::string& name, float value) const
    {
        glUniform1f(GetUniformLocation(name), value);
    }

    void Shader::SetIntArray(const std::string& name, int* values, uint32_t count)
    {      
        glUniform1iv(GetUniformLocation(name), count, values);
    }

}