#shader vertex
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;


out vec4 v_Color;
out vec3 v_Pos;


uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;

void main()
{
	gl_Position = u_ProjectionMatrix * u_ViewMatrix * vec4(aPos, 1.0);
	//gl_Position = vec4(aPos, 1.0);
    v_Pos = aPos;
    v_Color = aColor;
}

#shader fragment
#version 330 core

out vec4 FragColor;
in vec4 v_Color;
in vec3 v_Pos;
uniform vec4 u_Color;

void main()
{
    //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    FragColor = v_Color;
} 