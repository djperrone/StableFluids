#shader vertex

#version 330 core
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in mat4 instanceMatrix;
//layout(location = 6) in vec4 color;

out vec4 v_Pos;
out vec4 v_Color;

uniform mat4 u_ViewProjectionMatrix;

void main()
{
    v_Pos = aPos;
    v_Color = aColor;
    gl_Position = u_ViewProjectionMatrix * instanceMatrix * vec4(aPos.x, aPos.y, aPos.z, 1.0);
}

#shader fragment
#version 330 core

out vec4 Color;
in vec4 v_Pos;
in vec4 v_Color;

void main()
{    
    Color = v_Color;
}