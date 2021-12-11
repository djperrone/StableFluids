#shader vertex
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec2 aQuantity;
layout(location = 4) in float aTexIndex;


out vec4 v_Color;
out vec3 v_Pos;
out vec2 v_TexCoords;
out vec2 v_Quantity;
out float v_TexIndex;

uniform mat4 u_ViewProjectionMatrix;

void main()
{	
	gl_Position = u_ViewProjectionMatrix * vec4(aPos, 1.0);
	
    v_Pos = aPos;
    v_Color = aColor;
    v_TexCoords = aTexCoords;
    v_TexIndex = aTexIndex;
    v_Quantity = aQuantity;
}

#shader fragment
#version 330 core

out vec4 Color;

in vec3 v_Pos;
in vec4 v_Color;
in vec2 v_TexCoords;
in vec2 v_Quantity;
in float v_TexIndex;

uniform sampler2D u_Textures[32];

void main()
{   
   if(v_TexCoords.x == -1.0f)
   {
        Color = v_Color;
   }
    else
    {     
       Color = texture(u_Textures[int(v_TexIndex)], vec2(v_TexCoords.x * v_Quantity.x, v_TexCoords.y * v_Quantity.y)) * v_Color;
    }
} 