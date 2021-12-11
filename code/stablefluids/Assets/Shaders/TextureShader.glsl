#shader vertex
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec2 aTexCoords;
//layout(location = 3) in vec2 aQuantity;
//layout(location = 4) in float aTexIndex;


out vec4 v_Color;
out vec3 v_Pos;
out vec2 v_TexCoords;
//out float v_TexIndex;

//uniform mat4 u_ViewProjectionMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;

void main()
{
	//gl_Position = u_ProjectionMatrix * vec4(aPos, 1.0);
	gl_Position = u_ProjectionMatrix * u_ViewMatrix * vec4(aPos, 1.0);
	//gl_Position = vec4(aPos, 1.0);
    v_Pos = aPos;
    v_Color = aColor;
    v_TexCoords = aTexCoords;
   // v_TexIndex = aTexIndex;
}

#shader fragment
#version 330 core

out vec4 Color;
in vec4 v_Color;
in vec3 v_Pos;
in vec2 v_TexCoords;
//in float v_TexIndex;
//uniform vec4 u_Color;
uniform vec2 u_Quantity;


uniform sampler2D u_Texture;
//uniform sampler2D u_Textures[32];

void main()
{

   // Color = v_Color;
   // Color = texture(u_Textures[int(v_TexIndex)], vec2(v_TexCoords));
    //Color = texture(u_Texture, vec2(v_TexCoords));
    //Color = vec4(v_TexCoords, 0.0, 1.0);

//   if(v_TexCoords.x == -1.0f)
//   {
//        Color = v_Color;
//   }
//   else
//   {
      // Color = texture(u_Texture, vec2(v_TexCoords.x, v_TexCoords.y) * );
       Color = texture(u_Texture, vec2(v_TexCoords.x * u_Quantity.x, v_TexCoords.y * u_Quantity.y)) * v_Color;
//        Color = texture(u_Textures[int(v_TexIndex)], vec2(v_TexCoords * u_Quantity)) * v_Color;
//    // Color = vec4(v_TexCoords,0.0f,1.0f);
//
//   }


} 