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
    //v_Pos = vec3(gl_Position.x,gl_Position.y,gl_Position.z);
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
in vec4 v_TexCoords;
in vec2 v_Quantity;
in float v_TexIndex;

//uniform sampler2D u_Textures[32];

void main()
{       
    float distance = 1.0 - length(vec2(v_Pos.x - v_TexCoords.x, v_Pos.y - v_TexCoords.y));   
    //float distance = 1.0 - length(vec2(v_Pos.x, v_Pos.y));   
    float fade = 0.005;   
    float cutoff = 1.0 - (v_Quantity.x / 2.0);
    distance = smoothstep(cutoff, cutoff + fade, distance);
   
    Color = vec4(distance) * v_Color;
} 


// shadertoy.com
//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//{
//    // Normalized pixel coordinates (from 0 to 1)
//    vec2 uv = fragCoord/iResolution.xy * 2.0 - 1.0;
//    float aspect = iResolution.x / iResolution.y;
//    uv.x *= aspect;    
//  
//    float fade = 0.005;
//    float distance = 1.0 - length(uv);
//    distance = smoothstep(0.0, fade, distance);        
//     
//     fragColor.rgb = vec3(distance);    
//}