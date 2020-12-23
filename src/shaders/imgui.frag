#version 440

layout(location = 0) in vec2 v_texcoord;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
};

layout(binding = 1) uniform sampler2D tex;

void main()
{
    fragColor = v_color * texture(tex, v_texcoord);
}
