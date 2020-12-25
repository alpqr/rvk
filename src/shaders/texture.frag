#version 440
#extension GL_GOOGLE_include_directive: enable

#include "tonemap.inc"

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
};

layout(binding = 1) uniform sampler2D tex;

void main()
{
    vec4 c = srgb_to_linear(texture(tex, v_texcoord));
    fragColor = linear_to_srgb(c);
}
