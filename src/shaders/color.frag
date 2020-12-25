#version 440
#extension GL_GOOGLE_include_directive: enable

#include "tonemap.inc"

layout(location = 0) in vec3 v_color;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    float opacity;
};

void main()
{
    vec3 c = srgb_to_linear(v_color);
    fragColor = linear_to_srgb(vec4(c, opacity));
}
