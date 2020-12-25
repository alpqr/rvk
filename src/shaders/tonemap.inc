vec3 srgb_to_linear(vec3 c)
{
    return c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878);
}

vec4 srgb_to_linear(vec4 c)
{
    return vec4(srgb_to_linear(c.rgb), c.a);
}

vec3 linear_to_srgb(vec3 c)
{
    vec3 S1 = sqrt(c);
    vec3 S2 = sqrt(S1);
    vec3 S3 = sqrt(S2);
    return 0.585122381 * S1 + 0.783140355 * S2 - 0.368262736 * S3;
}

vec4 linear_to_srgb(vec4 c)
{
    return vec4(linear_to_srgb(c.rgb), c.a);
}
