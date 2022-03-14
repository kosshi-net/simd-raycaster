#version 330

in vec2 texcoord;

uniform sampler2D uTex;

out vec4 out_Color;

void main(void) {
	out_Color = vec4((texture2D(uTex, texcoord).rgb), 1.0);
}
