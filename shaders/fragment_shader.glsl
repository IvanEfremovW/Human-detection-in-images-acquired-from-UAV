#version 330 core

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D texture_0;

void main() {
    FragColor = texture(texture_0, texCoord);
}