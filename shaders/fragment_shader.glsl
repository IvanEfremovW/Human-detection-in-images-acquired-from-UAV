#version 330 core

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D imageTexture;

const vec3 STANDAR_ILLUMINANT = vec3(0.95047, 1, 1.0888); //D65


vec3 sRGB_to_linearRGB(vec3 srgb_color) {

    vec3 linear_color;

    for (int i = 0; i < 3; i++) {
        if (srgb_color[i] <= 0.04045) {
            linear_color[i] = srgb_color[i] / 12.92;
        }
        else {
            linear_color[i] = pow((srgb_color[i] + 0.055) / 1.055, 2.4);
        }
    }

    return linear_color;
}

vec3 linearRGB_to_sRGB(vec3 linearRGB_color) {

    vec3 sRGB_color;

    for (int i = 0; i < 3; i++) {
        if (linearRGB_color[i] <= 0.0031308) {
             sRGB_color[i] = 12.92 * linearRGB_color[i];
        } else {
             sRGB_color[i] = 1.055 * pow(linearRGB_color[i], 1.0 / 2.4) - 0.055;
        }
    }
    return sRGB_color;
}


vec3 linearRGB_to_XYZ(vec3 rgb_color) {

    return mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    ) * rgb_color;
}

vec3 XYZ_to_linearRGB(vec3 xyz_color) {

    return mat3(
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252
    ) * xyz_color;
}


float f(float t) {

    if (t > .008856) {
        return pow(t, 1.0 / 3.0);
    }
    else {
        return (7.787 * t) + (16.0 / 116.0);
    }
}

float inv_f(float t) {

    if (t > 0.2068966) {
        return pow(t, 3.0);
    } else {
        return (t - 16.0 / 116.0) / 7.787;
    }
}

vec3 xyz_to_lab(vec3 xyz_color) {

    vec3 xyz_color_norm = xyz_color / STANDAR_ILLUMINANT;

    float l = 116.0 * f(xyz_color_norm.y) - 16.0;
    float a = 500.0 * (f(xyz_color_norm.x) - f(xyz_color_norm.y));
    float b = 200.0 * (f(xyz_color_norm.y) - f(xyz_color_norm.z));

    return vec3(l, a, b);
}

vec3 lab_to_xyz(vec3 lab_color) {

    float x = STANDAR_ILLUMINANT.x * inv_f((lab_color.x + 16.0)/116.0 + lab_color.y/500.0);
    float y = STANDAR_ILLUMINANT.y * inv_f((lab_color.x + 16.0)/116.0);
    float z = STANDAR_ILLUMINANT.z * inv_f(((lab_color.x + 16.0)/116.0) - lab_color.z/200.0);

    return vec3(x, y, z);
}


vec3 sRGB_to_lab(vec3 sRGB_color) {
    vec3 linearRGB_color = sRGB_to_linearRGB(sRGB_color);
    vec3 xyz_color = linearRGB_to_XYZ(linearRGB_color);
    vec3 lab_color = xyz_to_lab(xyz_color);
    
    return lab_color;
}

vec3 lab_to_sRGB(vec3 lab_color) {
    vec3 xyz_color = lab_to_xyz(lab_color);
    vec3 linearRGB_color = XYZ_to_linearRGB(xyz_color);
    vec3 sRGB_color = linearRGB_to_sRGB(linearRGB_color);
    
    return sRGB_color;
}


void main() {

    vec3 sRGB_color = texture(imageTexture, texCoord).rgb;
    
    vec3 lab_color = sRGB_to_lab(sRGB_color);
    vec3 new_sRGB_color = lab_to_sRGB(lab_color);

    FragColor = vec4(new_sRGB_color, 1.0);
}