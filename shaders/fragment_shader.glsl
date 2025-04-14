#version 330 core

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D imageTexture;
uniform int u_tileSize;
uniform float u_clipLimit;

const int NUM_BINS = 101;
const vec3 STANDART_ILLUMINANT = vec3(0.95047, 1, 1.0888); //D65

// sRGB <-> LinearRGB

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

// LinearRGB <-> XYZ

vec3 linearRGB_to_XYZ(vec3 rgb_color) {
    /*
    return mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    ) * rgb_color;
    */
    return mat3(
        0.4124, 0.2126, 0.0193,
        0.3576, 0.7152, 0.1192, 
        0.1805, 0.0722, 0.9505
    ) * rgb_color;
}

vec3 XYZ_to_linearRGB(vec3 xyz_color) {
    /*
    return mat3(
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252
    ) * xyz_color;
    */
    return mat3(
        3.2406, -0.9689, 0.0557,
        -1.5372, 1.8758, -0.2040,
        -0.4986, 0.0415, 1.0570
    ) * xyz_color;
}

// XYZ <-> LAB

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

    vec3 xyz_color_norm = xyz_color / STANDART_ILLUMINANT;

    float l = 116.0 * f(xyz_color_norm.y) - 16.0;
    float a = 500.0 * (f(xyz_color_norm.x) - f(xyz_color_norm.y));
    float b = 200.0 * (f(xyz_color_norm.y) - f(xyz_color_norm.z));

    return vec3(l, a, b);
}

vec3 lab_to_xyz(vec3 lab_color) {

    float x = STANDART_ILLUMINANT.x * inv_f((lab_color.x + 16.0)/116.0 + lab_color.y/500.0);
    float y = STANDART_ILLUMINANT.y * inv_f((lab_color.x + 16.0)/116.0);
    float z = STANDART_ILLUMINANT.z * inv_f(((lab_color.x + 16.0)/116.0) - lab_color.z/200.0);

    return vec3(x, y, z);
}

// sRGB <-> LAB

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

// CLAHE

ivec2 getTileCoord(vec2 texCoord, sampler2D imageTexture) {
    return ivec2(floor(texCoord * vec2(textureSize(imageTexture, 0)) / float(u_tileSize)));
}

vec2 getLocalCoord(vec2 texCoord, sampler2D imageTexture) {
    return fract(texCoord * vec2(textureSize(imageTexture, 0)) / float(u_tileSize));
}

vec2 convertLocalToGlobalCoord(vec2 tileCoord, vec2 localCoord) {
    return ((tileCoord + localCoord) * u_tileSize) / vec2(textureSize(imageTexture, 0));
}

void calcutaleHistogram(sampler2D imageTexture, ivec2 tileCoord, out float histogram[NUM_BINS]) {
    for (int i = 0; i < NUM_BINS; ++i) {
        histogram[i] = 0.0;
    }

    for (int y = 0; y < u_tileSize; ++y) {
        for (int x = 0; x < u_tileSize; ++x) {
            vec2 offset = vec2(x, y) / u_tileSize;

            vec2 tilePixelCoord = convertLocalToGlobalCoord(tileCoord, offset);

            vec3 tilePixelRGB= texture(imageTexture, clamp(tilePixelCoord, 0.0, 1.0)).rgb;
            vec3 tilePixelLab = sRGB_to_lab(tilePixelRGB);

            float tilePixelLuminance = tilePixelLab.x;
            int index = int(tilePixelLuminance);

            histogram[index] += 1.0;
        }
    }
}

void clipAndRedistribute(inout float histogram[NUM_BINS]) {
    float excess = 0.0;

    for (int i = 0; i < NUM_BINS; ++i) {
        float limit = u_clipLimit * float(u_tileSize * u_tileSize) / NUM_BINS;
    
        if (histogram[i] > limit) {
            excess += histogram[i] - limit;
            histogram[i] = limit;
        }
    }

    float increment = excess / NUM_BINS;
    for (int i = 0; i < NUM_BINS; ++i) {
        histogram[i] += increment;
    }
}

void calculateCDF(in float histogram[NUM_BINS], out float cdf[NUM_BINS]) {
    cdf[0] = histogram[0];

    for (int i = 0; i < NUM_BINS; ++i) {
        cdf[i] = cdf[i-1] + histogram[i];
    }

    for (int i = 0; i < NUM_BINS; ++i) {
        cdf[i] /= float(u_tileSize * u_tileSize);
    }

}

void main() {

    vec3 sRGB_color = texture(imageTexture, texCoord).rgb;
    vec3 lab_color = sRGB_to_lab(sRGB_color);
    float L = lab_color.x;

    ivec2 tileCoord = getTileCoord(texCoord, imageTexture);
    vec2 localUV = getLocalCoord(texCoord, imageTexture);

    ivec2 tileCoord1 = tileCoord;
    ivec2 tileCoord2 = tileCoord + ivec2(1, 0);
    ivec2 tileCoord3 = tileCoord + ivec2(0, 1);
    ivec2 tileCoord4 = tileCoord + ivec2(1, 1);

    // Предотвращение выхода за границы изображения
    ivec2 numTiles = ivec2(textureSize(imageTexture, 0)) / u_tileSize;
    tileCoord2 = ivec2(min(tileCoord2.x, numTiles.x - 1), tileCoord2.y);
    tileCoord3 = ivec2(tileCoord3.x, min(tileCoord3.y, numTiles.y - 1));
    tileCoord4 = ivec2(min(tileCoord4.x, numTiles.x - 1), min(tileCoord4.y, numTiles.y - 1));

    float histogram1[NUM_BINS];
    float histogram2[NUM_BINS];
    float histogram3[NUM_BINS];
    float histogram4[NUM_BINS];

    calcutaleHistogram(imageTexture, tileCoord1, histogram1);
    clipAndRedistribute(histogram1);
    
    calcutaleHistogram(imageTexture, tileCoord2, histogram2);
    clipAndRedistribute(histogram2);
    
    calcutaleHistogram(imageTexture, tileCoord3, histogram3);
    clipAndRedistribute(histogram3);
    
    calcutaleHistogram(imageTexture, tileCoord4, histogram4);
    clipAndRedistribute(histogram4);

    float cdf1[NUM_BINS];
    float cdf2[NUM_BINS];
    float cdf3[NUM_BINS];
    float cdf4[NUM_BINS];

    calculateCDF(histogram1, cdf1);
    calculateCDF(histogram2, cdf2);
    calculateCDF(histogram3, cdf3);
    calculateCDF(histogram4, cdf4);

    // Билинейная интерполяция значений CDF
    float cdfValue = mix(mix(cdf1[int(L)], cdf2[int(L)], localUV.x),
                        mix(cdf3[int(L)], cdf4[int(L)], localUV.x), localUV.y);

    vec3 newLab = vec3(cdfValue * 100.0, lab_color.y, lab_color.z);
    vec3 new_sRGB_color = lab_to_sRGB(newLab);

    FragColor = vec4(clamp(new_sRGB_color, 0.0, 1.0), 1.0);
}