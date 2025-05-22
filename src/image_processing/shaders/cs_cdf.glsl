#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

struct TileData {
    float histogram[101];
    float cdf[101];
};

layout(std430, binding = 0) buffer HistogramCdfBuffer {
    TileData tileData[];
};

layout(rgba8, binding = 1) uniform image2D inputImage;

uniform int u_tileSize;
uniform float u_clipLimit;
uniform int imageWidth;
uniform int imageHeight;

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

    return mat3(
        0.4124, 0.2126, 0.0193,
        0.3576, 0.7152, 0.1192, 
        0.1805, 0.0722, 0.9505
    ) * rgb_color;

}

vec3 XYZ_to_linearRGB(vec3 xyz_color) {

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

ivec2 getTileCoordinates(int imageX, int imageY, int tileSize) {
    return ivec2(imageX / tileSize, imageY / tileSize);
}

vec2 getLocalCoordinates(int imageX, int imageY, int tileSize) {
    return vec2(float(imageX % tileSize) / tileSize, float(imageY % tileSize) / tileSize);
}

uint getTileIndex(int tileX, int tileY) {
    uint numTilesX = imageWidth / u_tileSize;
    return tileX + tileY * numTilesX;
}

shared int sharedHistogram[NUM_BINS]; // Shared memory for histogram

void main() {
    uint globalIdX = gl_GlobalInvocationID.x;
    uint globalIdY = gl_GlobalInvocationID.y;
    uint localIdX = gl_LocalInvocationID.x;
    uint localIdY = gl_LocalInvocationID.y;

    int imageX = int(globalIdX);
    int imageY = int(globalIdY);

    if(imageX >= imageWidth || imageY >= imageHeight){
        return;
    }

    ivec2 tileCoordinates = getTileCoordinates(imageX, imageY, u_tileSize);
    int tileX = tileCoordinates.x;
    int tileY = tileCoordinates.y; 

    uint tileIndex = getTileIndex(tileX, tileY);

    for (int i = 0; i < NUM_BINS; ++i) {
        sharedHistogram[i] = 0;
    }

    barrier(); // Ensure all workers have initialized histogram

    vec4 pixel = imageLoad(inputImage, ivec2(imageX, imageY));

    vec3 sRGB_color = pixel.rgb;
    vec3 lab_color = sRGB_to_lab(sRGB_color);
    float luminance = lab_color.x;

    int binIndex = int(luminance);

    // Calculate histogram
    atomicAdd(sharedHistogram[binIndex], 1);

    barrier(); // Ensure all increments are complete
    groupMemoryBarrier();

    // Performed by a single worker to avoid data races
    if (localIdX == 0 && localIdY == 0) {
        // Clip and redistribute histogram 
        float excess = 0.0;
        float limit = u_clipLimit * float(u_tileSize * u_tileSize) / NUM_BINS;

        float floatHistogram[NUM_BINS];
        for (int i = 0; i < NUM_BINS; ++i) {
            floatHistogram[i] = float(sharedHistogram[i]);
        }

        for (int i = 0; i < NUM_BINS; ++i) {
            if (floatHistogram[i] > limit) {
                excess += floatHistogram[i] - limit;
                floatHistogram[i] = limit;
            }
        }

        float increment = excess / NUM_BINS;
        for (int i = 0; i < NUM_BINS; ++i) {
                floatHistogram[i] += increment;
        }

        //Calculate CDF
        float cdf[NUM_BINS];
        cdf[0] = floatHistogram[0];

        for (int i = 1; i < NUM_BINS; ++i) {
            cdf[i] = cdf[i-1] + floatHistogram[i];
        }

        for (int i = 0; i < NUM_BINS; ++i) {
            tileData[tileIndex].histogram[i] = floatHistogram[i];
            cdf[i] /= float(u_tileSize * u_tileSize);
            tileData[tileIndex].cdf[i] = cdf[i];
        }
    }

    groupMemoryBarrier(); // Ensure CDF calculation is complete before exiting
}