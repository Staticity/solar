#version 410 core

uniform vec3 lightPosition_sphere;
uniform vec3 sphereRadius;
uniform int mapType;
uniform vec2 resolution;

uniform sampler2D sphereTexture;

out vec4 FragColor;

const float PI = 3.1415926535897932384626433832795;

void main() {
    // 
    // u = 0.5 + atan(position_dir.y, position_dir.x) / (2.0 * PI)
    // asin(position_dir.z) / PI - 0.5;

    vec2 uv = gl_FragCoord.xy / resolution;
    uv.y = 1 - uv.y;

    float latitude = 0;
    float longitude = 0;

    if (mapType == 0) {
        longitude = (uv.x + 0.5) * (2.0 * PI);
        latitude = (uv.y + 0.5) * PI;
    } else if (mapType == 1) {
        vec2 xy = vec2(.5, .5) - uv;
        float radius = length(xy);
        if (radius >= .5) {
            FragColor=vec4(1, 1, 1, 1) * .1;
            return;
        }

        float u = atan(xy.y, xy.x) / (2 * PI);
        float v = 1 - (radius / .5);
        uv = vec2(u, v);

        longitude = (uv.x + 0.5) * (2.0 * PI);
        latitude = (uv.y + 0.5) * PI;
    }

    vec3 normal_sphere = vec3(
        cos(longitude) * cos(latitude),
        sin(longitude) * cos(latitude),
        sin(latitude)
    );
    
    vec3 position_sphere = normal_sphere * sphereRadius;
    float value = dot(normalize(position_sphere - lightPosition_sphere), normal_sphere);
    float intensity = max(0, value);

    float threshold = .005;
    if (abs(value) <= .005) {
        float p = 1 - abs(value) / threshold;
        FragColor = p * vec4(1, 1, 55.0/255, 1);
        return;
    }
    float alpha = 0;
    FragColor = texture(sphereTexture, uv) * (alpha + (1 - alpha) * intensity);

}