#version 410 core

uniform vec3 lightPosition_sphere;
uniform float sphereRadius;
uniform float latitude;
uniform float longitude;
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

    float lat = 0;
    float longi = 0;

    if (mapType == 0) {
        longi = (uv.x + 0.5) * (2.0 * PI);
        lat = (uv.y + 0.5) * PI;
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

        longi = (uv.x + 0.5) * (2.0 * PI);
        lat = (uv.y + 0.5) * PI;
    }

    vec3 normal_sphere = vec3(
        cos(longi) * cos(lat),
        sin(longi) * cos(lat),
        sin(lat)
    );
    
    vec3 position_sphere = normal_sphere * vec3(sphereRadius);
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

    vec3 target_normal = vec3(
        cos(latitude) * cos(longitude + PI),
        cos(latitude) * sin(longitude + PI),
        sin(latitude)
    );

    // Angular distance (radians) between directions on sphere
    float cosAng = clamp(dot(normal_sphere, target_normal), -1.0, 1.0);
    float angRad = acos(cosAng);

    // Threshold in degrees
    float thresholdDeg = 2.0; // tweak: 1°-3° usually looks like a visible dot
    float thresholdRad = radians(thresholdDeg);

    if (angRad < thresholdRad) {
        FragColor = vec4(1, 0, 0, 1);
    }
}