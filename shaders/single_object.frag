#version 410 core

// Camera Parameters
uniform mat3 K;
uniform mat4 T_world_camera;

// Light parameters
uniform int isMatte;
uniform vec3 light_shape;

// Shape parameters
uniform int shapeType;
uniform mat4 T_shape_camera;
uniform vec4 shapeParameters[10];

// Shape texture
uniform sampler2D objectTexture;

// Output
out vec4 FragColor;

// Constants
const float PI = 3.1415926535897932384626433832795;
const int MaximumSteps = 1024;
const float MaximumDistance = 1e6;
const float MinimumDistanceRatio = 1e-6;

struct SDInfo {
    float dist;
    vec2 uv;
};

struct SDFHit {
    bool hit;
    vec3 normal;
    vec3 position;
    vec2 uv;
    int steps;
    float nearest;
    float travelled;
};

SDInfo signedDistance(vec3 position_world) {
    SDInfo info;    

    if (shapeType == 1) {
        float radius = shapeParameters[0].x;
        vec3 position_shape = vec4(position_world, 1.0).xyz;

        info.dist = length(position_shape) - radius;

        // UV calculation
        vec3 position_dir = normalize(position_shape);
        info.uv = vec2(
            1 - (0.5 + atan(position_dir.y, position_dir.x) / (2.0 * PI)),
            0.5 - asin(position_dir.z) / PI);
    } else if (shapeType == 2) {
        float radius = shapeParameters[0].x;
        float height = shapeParameters[0].y;
        vec3 position_shape = vec4(position_world, 1.0).xyz;
        info.dist = max(length(position_shape.xz) - radius, abs(position_shape.y) - height);

        // UV calculation
        vec2 position_dir_xz = normalize(position_shape.xy);
        info.uv = vec2(
            (atan(position_dir_xz.y, position_dir_xz.x) + PI) / (2 * PI),
            length(position_shape.xz) / radius
        );
    } else {
        info.dist = 0;
        info.uv = vec2(0, 0);
    }

    return info;
}

vec3 sdfNormal(vec3 position_world) {
    // if (shapeType == 1) {
    //     return -normalize(vec4(position_world, 1).xyz);
    // }

    float eps = 1e-5;// length(position_world) * MinimumDistanceRatio;

    float fx = signedDistance(position_world + vec3(eps, 0, 0)).dist;
    float fy = signedDistance(position_world + vec3(0, eps, 0)).dist;
    float fz = signedDistance(position_world + vec3(0, 0, eps)).dist;

    float bx = signedDistance(position_world - vec3(eps, 0, 0)).dist;
    float by = signedDistance(position_world - vec3(0, eps, 0)).dist;
    float bz = signedDistance(position_world - vec3(0, 0, eps)).dist;

    return (vec3(fx, fy, fz) - vec3(bx, by, bz)) / (2 * eps);
}

SDFHit raymarch(vec3 camera_world, vec3 direction) {
    SDFHit result;
    result.nearest = MaximumDistance;

    float t = 0.0;
    for (int it = 0; it < MaximumSteps && t < MaximumDistance; ++it) {
        vec3 position_world = camera_world + direction * t;
        SDInfo info = signedDistance(position_world);
        if (info.dist < result.nearest) {
            result.nearest = info.dist;
        }

        if (info.dist < t * MinimumDistanceRatio) {
            result.hit = true;
            result.normal = sdfNormal(position_world);
            result.position = position_world;
            result.uv = info.uv;
            result.steps = it + 1;
            result.travelled = t;

            return result;
        }

        t += info.dist;
    }
    
    result.hit = false;
    result.steps = MaximumSteps;
    result.travelled = t;
    return result;
}

void main() {
    // mat4 T_shape_camera = T_shape_world * T_world_camera;
    // vec3 light_shape = (T_shape_world * inverse(T_light_world))[3].xyz;

    mat3 R_shape_camera = mat3(
        T_shape_camera[0].xyz,
        T_shape_camera[1].xyz,
        T_shape_camera[2].xyz
    );
    vec3 t_shape_camera = T_shape_camera[3].xyz;
    vec3 ray_camera = normalize(inverse(K) * vec3(gl_FragCoord.xy, 1.0));
    SDFHit result = raymarch(t_shape_camera, R_shape_camera * ray_camera);

    if (result.hit) {
        vec4 textureColor = texture(objectTexture, result.uv);
        vec3 light_direction = normalize(light_shape);
        float intensity = max(1, dot(light_direction, result.normal));


        // If it's Matte, then no lighting affects it. It's always bright
        if (isMatte != 0) {
            intensity = 1;
        }

        // SDFHit shadow = raymarch(result.position - light_world * MinimumDistance * 100, -light_world);

        FragColor = textureColor * max(0, intensity);


        // FragColor = vec4(light_direction, 1);
        gl_FragDepth = length(t_shape_camera - result.position) / MaximumDistance;
        // if (result.steps > 10)
            // FragColor *= result.steps / float(MaximumSteps);
        // FragColor *= 1 - int(shadow.hit);
        // FragColor = vec4((1 + result.normal) / 2, 1);
        // FragColor = vec4(shadow.hit, shadow.hit, shadow.hit, 1);
        // FragColor=vec4(0, result.uv.x, 1,1);
        // FragColor=vec4(0,1,0,1);
        // FragColor=vec4(result.steps/MaximumSteps,0,0,1);
    } else {
        FragColor = vec4(0,0,0,1);
        gl_FragDepth = 1.0;
    }

}
