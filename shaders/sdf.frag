#version 410 core

const float PI = 3.1415926535897932384626433832795;

layout(std140) uniform TreeShapeTypes {
    int treeShapeTypes[1000];
};

layout(std140) uniform TreeParametersIndex {
    int treeParametersIndex[1000];
};

layout(std140) uniform ShapeParameters {
    vec4 shapeParameters[1000];
};

uniform sampler2D sdfTexture;


out vec4 FragColor;

uniform vec2 resolution;
uniform int numElements;
uniform mat4 T_world_camera;
uniform mat3 K;
uniform vec3 light_world;

const int MaximumSteps = 64;
const float MaximumDistance = 1e6;

struct SDInfo {
    int size;
    float dist;
    vec2 uv;
};

struct SDFHit {
    bool hit;
    int treeIndex;
    vec3 normal;
    vec3 position;
    vec2 uv;
};

SDInfo signedDistance(vec3 position_world, int treeIndex) {
    SDInfo info;

    int shapeType = treeShapeTypes[treeIndex];
    int pi = treeParametersIndex[treeIndex];

    if (shapeType == 1) {
        mat4 T_shape_world = mat4(
            shapeParameters[pi + 0],
            shapeParameters[pi + 1],
            shapeParameters[pi + 2],
            shapeParameters[pi + 3]
        );
        float radius = shapeParameters[pi + 4].x;
        vec3 position_shape = (T_shape_world * vec4(position_world, 1.0)).xyz;

        info.size = 1;
        info.dist = length(position_shape) - radius;
        vec3 position_dir = normalize(position_shape);
        info.uv = vec2(
            (atan(position_dir.y, position_dir.x) + PI) / (2 * PI),
            acos(position_dir.z) / PI
        );
    } else if (shapeType == 2) {
        mat4 T_shape_world = mat4(
            shapeParameters[pi + 0],
            shapeParameters[pi + 1],
            shapeParameters[pi + 2],
            shapeParameters[pi + 3]
        );
        float radius = shapeParameters[pi + 4].x;
        float height = shapeParameters[pi + 4].y;
        vec3 position_shape = (T_shape_world * vec4(position_world, 1.0)).xyz;
        info.size = 1;
        info.dist = max(length(position_shape.xz) - radius, abs(position_shape.y) - height);

        vec2 position_dir_xz = normalize(position_shape.xz);
        info.uv = vec2(
            (atan(position_dir_xz.y, position_dir_xz.x) + PI) / (2 * PI),
            length(position_shape.xz) / radius
        );
    } else {
        info.size = 0;
        info.dist = 0;
    }

    return info;
}

vec3 sdfNormal(vec3 position_world, int treeIndex) {
    float eps = 1e-6;

    float fx = signedDistance(position_world + vec3(eps, 0, 0), treeIndex).dist;
    float fy = signedDistance(position_world + vec3(0, eps, 0), treeIndex).dist;
    float fz = signedDistance(position_world + vec3(0, 0, eps), treeIndex).dist;

    float bx = signedDistance(position_world - vec3(eps, 0, 0), treeIndex).dist;
    float by = signedDistance(position_world - vec3(0, eps, 0), treeIndex).dist;
    float bz = signedDistance(position_world - vec3(0, 0, eps), treeIndex).dist;

    return (vec3(fx, fy, fz) - vec3(bx, by, bz)) / (2 * eps);
}

SDFHit raymarch(vec3 camera_world, vec3 direction) {
    float t = 0.0;
    for (int it = 0; it < MaximumSteps && t < MaximumDistance; ++it) {
        vec3 position_world = camera_world + direction * t;

        SDInfo start = signedDistance(position_world, 0);
        int i = start.size;
        float closestDistance = start.dist;
        int closestObjectIndex = 0;
        vec2 uv = start.uv;
        while (i < numElements) {
            SDInfo info = signedDistance(position_world, i);
            i += info.size;
            if (closestDistance < info.dist) {
                closestDistance = info.dist;
                closestObjectIndex = i;
                uv = info.uv;
            }
        }

        if (closestDistance < 1e-3) {
            SDFHit result;
            result.hit = true;
            result.normal = sdfNormal(position_world, closestObjectIndex);
            result.position = position_world;
            result.uv = uv;
            return result;
        }

        t += closestDistance;
    }
    
    SDFHit result;
    result.hit = false;
    return result;
}

void main() {    
    mat3 R_world_camera = mat3(
        T_world_camera[0].xyz,
        T_world_camera[1].xyz,
        T_world_camera[2].xyz
    );
    vec3 t_world_camera = T_world_camera[3].xyz;
    vec3 ray_camera = normalize(inverse(K) * vec3(gl_FragCoord.xy, 1.0));

    SDFHit result = raymarch(t_world_camera, R_world_camera * ray_camera);
    if (result.hit) {
        FragColor = vec4(result.uv, 0, 1.0);
        vec4 textureColor = texture(sdfTexture, result.uv);
        float intensity = max(0, dot(light_world, result.normal));

        FragColor = textureColor * max(intensity, .05);
        // FragColor = vec4(1,1,1,1);
        gl_FragDepth = length(t_world_camera - result.position);
    } else {
        FragColor = vec4(0,0,0,1);
        gl_FragDepth = MaximumDistance;
    }
}
