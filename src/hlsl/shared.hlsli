#define CAMERA_Z_MIN 0.01f
#define CAMERA_Z_MAX 1000.0f

#ifndef __cplusplus

#define PI 3.14159265358979323846f
#define TWO_PI (2.0f * PI)
#define ONE_OVER_PI (1.0f / PI)
#define ONE_OVER_TWO_PI (1.0f / TWO_PI)
#define FLT_MAX 3.402823466e+38F
#define UINT_MAX 0xffffffff
#define RADIAN(d) (d * (PI / 180.0f))
#define DEGREE(r) (r * (180.0f / PI))

#define HDR_SCALE_FACTOR (80.0f / 10000.0f)

#define USE_PCG 0
#if USE_PCG
typedef uint4 RngState;
#else
typedef uint RngState;
#endif

uint4 pcg4d(uint4 v) {
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v = v ^ (v >> 16u);

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

uint xorshift(inout uint rngState) {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}

uint jenkinsHash(uint x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

float uintToFloat(uint x) {
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

#if USE_PCG
RngStateType initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber) {
	return RngStateType(pixelCoords.xy, frameNumber, 0); //< Seed for PCG uses a sequential sample number in 4th channel, which increments on every RNG call and starts from 0
}

float rand(inout RngStateType rngState) {
	rngState.w++; //< Increment sample index
	return uintToFloat(pcg4d(rngState).x);
}
#else
RngState initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber) {
    RngState seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
    return jenkinsHash(seed);
}

float rand(inout RngState rngState) {
    return uintToFloat(xorshift(rngState));
}
#endif

float srgbToLinear(float e) {
    if (e <= 0.04045f)
        return e / 12.92f;
    return pow((e + 0.055f) / 1.055f, 2.4f);
}

float3 srgbToLinear(float3 srgb) {
    return float3(srgbToLinear(srgb.x), srgbToLinear(srgb.y), srgbToLinear(srgb.z));
}

float linearToSRGB(float e) {
    if (e <= 0.0031308f)
        return e * 12.92f;
    return 1.055f * pow(e, 1.0f / 2.4f) - 0.055f;
}

float3 linearToSRGB(float3 rgb) {
    return float3(linearToSRGB(rgb.x), linearToSRGB(rgb.y), linearToSRGB(rgb.z));
}

float linearToPQ(float e) {
    const float m1 = 0.1593017578125f;
    const float m2 = 78.84375f;
    const float c1 = 0.8359375f;
    const float c2 = 18.8515625f;
    const float c3 = 18.6875f;
    float ym1 = pow(e, m1);
    return pow((c1 + c2 * ym1) / (1.0 + c3 * ym1), m2);
}

float3 linearToPQ(float3 rgb) {
    return float3(linearToPQ(rgb.x), linearToPQ(rgb.y), linearToPQ(rgb.z));
}

float3 bt709To2020(float3 rgb) {
    const float3x3 mat = {
        0.6274f, 0.3293f, 0.0433f,
		0.0691f, 0.9195f, 0.0114f,
		0.0164f, 0.0880f, 0.8956f,
    };
    return mul(mat, rgb);
}

float luminance(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

float luminance_bt2020(float3 rgb) {
    return dot(rgb, float3(0.2627f, 0.6780f, 0.0593f));
}

uint packRGBA(float4 rgba) {
    return 0;
}

uint packRGB(float3 rgb) {
    return 0;
}

float4 unpackRGBA(uint color) {
    return float4(
        ((color >> 24) & 0xff) / 255.0f,
        ((color >> 16) & 0xff) / 255.0f,
        ((color >> 8) & 0xff) / 255.0f,
        (color & 0xff) / 255.0f
   );
}

float3 unpackRGB(uint color) {
    return float3(
        ((color >> 24) & 0xff) / 255.0f,
        ((color >> 16) & 0xff) / 255.0f,
        ((color >> 8) & 0xff) / 255.0f
    );
}

float3 hashIntToColor(int i) {
    uint hash = jenkinsHash(i);
    float r = ((hash >> 0) & 0xFF) / 255.0f;
    float g = ((hash >> 8) & 0xFF) / 255.0f;
    float b = ((hash >> 16) & 0xFF) / 255.0f;
    return float3(r, g, b);
}

float2 barycentricsLerp(float2 barycentrics, float2 vertAttrib0, float2 vertAttrib1, float2 vertAttrib2) {
    return vertAttrib0 + barycentrics.x * (vertAttrib1 - vertAttrib0) + barycentrics.y * (vertAttrib2 - vertAttrib0);
}

float3 barycentricsLerp(float2 barycentrics, float3 vertAttrib0, float3 vertAttrib1, float3 vertAttrib2) {
    return vertAttrib0 + barycentrics.x * (vertAttrib1 - vertAttrib0) + barycentrics.y * (vertAttrib2 - vertAttrib0);
}

bool barycentricsOnEdge(float2 barycentrics, float edgeThickness) {
    return (barycentrics.x < edgeThickness) || (barycentrics.y < edgeThickness) || ((1.0 - barycentrics.x - barycentrics.y) < edgeThickness);
}

float2 equirectangularMapping(float3 dir) {
    float theta = atan2(dir.z, dir.x);
    float phi = acos(dir.y);
    return float2((theta + PI) / (2.0f * PI), phi / PI);
}

float2 octWrap(float2 v) {
    return float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

float2 encodeNormalOctahedron(float3 n) {
    float2 p = float2(n.x, n.y) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.0f) ? octWrap(p) : p;
    return p;
}

float3 decodeNormalOctahedron(float2 p) {
    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float2 tmp = (n.z < 0.0f) ? octWrap(float2(n.x, n.y)) : float2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

float3 offsetRay(const float3 p, const float3 n) {
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(
		asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
		asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
		asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
		abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
		abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

float3 offsetRayShadow(float3 p, float3 a, float3 b, float3 c, float3 na, float3 nb, float3 nc, float u, float v, float w) {
    float3 tmpu = p - a;
    float3 tmpv = p - b;
    float3 tmpw = p - c;
    float dotu = min(0.0, dot(tmpu, na));
    float dotv = min(0.0, dot(tmpv, nb));
    float dotw = min(0.0, dot(tmpw, nc));
    tmpu -= dotu * na;
    tmpv -= dotv * nb;
    tmpw -= dotw * nc;
    float3 pp = p + (u * tmpu) + (v * tmpv) + (w * tmpw);
    return pp;
}

float3 triangleGeometryNormal(float3 p0, float3 p1, float3 p2) {
    float3 edge20 = p2 - p0;
    //float3 edge21 = p2 - p1;
    float3 edge10 = p1 - p0;
    return normalize(cross(edge20, edge10));
}

//RayDesc cameraRayPinhole(float2 pixelCoord, float2 imageSize, float4x4 cameraRotationMat, float3 cameraPosition, float cameraFovVertical) {
//    pixelCoord -= imageSize / 2.0;
//    float tanHalfAngle = tan(cameraFovVertical / 2.0);
//    float aspectScale = imageSize.y / 2.0;
//    RayDesc ray;
//    ray.Origin = cameraPosition;
//    ray.TMin = CAMERA_Z_MIN;
//    ray.TMax = CAMERA_Z_MAX;
//    ray.Direction = normalize(float3(float2(pixelCoord.x, -pixelCoord.y) * tanHalfAngle / aspectScale, 1.0));
//    ray.Direction = mul(float3x3(cameraRotationMat[0].xyz, cameraRotationMat[1].xyz, cameraRotationMat[2].xyz), ray.Direction);
//    return ray;
//}

//RayDesc cameraRayPanini(float2 pixelCoord, float2 imageSize, float4x4 cameraRotationMat, float3 cameraPosition, float cameraFovVertical, float paniniDistance, float verticalCompression) {
//    pixelCoord -= imageSize / 2.0;
//    float halfFOV = cameraFovVertical / 2.0;
//    float halfPaniniFOV = atan2(sin(halfFOV), cos(halfFOV) + paniniDistance);
//    float2 hvPan = (tan(halfPaniniFOV)).xx * (1.0 + paniniDistance); 
//    hvPan *= pixelCoord / imageSize.y;
//    hvPan.x = atan(hvPan.x / (1.0 + paniniDistance));
//    float n = sin(hvPan.x) * paniniDistance;
//    float M = sqrt(1.0 - n * n) + paniniDistance * cos(hvPan.x);
//    float x = sin(hvPan.x) * M;
//    float z = (cos(hvPan.x) * M) - paniniDistance;
//    float S = (paniniDistance + 1) / (paniniDistance + z);
//    float y = lerp(hvPan.y / S, hvPan.y * z, verticalCompression);
//    RayDesc ray;
//    ray.Origin = cameraPosition;
//    ray.TMin = CAMERA_Z_MIN;
//    ray.TMax = CAMERA_Z_MAX;
//    ray.Direction = normalize(float3(x, -y, z));
//    ray.Direction = mul(float3x3(cameraRotationMat[0].xyz, cameraRotationMat[1].xyz, cameraRotationMat[2].xyz), ray.Direction);
//    return ray;
//}

RayDesc cameraRayPinhole(float2 pixelCoord, float2 imageResolution, float4x4 cameraViewMat, float4x4 cameraProjMat) {
    pixelCoord = (pixelCoord / imageResolution) * 2.0 - 1.0;
    RayDesc ray;
    ray.Origin = cameraViewMat[3].xyz;
    ray.TMin = CAMERA_Z_MIN;
    ray.TMax = CAMERA_Z_MAX;
    float aspect = cameraProjMat[1][1] / cameraProjMat[0][0];
    float tanHalfFovY = 1.0 / cameraProjMat[1][1];
    ray.Direction = normalize(
        (pixelCoord.x * cameraViewMat[0].xyz * tanHalfFovY * aspect) -
        (pixelCoord.y * cameraViewMat[1].xyz * tanHalfFovY) +
        cameraViewMat[2].xyz
    );
    return ray;
}

RayDesc cameraRayThinLens(float2 pixelCoord, float2 imageResolution, float4x4 cameraViewMat, float4x4 cameraProjMat, float focusDistance, float apertureSize, inout RngState rngState) {
    RayDesc ray = cameraRayPinhole(pixelCoord, imageResolution, cameraViewMat, cameraProjMat);
    float3 focalPoint = ray.Origin + ray.Direction * focusDistance;
    float2 apertureSample;
    {
        float2 u = float2(rand(rngState), rand(rngState));
        float a = sqrt(u.x);
        float b = TWO_PI * u.y;
        apertureSample = float2(a * cos(b), a * sin(b)) * apertureSize;
    }
    float3 rightVector = cameraViewMat[0].xyz;
    float3 upVector = cameraViewMat[1].xyz;
    ray.Origin = ray.Origin + rightVector * apertureSample.x + upVector * apertureSample.y;
    ray.Direction = normalize(focalPoint - ray.Origin);
    return ray;
}

void anisotropicEllipseAxes(float3 p, float3 f, float3 d, float rayConeRadiusAtIntersection,
                            float3 position0, float3 position1, float3 position2,
                            float2 txcoord0, float2 txcoord1, float2 txcoord2,
                            float2 interpolatedTexCoordsAtIntersection,
                            out float2 texGradient1, out float2 texGradient2) {
    float3 a1 = d - dot(f, d) * f;
    float3 p1 = a1 - dot(d, a1) * d;
    a1 *= rayConeRadiusAtIntersection / max(0.0001f, length(p1));
    
    float3 a2 = cross(f, a1);
    float3 p2 = a2 - dot(d, a2) * d;
    a2 *= rayConeRadiusAtIntersection / max(0.0001f, length(p2));
    
    float3 eP, delta = p - position0;
    float3 e1 = position1 - position0;
    float3 e2 = position2 - position0;
    float oneOverAreaTriangle = 1.0f / dot(f, cross(e1, e2));
    eP = delta + a1;
    float u1 = dot(f, cross(eP, e2)) * oneOverAreaTriangle;
    float v1 = dot(f, cross(e1, eP)) * oneOverAreaTriangle;
    texGradient1 = (1.0f - u1 - v1) * txcoord0 + u1 * txcoord1 + v1 * txcoord2 - interpolatedTexCoordsAtIntersection;
    eP = delta + a2;
    float u2 = dot(f, cross(eP, e2)) * oneOverAreaTriangle;
    float v2 = dot(f, cross(e1, eP)) * oneOverAreaTriangle;
    texGradient2 = (1.0f - u2 - v2) * txcoord0 + u2 * txcoord1 + v2 * txcoord2 - interpolatedTexCoordsAtIntersection;
}

#endif

struct Vertex {
#ifdef __cplusplus
    float3 position;
    float3 normal;
    float4 tangent; // bitangent = cross(normal, tangent.xyz) * tangent.w
    float2 uv;
    uint16_4 joints;
    float4 jointWeights;
#else
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint16_t joints[4];
    float4 jointWeights;
#endif
};

struct VertexSkinned {
#ifdef __cplusplus
    float3 position;
    float3 normal;
    float2 uv;
    uint16_4 joints;
    float4 jointWeights;
#else
    float3 position;
    float3 normal;
    float2 uv;
    uint16_t joints[4];
    float4 jointWeights;
#endif
};

struct Joint {
#ifdef __cplusplus
    DirectX::XMMATRIX transform;
    DirectX::XMMATRIX inverseBindTransform;
#else
    float4x4 globalTransform;
    float4x4 inverseBindTransform;
#endif
};

enum LightType : uint {
    LightTypeDirectional,
    LightTypePoint,
    LightTypeSpot,
    LightTypeSpherical
};

struct Light {
#ifdef __cplusplus
    LightType type;
    float3 intensity;
    float3 position;
    float3 direction;
#else
    LightType type;
    float3 intensity;
    float3 position;
    float3 direction;
#endif
};

struct RenderInfo {
#ifdef __cplusplus
    XMMatrix cameraViewMat;
    XMMatrix cameraViewMatInverseTranspose;
    XMMatrix cameraProjectMat;
    XMMatrix cameraViewProjectMat;
    XMMatrix cameraViewProjectMatInverse;
    XMMatrix cameraViewProjectMatPrevFrame;
    float cameraFovVertical;
    uint pathTracerAccumulationFrameCount;
    uint pathTracerDepthOfField;
#else
    float4x4 cameraViewMat;
    float4x4 cameraViewMatInverseTranspose;
    float4x4 cameraProjectMat;
    float4x4 cameraViewProjectMat;
    float4x4 cameraViewProjectMatInverse;
    float4x4 cameraViewProjectMatPrevFrame;
    float cameraFovVertical;
    uint pathTracerAccumulationFrameCount;
    uint pathTracerDepthOfField;
#endif
};

// also used as instanceMask in D3D12_RAYTRACING_INSTANCE_DESC
enum ObjectType : uint {
    ObjectTypeNone = 0x01,
    ObjectTypePlayer = 0x01 << 1,
    ObjectTypeGameObject = 0x01 << 2,
};

enum BLASInstanceFlag : uint {
    BLASInstanceFlagHighlightTriangleEdges = 0x01,
    BLASInstanceFlagForcedColor = 0x01 << 1,
};

struct BLASInstanceInfo {
#ifdef __cplusplus
    uint blasGeometriesOffset;
    uint flags;
    uint color;
    ObjectType objectType;
    uint objectIndex;
    uint meshNodeIndex;
    XMFloat3x3 transformNormalMat;
    XMFloat4x3 transformMatPrevFrame;
#else
    uint blasGeometriesOffset;
    uint flags;
    uint color;
    ObjectType objectType;
    uint objectIndex;
    uint meshNodeIndex;
    float3x3 transformNormalMat;
    float3x4 transformMatPrevFrame;
#endif
};

enum AlphaMode : uint {
    AlphaModeOpaque = 0,
    AlphaModeMask = 1,
    AlphaModeBlend = 2,
};

struct BLASGeometryInfo {
#ifdef __cplusplus
    uint descriptorsHeapOffset = 0;
    float4 baseColorFactor = {0.8f, 0.8f, 0.8f, 1.0f};
    AlphaMode alphaMode = AlphaModeOpaque;
    float alphaCutOff = 0.5f;
    float roughnessFactor = 1;
    float metallicFactor = 0;
    float3 emissiveFactor = {0, 0, 0};
#else
    uint descriptorsHeapOffset;
    float4 baseColorFactor;
    AlphaMode alphaMode;
    float alphaCutOff;
    float roughnessFactor;
    float metallicFactor;
    float3 emissiveFactor;
#endif
};

#ifdef __cplusplus
struct RayDesc {
    float3 origin;
    float min;
    float3 dir;
    float max;
};
#endif

struct CollisionQuery {
#ifdef __cplusplus
    RayDesc rayDesc;
    uint32 instanceInclusionMask;
#else
    RayDesc rayDesc;
    uint instanceInclusionMask;
#endif
};

struct CollisionQueryResult {
#ifdef __cplusplus
    float3 distance;
    ObjectType objectType;
    uint objectIndex;
    uint meshNodeIndex;
#else
    float3 distance;
    ObjectType objectType;
    uint objectIndex;
    uint meshNodeIndex;
#endif
};

enum CompositeFlag : uint {
    CompositeFlagHDR = 0x01,
};
