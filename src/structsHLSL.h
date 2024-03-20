// structs shared between cpp and hlsl

struct Vertex {
#ifdef __cplusplus
    float3 position;
    float3 normal;
    float4 tangent;  // bitangent = cross(normal, tangent.xyz) * tangent.w
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

struct Material {
#ifdef __cplusplus
    float3 diffuse;
    uint32 diffuseTextureIndex;
    float3 emissive;
    uint32 emissiveTextureIndex;
    // uint32 normalTextureIndex;
#else
    float3 diffuse;
    uint diffuseTextureIndex;
    float3 emissive;
    uint emissiveTextureIndex;
    // uint32 normalTextureIndex;
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
    XMMATRIX cameraViewMat;
    XMMATRIX cameraViewMatInverseTranspose;
    XMMATRIX cameraProjectMat;
    // XMMATRIX cameraViewProjectMat;
    // XMMATRIX cameraProjectViewInverseMat;
#else
    float4x4 cameraViewMat;
    float4x4 cameraViewMatInverseTranspose;
    float4x4 cameraProjectMat;
    // float4x4 cameraViewProjectMat;
    // float4x4 cameraViewProjectInverseMat;
#endif
};

enum ObjectType : uint {
    ObjectTypeNone = 0x01,
    ObjectTypePlayer = 0x01 << 1,
    ObjectTypeSkybox = 0x01 << 2,
    ObjectTypeGameObject = 0x01 << 3,
};

enum TLASInstanceFlag : uint {
    TLASInstanceFlagSelected = 0x01,
    TLASInstanceFlagReferencePlane = 0x01 << 1,
};

struct TLASInstanceInfo {
#ifdef __cplusplus
    ObjectType objectType;
    uint32 objectIndex;
    uint32 flags;
    uint32 blasGeometriesOffset;
#else
    ObjectType objectType;
    uint objectIndex;
    uint flags;
    uint blasGeometriesOffset;
#endif
};

struct BLASGeometryInfo {
#ifdef __cplusplus
    float4 baseColorFactor;
#else
    float4 baseColorFactor;
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
    uint32 padding[3];
#else
    RayDesc rayDesc;
    uint instanceInclusionMask;
    uint padding[3];
#endif
};

struct CollisionQueryResult {
#ifdef __cplusplus
    float3 distance;
    uint32 instanceIndex;
#else
    float3 distance;
    uint instanceIndex;
#endif
};

enum CompositeFlag : uint {
    CompositeFlagHDR = 0x01,
    CompositeFlagDirectWrite = 0x01 << 1,
};

struct ShapeCircle {
#ifdef __cplusplus
    float2 center;
    float radius;
    float padding;
#else
    float2 center;
    float radius;
    float padding;
#endif
};

struct ShapeLine {
#ifdef __cplusplus
    float2 p0;
    float2 p1;
    float thickness;
    float3 padding;
#else
    float2 p0;
    float2 p1;
    float thickness;
    float3 padding;
#endif
};