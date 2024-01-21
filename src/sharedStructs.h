struct Vertex {
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
    uint diffuseTextureIndex;
    float3 emissive;
    uint emissiveTextureIndex;
    // uint normalTextureIndex;
#else
    float3 diffuse;
    uint diffuseTextureIndex;
    float3 emissive;
    uint emissiveTextureIndex;
    // uint normalTextureIndex;
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
    XMMATRIX cameraProjMat;
    uint resolution[2];
    uint hdr;
    float frameTime;
#else
    float4x4 cameraViewMat;
    float4x4 cameraProjMat;
    uint2 resolution;
    uint hdr;
    float frameTime;
#endif
};

enum ObjectType : uint {
    ObjectTypeNone,
    ObjectTypePlayer,
    ObjectTypeSkybox,
    ObjectTypeStaticObject,
    ObjectTypeDynamicObject,
};

struct TLASInstanceInfo {
#ifdef __cplusplus
    ObjectType objectType;
    uint objectIndex;
    uint selected;
    uint skinJointsDescriptor;
    uint blasGeometriesOffset;
    uint padding[3];
#else
    ObjectType objectType;
    uint objectIndex;
    uint selected;
    uint skinJointsDescriptor;
    uint blasGeometriesOffset;
    uint padding[3];
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
    uint instanceInclusionMask;
    uint padding[3];
#else
    RayDesc rayDesc;
    uint instanceInclusionMask;
    uint padding[3];
#endif
};

struct CollisionQueryResult {
#ifdef __cplusplus
    float3 distance;
    uint instanceIndex;
#else
    float3 distance;
    uint instanceIndex;
#endif
};
