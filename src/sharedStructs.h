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
    uint mouseSelectPosition[2];
    uint hdr;
    float frameTime;
    float padding0[2];
#else
    float4x4 cameraViewMat;
    float4x4 cameraProjMat;
    uint2 resolution;
    uint2 mouseSelectPosition;
    uint hdr;
    float frameTime;
    float2 padding0;
#endif
};

enum WorldObjectType : uint {
    WorldObjectTypeNone,
    WorldObjectTypePlayer,
    WorldObjectTypeSkybox,
    WorldObjectTypeStaticObject,
    WorldObjectTypeDynamicObject,
};

struct TLASInstanceInfo {
#ifdef __cplusplus
    WorldObjectType objectType;
    uint objectIndex;
    uint selected;
    uint skinJointsDescriptor;
    uint blasGeometriesOffset;
    uint padding[3];
#else
    WorldObjectType objectType;
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

struct ReadBackBuffer {
#ifdef __cplusplus
    uint mouseSelectInstanceIndex;
    float padding0[3];
#else
    uint mouseSelectInstanceIndex;
    float padding0[3];
#endif
};

struct CollisionQuery {
#ifdef __cplusplus
    float3 rayPosition;
    float rayMin;
    float3 rayDir;
    float rayMax;
#else
    float3 rayPosition;
    float rayMin;
    float3 rayDir;
    float rayMax;
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
