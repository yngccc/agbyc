struct Vertex {
#ifdef __cplusplus
    float3 position;
    float3 normal;
    float2 uv;
    ushort4 joints;
    float4 jointWeights;
#else
    float3 position;
    float3 normal;
    float2 uv;
    uint2 joints;
    float4 jointWeights;
#endif
};

struct Joint {
#ifdef __cplusplus
    DirectX::XMMATRIX globalTransform;
    DirectX::XMMATRIX inverseBindTransform;
#else
    float4x4 globalTransform;
    float4x4 inverseBindTransform;
#endif
};

struct Material {
#ifdef __cplusplus
    float3 diffuse;
    int diffuseTextureIndex;
    float3 emissive;
    int emissiveTextureIndex;
    // int normalTextureIndex;
#else
    float3 diffuse;
    int diffuseTextureIndex;
    float3 emissive;
    int emissiveTextureIndex;
    // int normalTextureIndex;
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
    DirectX::XMMATRIX cameraViewMat;
    DirectX::XMMATRIX cameraProjMat;
    uint32 resolution[2];
    uint32 mouseSelectPosition[2];
    uint32 hdr;
    float frameTime;
    float padding0[2];
    float3 playerPosition;
    float padding1;
    float3 playerVelocity;
    float playerVelocityMax;
    float3 playerAcceleration;
    float padding2;
#else
    float4x4 cameraViewMat;
    float4x4 cameraProjMat;
    uint2 resolution;
    uint2 mouseSelectPosition;
    uint hdr;
    float frameTime;
    float2 padding0;
    float3 playerPosition;
    float padding1;
    float3 playerVelocity;
    float playerVelocityMax;
    float3 playerAcceleration;
    float padding2;
#endif
};

enum SceneObjectType : uint {
    SceneObjectTypeNone,
    SceneObjectTypePlayer,
    SceneObjectTypeSkybox,
    SceneObjectTypeStaticObject,
};

struct TLASInstanceInfo {
#ifdef __cplusplus
    SceneObjectType objectType;
    uint32 objectIndex;
    uint32 selected;
    uint32 skinJointsDescriptor;
#else
    SceneObjectType objectType;
    uint objectIndex;
    uint selected;
    uint skinJointsDescriptor;
#endif
};

struct ReadBackBuffer {
#ifdef __cplusplus
    uint32 mouseSelectInstanceIndex;
    float3 playerPosition;
    float padding0;
    float3 playerVelocity;
    float padding1;
    float3 playerAcceleration;
    float padding2;
#else
    uint mouseSelectInstanceIndex;
    float3 playerPosition;
    float padding0;
    float3 playerVelocity;
    float padding1;
    float3 playerAcceleration;
    float padding2;
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
    uint32 instanceIndex;
#else
    float3 distance;
    uint instanceIndex;
#endif
};
