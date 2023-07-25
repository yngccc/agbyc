struct Vertex {
#ifdef __cplusplus
	float position[3];
	float normal[3];
	float uv[2];
#else
	float3 position;
	float3 normal;
	float2 uv;
#endif
};

struct Material {
#ifdef __cplusplus
	float diffuse[3];
	int diffuseTextureIndex;
	float emissive[3];
	int emissiveTextureIndex;
	//int normalTextureIndex;
#else
	float3 diffuse;
	int diffuseTextureIndex;
	float3 emissive;
	int emissiveTextureIndex;
	//int normalTextureIndex;
#endif
};

#define DIRECTIONAL_LIGHT 0
#define POINT_LIGHT 1

struct Light {
#ifdef __cplusplus
	int type;
	float position[3] = { 0, 0, 0 };
	float direction[3] = { 0, 1, 0 };
	float color[3] = { 1, 1, 1 };
#else
	int type;
	float3 position;
	float3 direction;
	float3 color;
#endif
};

struct RenderInfo {
#ifdef __cplusplus
	XMMATRIX cameraViewMat;
	XMMATRIX cameraProjMat;
	uint resolution[2];
	uint mouseSelectPosition[2];
	uint hdr; float frameTime; float padding0[2];
	float3 playerPosition; float padding1;
	float3 playerVelocity; float playerVelocityMax;
	float3 playerAcceleration; float padding2;
#else
	float4x4 cameraViewMat;
	float4x4 cameraProjMat;
	uint2 resolution;
	uint2 mouseSelectPosition;
	uint hdr; float frameTime; float2 padding0;
	float3 playerPosition; float padding1;
	float3 playerVelocity; float playerVelocityMax;
	float3 playerAcceleration; float padding2;
#endif
};

enum SceneObjectType : uint {
	SceneObjectTypeNone,
	SceneObjectTypePlayer,
	SceneObjectTypeTerrain,
	SceneObjectTypeSkybox,
	SceneObjectTypeBuilding,
};

struct TLASInstanceInfo {
#ifdef __cplusplus
	SceneObjectType objectType;
	uint objectIndex;
	uint selected;
#else
	SceneObjectType objectType;
	uint objectIndex;
	uint selected;
#endif
};

struct ReadBackBuffer {
#ifdef __cplusplus
	uint mouseSelectInstanceIndex;
	float3 playerPosition; float padding0;
	float3 playerVelocity; float padding1;
	float3 playerAcceleration; float padding2;
#else
	uint mouseSelectInstanceIndex;
	float3 playerPosition; float padding0;
	float3 playerVelocity; float padding1;
	float3 playerAcceleration; float padding2;
#endif
};

struct CollisionQuery {
#ifdef __cplusplus
	float3 rayPosition; float rayMin;
	float3 rayDir; float rayMax;
#else
	float3 rayPosition; float rayMin;
	float3 rayDir; float rayMax;
#endif
};

struct CollisionQueryResult {
#ifdef __cplusplus
	float3 distance; uint instanceIndex;
#else
	float3 distance; uint instanceIndex;
#endif
};
