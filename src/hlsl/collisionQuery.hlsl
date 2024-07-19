#include "shared.h"

GlobalRootSignature
globalRootSig = { 
    "RootFlags(0),"
    "SRV(t0), SRV(t1), SRV(t2), UAV(u0)",
};

RaytracingAccelerationStructure bvh : register(t0);
StructuredBuffer<CollisionQuery> collisionQueries : register(t1);
StructuredBuffer<BLASInstanceInfo> blasInstanceInfos : register(t2);
RWStructuredBuffer<CollisionQueryResult> collisionQueryResults : register(u0);


RaytracingPipelineConfig pipelineConfig = {1};
RaytracingShaderConfig shaderConfig = {24, 8};
TriangleHitGroup hitGroup = {"", "closestHit"};

struct RayPayload {
    float3 distance;
    uint objectType;
    uint objectIndex;
    uint meshNodeIndex;
};

[shader("raygeneration")]
void rayGen() {
    uint collisionQueryIndex = DispatchRaysIndex().x;
    CollisionQuery query = collisionQueries[collisionQueryIndex];
    RayPayload payload;
    TraceRay(bvh, RAY_FLAG_NONE, query.instanceInclusionMask, 0, 0, 0, query.rayDesc, payload);
    CollisionQueryResult result;
    result.distance = payload.distance;
    result.objectType = (ObjectType)payload.objectType;
    result.objectIndex = payload.objectIndex;
    result.meshNodeIndex = payload.meshNodeIndex;
    collisionQueryResults[collisionQueryIndex] = result;
}

[shader("miss")]
void miss(inout RayPayload payload) {
    payload.objectType = ObjectTypeNone;
    payload.objectIndex = UINT_MAX;
    payload.meshNodeIndex = UINT_MAX;
}

[shader("closesthit")]
void closestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo info = blasInstanceInfos[InstanceIndex()];
    payload.distance = WorldRayDirection() * RayTCurrent();
    payload.objectType = info.objectType;
    payload.objectIndex = info.objectIndex;
    payload.meshNodeIndex = info.meshNodeIndex;
}
