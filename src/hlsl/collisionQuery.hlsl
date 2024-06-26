#include "shared.h"

GlobalRootSignature
globalRootSig = { 
    "RootFlags(0),"
    "SRV(t0), SRV(t1), UAV(u0)",
};

RaytracingAccelerationStructure bvh : register(t0);
StructuredBuffer<CollisionQuery> collisionQueries : register(t1);
RWStructuredBuffer<CollisionQueryResult> collisionQueryResults : register(u0);

RaytracingPipelineConfig pipelineConfig = {1};
RaytracingShaderConfig shaderConfig = {16, 8};
TriangleHitGroup hitGroup = {"", "closestHit"};

struct RayPayload {
    CollisionQueryResult collisionQueryResult;
};

[shader("raygeneration")]
void rayGen() {
    uint collisionQueryIndex = DispatchRaysIndex().x;
    CollisionQuery query = collisionQueries[collisionQueryIndex];
    RayPayload rayPayload;
    TraceRay(bvh, RAY_FLAG_NONE, query.instanceInclusionMask, 0, 0, 0, query.rayDesc, rayPayload);
    collisionQueryResults[collisionQueryIndex] = rayPayload.collisionQueryResult;
}

[shader("miss")]
void miss(inout RayPayload payload) {
    payload.collisionQueryResult.instanceIndex = UINT_MAX;
}

[shader("closesthit")]
void closestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.collisionQueryResult.instanceIndex = InstanceIndex();
    payload.collisionQueryResult.distance = WorldRayDirection() * RayTCurrent();
}
