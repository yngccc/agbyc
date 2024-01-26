#include "shared.hlsli"
#include "../structsHLSL.h"

GlobalRootSignature globalRootSig = { "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)" };
RaytracingPipelineConfig pipelineConfig = { 1 };
RaytracingShaderConfig shaderConfig = { 16, 8 };
TriangleHitGroup hitGroup = { "", "closestHit" };

struct RayPayload {
    CollisionQueryResult collisionQueryResult;
};

[shader("raygeneration")]
void rayGen() {
    BVH_DESCRIPTOR(bvh);
    COLLISION_QUERIES_DESCRIPTOR(collisionQueries);
    COLLISION_QUERY_RESULTS_DESCRIPTOR(collisionQueryResults);
    
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
