#include "shared.hlsli"
#include "sceneStructs.h"

GlobalRootSignature globalRootSig = { "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)" };

RaytracingPipelineConfig pipelineConfig = { 1 };

RaytracingShaderConfig shaderConfig = { 4, 8 };

TriangleHitGroup hitGroup = { "", "closestHit" };

struct RayPayload {
    uint instanceIndex;
};

[shader("raygeneration")]
void rayGen() {
    RENDER_INFO_DESCRIPTOR(renderInfo);
    BVH_DESCRIPTOR(bvh);
    READBACK_BUFFER_DESCRIPTOR(readBackBuffer);
	
    if (renderInfo.mouseSelectPosition.x == UINT_MAX) {
        readBackBuffer[0].mouseSelectInstanceIndex = UINT_MAX;
    }
    else {
        float2 pixelCoord = ((float2(renderInfo.mouseSelectPosition) + 0.5) / renderInfo.resolution) * 2.0 - 1.0;
        float aspectRatio = renderInfo.cameraProjMat[1][1] / renderInfo.cameraProjMat[0][0];
        float tanHalfFovY = 1.0 / renderInfo.cameraProjMat[1][1];

        RayPayload rayPayload;
        RayDesc ray;
        ray.Origin = renderInfo.cameraViewMat[3].xyz;
        ray.TMin = 0;
        ray.TMax = 1000;
        ray.Direction = normalize(
			(pixelCoord.x * renderInfo.cameraViewMat[0].xyz * tanHalfFovY * aspectRatio) -
			(pixelCoord.y * renderInfo.cameraViewMat[1].xyz * tanHalfFovY) +
			renderInfo.cameraViewMat[2].xyz
		);
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, rayPayload);
        readBackBuffer[0].mouseSelectInstanceIndex = rayPayload.instanceIndex;
    }
	{
        RayPayload rayPayload;
        RayDesc ray;
        ray.Origin = renderInfo.playerPosition;
        ray.TMin = 0.0;
        ray.TMax = length(renderInfo.playerVelocity) * renderInfo.frameTime;
        ray.Direction = renderInfo.playerVelocity;
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, rayPayload);
        if (rayPayload.instanceIndex == UINT_MAX) {
            readBackBuffer[0].playerPosition = renderInfo.playerPosition + renderInfo.playerVelocity * renderInfo.frameTime;
        }
        else {
            readBackBuffer[0].playerPosition = renderInfo.playerPosition;
        }
        readBackBuffer[0].playerVelocity = renderInfo.playerVelocity;
        readBackBuffer[0].playerAcceleration = renderInfo.playerAcceleration;
    }
}

[shader("miss")]
void miss(inout RayPayload payload) {
    payload.instanceIndex = UINT_MAX;
}

[shader("closesthit")]
void closestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.instanceIndex = InstanceIndex();
}
