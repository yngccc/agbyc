#include "shared.hlsli"
#include "../sharedStructs.h"

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
    } else {
        float2 pixelCoord = ((float2(renderInfo.mouseSelectPosition) + 0.5) / float2(renderInfo.resolution)) * 2.0 - 1.0;
        RayDesc ray = generatePinholeCameraRay(pixelCoord, renderInfo.cameraViewMat, renderInfo.cameraProjMat);
        RayPayload rayPayload;
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, rayPayload);
        readBackBuffer[0].mouseSelectInstanceIndex = rayPayload.instanceIndex;
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
