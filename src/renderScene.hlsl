#include "shared.hlsli"
#include "sceneStructs.h"

GlobalRootSignature globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
	"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"
};

sampler bilinearSampler : register(s0);

RaytracingPipelineConfig pipelineConfig = { 3 };

RaytracingShaderConfig shaderConfig = { 52, 8 };

TriangleHitGroup primaryRayHitGroup = { "", "primaryRayClosestHit" };

TriangleHitGroup secondaryRayHitGroup = { "", "secondaryRayClosestHit" };

struct RayPayload {
    bool edge;
    float3 position;
    float3 normal;
    float3 diffuse;
    float3 emissive;
};

[shader("raygeneration")]
void rayGen() {
    uint2 resolution = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    float2 pixelCoord = ((float2(pixelIndex) + 0.5) / resolution) * 2.0 - 1.0;

    RWTexture2D<float3> renderTexture = RENDER_TEXTURE_DESCRIPTOR;
    RWTexture2D<float3> accumulationRenderTexture = ACCUMULATION_RENDER_TEXTURE_DESCRIPTOR;
    ConstantBuffer<RenderInfo> renderInfo = RENDER_INFO_DESCRIPTOR;
    RaytracingAccelerationStructure BVH = BVH_DESCRIPTOR;
    StructuredBuffer<TLASInstanceInfo> instanceInfos = TLAS_INSTANCE_INFOS_DESCRIPTOR;
    Texture2D<float3> skyboxTexture = SKYBOX_TEXTURE_DESCRIPTOR;

    float aspectRatio = renderInfo.cameraProjMat[1][1] / renderInfo.cameraProjMat[0][0];
    float tanHalfFovY = 1.0 / renderInfo.cameraProjMat[1][1];

    RayPayload rayPayload = (RayPayload)0;
    RayDesc primaryRayDesc;
    primaryRayDesc.Origin = renderInfo.cameraViewMat[3].xyz;
    primaryRayDesc.TMin = 0;
    primaryRayDesc.TMax = 1000;
    primaryRayDesc.Direction = normalize(
		(pixelCoord.x * renderInfo.cameraViewMat[0].xyz * tanHalfFovY * aspectRatio) -
		(pixelCoord.y * renderInfo.cameraViewMat[1].xyz * tanHalfFovY) +
		renderInfo.cameraViewMat[2].xyz
	);
    TraceRay(BVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, primaryRayDesc, rayPayload);

    if (rayPayload.position.x == -FLT_MAX) {
        float3 viewDir = primaryRayDesc.Direction;
        float theta = atan2(viewDir.z, viewDir.x);
        float phi = acos(viewDir.y);
        float3 skyboxColor = skyboxTexture.SampleLevel(bilinearSampler, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
        renderTexture[pixelIndex] = skyboxColor / 10;
    }
    else {
        if (rayPayload.edge) {
            renderTexture[pixelIndex] = float3(0, 1, 0);
        }
        else {
            float3 lightDir = normalize(float3(1, 1, 1));
            float3 lightColor = float3(1, 1, 1);
            float3 ambient = float3(0.025, 0.025, 0.025);
            float d = saturate(dot(rayPayload.normal, lightDir));
            renderTexture[pixelIndex] = d;
        }

        //RayDesc shadowRayDesc;
        //shadowRayDesc.Origin = rayPayload.position;
        //shadowRayDesc.Origin += rayPayload.normal * 0.005;
        //shadowRayDesc.TMin = 0;
        //shadowRayDesc.TMax = 1000;
        //shadowRayDesc.Direction = lightDir;
        //TraceRay(BVH, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRayDesc, rayPayload);
        //if (rayPayload.position.x == -FLT_MAX) {
        //    renderTexture[pixelIndex] = saturate(lightColor * d + ambient);
        //}
        //else {
        //    renderTexture[pixelIndex] = ambient;
        //}
    }
}

[shader("miss")]
void primaryRayMiss(inout RayPayload payload) {
    payload.position.x = -FLT_MAX;
}

[shader("closesthit")]
void primaryRayClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    uint verticesDescriptorIndex = InstanceID() + GeometryIndex() * 2;
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex + 1)];
    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex vertex0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex vertex1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex vertex2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
    float3x4 transform = ObjectToWorld3x4();
    float3x3 normalTransform = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    payload.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.normal = barycentricsLerp(trigAttribs.barycentrics, vertex0.normal, vertex1.normal, vertex2.normal);
    payload.normal = normalize(mul(normalTransform, payload.normal));
    payload.edge = false;
    
    StructuredBuffer<TLASInstanceInfo> instanceInfoss = ResourceDescriptorHeap[4];
    TLASInstanceInfo instanceInfo = instanceInfoss[InstanceIndex()];
    if (instanceInfo.selected) {
        if (trigAttribs.barycentrics.x < 0.02) {
            payload.edge = true;
        }
        else if (trigAttribs.barycentrics.y < 0.02) {
            payload.edge = true;
        }
        else if ((1 - trigAttribs.barycentrics.x - trigAttribs.barycentrics.y) < 0.02) {
            payload.edge = true;
        }
    }
}

[shader("miss")]
void secondaryRayMiss(inout RayPayload payload) {
    payload.position.x = -FLT_MAX;
}

[shader("closesthit")]
void secondaryRayClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.position.x = 0;
}