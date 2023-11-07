#include "shared.hlsli"
#include "../sceneStructs.h"

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
    uint2 imageSize = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    float2 pixelCoord = ((float2(pixelIndex) + 0.5f) / float2(imageSize)) * 2.0f - 1.0f;

    RENDER_TEXTURE_UAV_DESCRIPTOR(renderTexture);
    RENDER_INFO_DESCRIPTOR(renderInfo);
    BVH_DESCRIPTOR(bvh);
    TLAS_INSTANCES_INFOS_DESCRIPTOR(instanceInfos);
    SKYBOX_TEXTURE_DESCRIPTOR(skyboxTexture);

    RayPayload primaryRayPayload = (RayPayload) 0;
    RayDesc primaryRay = generatePinholeCameraRay(pixelCoord, renderInfo.cameraViewMat, renderInfo.cameraProjMat);
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, primaryRay, primaryRayPayload);
    if (primaryRayPayload.position.x == -FLT_MAX) {
        float3 viewDir = primaryRay.Direction;
        float theta = atan2(viewDir.z, viewDir.x);
        float phi = acos(viewDir.y);
        float3 skyboxColor = skyboxTexture.SampleLevel(bilinearSampler, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
        renderTexture[pixelIndex] = float4(skyboxColor / 10, 0);
    }
    else {
        if (primaryRayPayload.edge) {
            renderTexture[pixelIndex] = float4(0, 1, 0, 0);
        }
        else {
            float3 lightDir = normalize(float3(1, 1, 1));
            float ndotl = dot(primaryRayPayload.normal, lightDir);
            renderTexture[pixelIndex] = float4(primaryRayPayload.diffuse * ndotl, 0);
        }
    }
    //float3 lightDir = normalize(float3(1, 1, 1));
    //float3 lightColor = float3(1, 1, 1);
    //float3 ambient = float3(0.025, 0.025, 0.025);
    //float d = saturate(dot(primaryRayPayload.normal, lightDir));
    //renderTexture[pixelIndex] = d;

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

[shader("miss")]
void primaryRayMiss(inout RayPayload payload) {
    payload.position.x = -FLT_MAX;
}

[shader("closesthit")]
void primaryRayClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    TLAS_INSTANCES_INFOS_DESCRIPTOR(instancesInfos);
    BLAS_GEOMETRIES_INFOS_DESCRIPTOR(blasGeometriesInfos);
    TLASInstanceInfo instanceInfo = instancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[instanceInfo.blasGeometriesOffset + GeometryIndex()];

    uint verticesDescriptorIndex = InstanceID() + GeometryIndex() * 3;
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex + 1)];
    Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex + 2)];
    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex vertex0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex vertex1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex vertex2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];

    float3x4 transform = ObjectToWorld3x4();
    float3x3 normalTransform = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    payload.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.normal = barycentricsLerp(trigAttribs.barycentrics, vertex0.normal, vertex1.normal, vertex2.normal);
    payload.normal = normalize(mul(normalTransform, payload.normal));
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, vertex0.uv, vertex1.uv, vertex2.uv);
    payload.diffuse = baseColorTexture.SampleLevel(bilinearSampler, uv, 0);
    payload.diffuse *= blasGeometryInfo.baseColorFactor.xyz;
    payload.edge = false;
    
    if (instanceInfo.selected) {
        payload.edge = barycentricsOnEdge(trigAttribs.barycentrics, 0.02);
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
