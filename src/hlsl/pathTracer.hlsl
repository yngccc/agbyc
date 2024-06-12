#include "shared.h"

GlobalRootSignature
globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
    "CBV(b0), SRV(t0), SRV(t1), SRV(t2),"
    "DescriptorTable(UAV(u0), UAV(u1), SRV(t3)),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, mipLODBias = 0, maxAnisotropy = 16),"
};

ConstantBuffer<RenderInfo> renderInfo : register(b0);
RWTexture2D<float3> renderTexture : register(u0);
RWTexture2D<float3> renderAccumulationTexture : register(u1);
RaytracingAccelerationStructure bvh : register(t0);
StructuredBuffer<BLASInstanceInfo> blasInstancesInfos : register(t1);
StructuredBuffer<BLASGeometryInfo> blasGeometriesInfos : register(t2);
Texture2D<float3> skyboxTexture : register(t3);
sampler textureSampler : register(s0);

RaytracingPipelineConfig pipelineConfig = {1 /*MaxTraceRecursionDepth*/};
RaytracingShaderConfig shaderConfig = {40 /*UINT MaxPayloadSizeInBytes*/, 8 /*UINT MaxAttributeSizeInBytes*/};
TriangleHitGroup rayHitGroup = {"rayAnyHit", "rayClosestHit"};

struct RayPayload {
    float3 position;
    int materialID;
    float2 geometryNormalEncoded;
    float2 shadingNormalEncoded;
    float2 uv;
};

[shader("raygeneration")]
void rayGen() {
    uint2 resolution = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    uint rngState = initRNG(pixelIndex, resolution, renderInfo.accumulationFrameCount);
    float2 offset = float2(rand(rngState), rand(rngState));
    offset = lerp(float2(-0.5, -0.5), float2(0.5, 0.5), offset);
    float2 pixelCoord = ((float2(pixelIndex) + offset + 0.5) / float2(resolution)) * 2.0 - 1.0;
    RayDesc ray = pinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    RayPayload rayPayload = (RayPayload)0;
    float3 radian = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    for (int bounce = 0; bounce < 4; bounce++) {
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, rayPayload);
        if (rayPayload.materialID == -1) {
            float theta = atan2(ray.Direction.z, ray.Direction.x);
            float phi = acos(ray.Direction.y);
            float3 skyboxColor = skyboxTexture.SampleLevel(textureSampler, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
            radian += throughput * skyboxColor;
            break;
        }
    }
    if (renderInfo.accumulationFrameCount == 1) {
        renderAccumulationTexture[pixelIndex] = float3(0, 0, 0);
    }
    renderAccumulationTexture[pixelIndex] += radian;
    renderTexture[pixelIndex] = renderAccumulationTexture[pixelIndex] / renderInfo.accumulationFrameCount;
}

[shader("miss")]
void rayMiss(inout RayPayload payload : SV_RayPayload) {
    payload.materialID = -1;
}

[shader("closesthit")]
void rayClosestHit(inout RayPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];
    
    uint descriptorIndex = blasInstanceInfo.descriptorsHeapOffset + GeometryIndex() * 6; /* number of descriptors in a blas geometry */
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 1)];
    Texture2D<float3> emissiveTexture = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 2)];
    Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 3)];
    Texture2D<float3> metallicRoughnessTexture = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 4)];
    Texture2D<float3> normalTexture = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 5)];

    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex v1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex v2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
    float3x4 transform = ObjectToWorld3x4();
    float3x3 normalTransform = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    float3 p0 = mul(transform, float4(v0.position, 1));
    float3 p1 = mul(transform, float4(v1.position, 1));
    float3 p2 = mul(transform, float4(v2.position, 1));
    float3 n0 = normalize(mul(normalTransform, v0.normal));
    float3 n1 = normalize(mul(normalTransform, v1.normal));
    float3 n2 = normalize(mul(normalTransform, v2.normal));
    float3 position = barycentricsLerp(trigAttribs.barycentrics, p0, p1, p2);
    float3 shadingNormal = normalize(barycentricsLerp(trigAttribs.barycentrics, n0, n1, n2));
    float3 geometryNormal = triangleGeometryNormal(p0, p1, p2);
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
        
    float3 baseColor;
    float3 lightDir = normalize(float3(1, 1, 1));
    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        baseColor = float3(blasInstanceInfo.color & 0xff000000, blasInstanceInfo.color & 0x00ff0000, blasInstanceInfo.color & 0x0000ff00);
        //payload.radian += baseColor;
    }
    else {
        baseColor = baseColorTexture.SampleLevel(textureSampler, uv, 0, 0) * blasGeometryInfo.baseColor;
        float ndotl = dot(shadingNormal, lightDir);
        //payload.radian += baseColor * ndotl;
    }
}

[shader("anyhit")]
void rayAnyHit(inout RayPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    IgnoreHit();
}
