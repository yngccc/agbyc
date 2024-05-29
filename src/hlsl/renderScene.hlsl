#include "shared.hlsli"
#include "../structsHLSL.h"

GlobalRootSignature
globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
    "CBV(b0), SRV(t0), SRV(t1), SRV(t2),"
    "DescriptorTable(UAV(u0), UAV(u1), SRV(t3)),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, mipLODBias = 0, maxAnisotropy = 16)"
};

RWTexture2D<float4> renderTexture : register(u0);
RWTexture2D<float> depthTexture : register(u1);
ConstantBuffer<RenderInfo> renderInfo : register(b0);
RaytracingAccelerationStructure bvh : register(t0);
StructuredBuffer<BLASInstanceInfo> blasInstancesInfos : register(t1);
StructuredBuffer<BLASGeometryInfo> blasGeometriesInfos : register(t2);
Texture2D<float3> skyboxTexture : register(t3);
sampler textureSampler : register(s0);

RaytracingPipelineConfig pipelineConfig = {4 /*MaxTraceRecursionDepth*/};

RaytracingShaderConfig shaderConfig = {16 /*UINT MaxPayloadSizeInBytes*/, 8 /*UINT MaxAttributeSizeInBytes*/};

TriangleHitGroup primaryRayHitGroup = {"primaryRayAnyHit", "primaryRayClosestHit"};

TriangleHitGroup secondaryRayHitGroup = {"", "secondaryRayClosestHit"};

struct PrimaryRayPayload {
    float3 color;
    float depth;
};

struct SecondaryRayPayload {
    bool hit;
};

[shader("raygeneration")]
void rayGen() {
    uint2 imageSize = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    float2 pixelCoord = ((float2(pixelIndex) + 0.5f) / float2(imageSize)) * 2.0f - 1.0f;

    RayDesc primaryRay = pinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    PrimaryRayPayload primaryRayPayload;
    primaryRayPayload.color = float3(0, 0, 0);
    primaryRayPayload.depth = 0;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, primaryRay, primaryRayPayload);
    renderTexture[pixelIndex] = float4(primaryRayPayload.color, 0);
    depthTexture[pixelIndex] = primaryRayPayload.depth;
}

[shader("miss")]
void primaryRayMiss(inout PrimaryRayPayload payload) {
    float3 viewDir = WorldRayDirection();
    float theta = atan2(viewDir.z, viewDir.x);
    float phi = acos(viewDir.y);
    float3 skyboxColor = skyboxTexture.SampleLevel(textureSampler, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
    payload.color += skyboxColor * 0.2;
    payload.depth = 1.0;
}

[shader("closesthit")]
void primaryRayClosestHit(inout PrimaryRayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
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
    float3 normal = normalize(barycentricsLerp(trigAttribs.barycentrics, n0, n1, n2));
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
        
    uint baseColorTextureWidth, baseColorTextureHeight;
    baseColorTexture.GetDimensions(baseColorTextureWidth, baseColorTextureHeight);
    float fovy = RADIAN(50);
    float alpha = atan(2.0 * tan(fovy * 0.5) / (float)baseColorTextureHeight);
    float radius = RayTCurrent() * tan(alpha);
    float2 texGradient1, texGradient2;
    anisotropicEllipseAxes(position, normal, WorldRayDirection(), radius, p0, p1, p2, v0.uv, v1.uv, v2.uv, uv, texGradient1, texGradient2);

    float3 baseColor;
    float3 lightDir = normalize(float3(1, 1, 1));
    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        baseColor = float3(blasInstanceInfo.color & 0xff000000, blasInstanceInfo.color & 0x00ff0000, blasInstanceInfo.color & 0x0000ff00);
        payload.color += baseColor;
    }
    else {
        baseColor = baseColorTexture.SampleGrad(textureSampler, uv, texGradient1, texGradient2) * blasGeometryInfo.baseColor;
        float ndotl = dot(normal, lightDir);
        payload.color += baseColor * ndotl;
    }

    float4 ndc = mul(renderInfo.cameraViewProjectMat, float4(position, 1));
    payload.depth = ndc.z / ndc.w;
    
    SecondaryRayPayload rayPayload;
    RayDesc shadowRay;
    shadowRay.Origin = position + normal * 0.008;
    shadowRay.TMin = 0.0f;
    shadowRay.TMax = 1000.0f;
    shadowRay.Direction = lightDir;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRay, rayPayload);
    if (rayPayload.hit) {
        payload.color *= 0.1;
    }
}

[shader("anyhit")]
void primaryRayAnyHit(inout PrimaryRayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];

    uint descriptorIndex = blasInstanceInfo.descriptorsHeapOffset + GeometryIndex() * 6; /* number of descriptors in a blas geometry */
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 1)];
    Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex + 3)];

    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex v1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex v2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
             
    float3 baseColor;
    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        baseColor = float3(blasInstanceInfo.color & 0xff000000, blasInstanceInfo.color & 0x00ff0000, blasInstanceInfo.color & 0x0000ff00);
    }
    else {
        baseColor = baseColorTexture.SampleLevel(textureSampler, uv, 0) * blasGeometryInfo.baseColor;
    }
    payload.color += (baseColor * 0.25);
    
    IgnoreHit();
}

[shader("miss")]
void secondaryRayMiss(inout SecondaryRayPayload payload) {
    payload.hit = false;
}

[shader("closesthit")]
void secondaryRayClosestHit(inout SecondaryRayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.hit = true;
}
