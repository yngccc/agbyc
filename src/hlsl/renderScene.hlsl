#include "shared.h"

GlobalRootSignature globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
    "CBV(b0), SRV(t0), SRV(t1), SRV(t2),"
    "DescriptorTable(UAV(u0), UAV(u1), UAV(u2), SRV(t3)),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, mipLODBias = 0, maxAnisotropy = 16)"
};

RWTexture2D<float3> renderTexture : register(u0);
RWTexture2D<float> depthTexture : register(u1);
RWTexture2D<float2> motionVectorTexture : register(u2);
ConstantBuffer<RenderInfo> renderInfo : register(b0);
RaytracingAccelerationStructure bvh : register(t0);
StructuredBuffer<BLASInstanceInfo> blasInstancesInfos : register(t1);
StructuredBuffer<BLASGeometryInfo> blasGeometriesInfos : register(t2);
Texture2D<float3> skyboxTexture : register(t3);
sampler textureSampler : register(s0);

RaytracingPipelineConfig pipelineConfig = {2 /*MaxTraceRecursionDepth*/};

RaytracingShaderConfig shaderConfig = {16 /*UINT MaxPayloadSizeInBytes*/, 8 /*UINT MaxAttributeSizeInBytes*/};

TriangleHitGroup rayHitGroupPrimary = {"rayAnyHitPrimary", "rayClosestHitPrimary"};

TriangleHitGroup rayHitGroupShadow = {"", "rayClosestHitShadow"};

struct RayPayloadPrimary {
    float3 color;
    float depth;
};

struct RayPayloadShadow {
    bool hit;
};

[shader("raygeneration")]
void rayGen() {
    uint2 resolution = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    float2 pixelCoord = ((float2(pixelIndex) + 0.5) / float2(resolution)) * 2.0 - 1.0;

    RayDesc ray = pinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    RayPayloadPrimary payload = (RayPayloadPrimary)0;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    renderTexture[pixelIndex] = payload.color;
    depthTexture[pixelIndex] = payload.depth;
}

[shader("miss")]
void rayMissPrimary(inout RayPayloadPrimary payload) {
    float2 uv = equirectangularMapping(WorldRayDirection());
    float3 skyboxColor = skyboxTexture.SampleLevel(textureSampler, uv, 0);
    payload.color += skyboxColor * 0.4;
    payload.depth = 1.0;
}

[shader("closesthit")]
void rayClosestHitPrimary(inout RayPayloadPrimary payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];
    
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 1)];
    Texture2D<float3> emissiveTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 2)];
    Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 3)];
    Texture2D<float3> metallicRoughnessTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 4)];
    Texture2D<float3> normalTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 5)];

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
    if (dot(geometryNormal, -WorldRayDirection()) < 0.0f) geometryNormal = -geometryNormal;
    if (dot(geometryNormal, shadingNormal) < 0.0f) shadingNormal = -shadingNormal;
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
        
    uint baseColorTextureWidth, baseColorTextureHeight;
    baseColorTexture.GetDimensions(baseColorTextureWidth, baseColorTextureHeight);
    float fovy = RADIAN(50.0f);
    float alpha = atan(2.0f * tan(fovy * 0.5f) / (float)baseColorTextureHeight);
    float radius = RayTCurrent() * tan(alpha);
    float2 texGradient1, texGradient2;
    anisotropicEllipseAxes(position, shadingNormal, WorldRayDirection(), radius, p0, p1, p2, v0.uv, v1.uv, v2.uv, uv, texGradient1, texGradient2);

    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        float3 baseColor = float3(blasInstanceInfo.color & 0xff000000, blasInstanceInfo.color & 0x00ff0000, blasInstanceInfo.color & 0x0000ff00);
        payload.color += baseColor;
    }
    else {
        float3 lightDir = normalize(float3(1, 1, 1));
        float3 baseColor = baseColorTexture.SampleGrad(textureSampler, uv, texGradient1, texGradient2) * blasGeometryInfo.baseColor;
        float ndotl = dot(shadingNormal, lightDir);
        payload.color += baseColor/* * ndotl*/;
    }

    float4 ndc = mul(renderInfo.cameraViewProjectMat, float4(position, 1));
    payload.depth = ndc.z / ndc.w;
    
    //RayPayloadShadow rayPayload;
    //RayDesc shadowRay;
    //shadowRay.Origin = offsetRay(position, geometryNormal);
    ////shadowRay.Origin = offsetRayShadow(position, p0, p1, p2, n0, n1, n2, trigAttribs.barycentrics.x, trigAttribs.barycentrics.y, 1.0 - trigAttribs.barycentrics.x - trigAttribs.barycentrics.y);
    //shadowRay.TMin = 0.0f;
    //shadowRay.TMax = 1000.0f;
    //shadowRay.Direction = lightDir;
    //TraceRay(bvh, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRay, rayPayload);
    //if (rayPayload.hit) {
    //    payload.color *= 0.1;
    //}
}

[shader("anyhit")]
void rayAnyHitPrimary(inout RayPayloadPrimary payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];

    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 1)];
    Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 3)];

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
void rayMissShadow(inout RayPayloadShadow payload) {
    payload.hit = false;
}

[shader("closesthit")]
void rayClosestHitShadow(inout RayPayloadShadow payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.hit = true;
}
