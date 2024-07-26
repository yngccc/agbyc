#include "shared.h"

GlobalRootSignature globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
    "CBV(b0), SRV(t0), SRV(t1), SRV(t2),"
    "DescriptorTable(UAV(u0), UAV(u1), SRV(t3)),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, mipLODBias = 0, maxAnisotropy = 16),"
};

ConstantBuffer<RenderInfo> renderInfo : register(b0);
RWTexture2D<float3> renderTexture : register(u0);
RWTexture2D<float3> pathTracerAccumulationTexture : register(u1);
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
    uint blasGeometryInfoIndex;
    float2 geometryNormalEncoded;
    float2 shadingNormalEncoded;
    float2 uv;
};

#define MAX_BOUNCES 16
#define MIN_BOUNCES 3

[shader("raygeneration")]
void rayGen() {
    uint2 resolution = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    uint rngState = initRNG(pixelIndex, resolution, renderInfo.pathTracerAccumulationFrameCount);
    float2 offset = float2(rand(rngState), rand(rngState));
    offset = lerp(float2(-0.5, -0.5), float2(0.5, 0.5), offset);
    float2 pixelCoord = ((float2(pixelIndex) + offset + 0.5) / float2(resolution)) * 2.0 - 1.0;
    RayDesc ray = pinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    RayPayload payload = (RayPayload)0;
    float3 radian = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
        if (payload.blasGeometryInfoIndex == UINT_MAX) {
            if (bounce == 0) {
                float2 uv = equirectangularMapping(ray.Direction);
                float3 skyboxColor = skyboxTexture.SampleLevel(textureSampler, uv, 0);
                radian += throughput * skyboxColor * 0.4;
            }
            else {
                radian += throughput * float3(1.0, 1.0, 1.0);
            }
            break;
        }
        if (bounce == (MAX_BOUNCES - 1)) break;
		if (bounce > MIN_BOUNCES) {
			float rrProbability = min(0.95f, luminance(throughput));
			if (rrProbability < rand(rngState)) break;
			else throughput /= rrProbability;
		}
        float3 geometryNormal = decodeNormalOctahedron(payload.geometryNormalEncoded);
        float3 shadingNormal = decodeNormalOctahedron(payload.shadingNormalEncoded);
        float3 viewDir = -ray.Direction;
        if (dot(geometryNormal, viewDir) < 0.0) geometryNormal = -geometryNormal;
        if (dot(geometryNormal, shadingNormal) < 0.0) shadingNormal = -shadingNormal;
       //if (dot(shadingNormal, viewDir) <= 0.0f) break;
        
        BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[NonUniformResourceIndex(payload.blasGeometryInfoIndex)];
        Texture2D<float3> emissiveTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 2)];
        Texture2D<float4> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 3)];
        Texture2D<float3> metallicRoughnessTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 4)];
        Texture2D<float3> normalTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 5)];
        
        float4 baseColor = baseColorTexture.SampleLevel(textureSampler, payload.uv, 0) * blasGeometryInfo.baseColorFactor;
        float4 rotation = getRotationFromZAxis(shadingNormal);
        float3 sampleDirection = sampleHemisphereCosineWeighted(float2(rand(rngState), rand(rngState)));
        sampleDirection = rotatePoint(rotation, sampleDirection);
        if (dot(geometryNormal, sampleDirection) <= 0.0f) break;
        throughput *= baseColor.xyz;
        ray.Origin = offsetRay(payload.position, geometryNormal);
        ray.Direction = sampleDirection;
    }
    if (renderInfo.pathTracerAccumulationFrameCount == 1) {
        pathTracerAccumulationTexture[pixelIndex] = float3(0, 0, 0);
    }
    pathTracerAccumulationTexture[pixelIndex] += radian;
    renderTexture[pixelIndex] = pathTracerAccumulationTexture[pixelIndex] / renderInfo.pathTracerAccumulationFrameCount;
}

[shader("miss")]
void rayMiss(inout RayPayload payload : SV_RayPayload) {
    payload.blasGeometryInfoIndex = UINT_MAX;
}

[shader("closesthit")]
void rayClosestHit(inout RayPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];
    
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 1)];
    //Texture2D<float3> emissiveTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 2)];
    //Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 3)];
    //Texture2D<float3> metallicRoughnessTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 4)];
    //Texture2D<float3> normalTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 5)];

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
        
    payload.blasGeometryInfoIndex = blasInstanceInfo.blasGeometriesOffset + GeometryIndex();
    payload.position = position;
    payload.geometryNormalEncoded = encodeNormalOctahedron(geometryNormal);
    payload.shadingNormalEncoded = encodeNormalOctahedron(shadingNormal);
    payload.uv = uv;
}

[shader("anyhit")]
void rayAnyHit(inout RayPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    IgnoreHit();
}
