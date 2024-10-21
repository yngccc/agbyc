#include "shared.hlsli"
#include "brdf.hlsli"

GlobalRootSignature
globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
    "CBV(b0), SRV(t0), SRV(t1), SRV(t2),"
    "DescriptorTable(UAV(u0), UAV(u1), UAV(u2), SRV(t3)),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP, maxAnisotropy = 16, mipLODBias = 0),"
	"StaticSampler(s1, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR, maxAnisotropy = 16, mipLODBias = 0),"
	"StaticSampler(s2, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP, maxAnisotropy = 16, mipLODBias = 0),"
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

RaytracingPipelineConfig pipelineConfig = { 2 /*MaxTraceRecursionDepth*/ };

RaytracingShaderConfig shaderConfig = { 64 /*UINT MaxPayloadSizeInBytes*/, 8 /*UINT MaxAttributeSizeInBytes*/ };

TriangleHitGroup rayHitGroupPrimary = { "rayAnyHitPrimary", "rayClosestHitPrimary" };

TriangleHitGroup rayHitGroupShadow = { "", "rayClosestHitShadow" };

struct RayPayloadPrimary {
    float3 color;
    float depth;
    float4 transmissions;
    float4 depths;
    uint4 colors;
};

struct RayPayloadShadow {
    bool hit;
};

[shader("raygeneration")]
void rayGen() {
    uint2 resolution = DispatchRaysDimensions().xy;
    uint2 pixelIndex = DispatchRaysIndex().xy;
    float2 pixelCoord = float2(pixelIndex) + 0.5;
    RayDesc ray = cameraRayPinhole(pixelCoord, resolution, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    RayPayloadPrimary payload;
    payload.color = float3(0, 0, 0);
    payload.depth = 0;
    payload.transmissions = (1, 1, 1, 1);
    payload.depths = (1, 1, 1, 1);
    payload.colors = uint4(0, 0, 0, 0);
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    //payload.color = payload.color * payload.transmissions[3] + (1.0 - payload.transmissions[3]) * unpackRGB(payload.colors[3]);
    //payload.color = payload.color * payload.transmissions[2] + (1.0 - payload.transmissions[2]) * unpackRGB(payload.colors[2]);
    //payload.color = payload.color * payload.transmissions[1] + (1.0 - payload.transmissions[1]) * unpackRGB(payload.colors[1]);
    //payload.color = payload.color * payload.transmissions[0] + (1.0 - payload.transmissions[0]) * unpackRGB(payload.colors[0]);
    renderTexture[pixelIndex] = payload.color;
    depthTexture[pixelIndex] = payload.depth;
}

[shader("miss")]
void rayMissPrimary(inout RayPayloadPrimary payload) {
    float2 uv = equirectangularMapping(WorldRayDirection());
    float3 skyboxColor = skyboxTexture.SampleLevel(textureSampler, uv, 0);
    payload.color += skyboxColor * 0.4;
    payload.depth = 1;
}

[shader("closesthit")]
void rayClosestHitPrimary(inout RayPayloadPrimary payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];
    
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 1)];
    Texture2D<float4> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 2)];
    Texture2D<float3> metallicRoughnessTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 3)];
    Texture2D<float2> normalTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 4)];
    Texture2D<float3> emissiveTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 5)];

    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex v1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex v2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
    float3 p0 = mul(ObjectToWorld3x4(), float4(v0.position, 1));
    float3 p1 = mul(ObjectToWorld3x4(), float4(v1.position, 1));
    float3 p2 = mul(ObjectToWorld3x4(), float4(v2.position, 1));
    float3 n0 = normalize(mul(blasInstanceInfo.transformNormalMat, v0.normal));
    float3 n1 = normalize(mul(blasInstanceInfo.transformNormalMat, v1.normal));
    float3 n2 = normalize(mul(blasInstanceInfo.transformNormalMat, v2.normal));
    float3 b0 = normalize(cross(v0.normal, v0.tangent.xyz) * v0.tangent.w);
    float3 b1 = normalize(cross(v1.normal, v1.tangent.xyz) * v1.tangent.w);
    float3 b2 = normalize(cross(v2.normal, v2.tangent.xyz) * v2.tangent.w);
    
    float3 position = barycentricsLerp(trigAttribs.barycentrics, p0, p1, p2);
    float3 normal = barycentricsLerp(trigAttribs.barycentrics, n0, n1, n2);
    float3 tangent = barycentricsLerp(trigAttribs.barycentrics, v0.tangent.xyz, v1.tangent.xyz, v2.tangent.xyz);
    float3 bitangent = barycentricsLerp(trigAttribs.barycentrics, b0, b1, b2);
    tangent = normalize(mul(blasInstanceInfo.transformNormalMat, tangent));
    bitangent = normalize(mul(blasInstanceInfo.transformNormalMat, bitangent));
    float3 geometryNormal = triangleGeometryNormal(p0, p1, p2);
    if (dot(geometryNormal, -WorldRayDirection()) < 0) {
        geometryNormal = -geometryNormal;
    }
    if (dot(geometryNormal, normal) < 0) {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
        
    uint baseColorTextureWidth, baseColorTextureHeight;
    uint normalTextureWidth, normalTextureHeight;
    baseColorTexture.GetDimensions(baseColorTextureWidth, baseColorTextureHeight);
    normalTexture.GetDimensions(normalTextureWidth, normalTextureHeight);

    float2 texGrad1, texGrad2;
    float alpha = atan(2.0 * tan(renderInfo.cameraFovVertical * 0.5) / (float) baseColorTextureHeight);
    float radius = RayTCurrent() * tan(alpha);
    anisotropicEllipseAxes(position, normal, WorldRayDirection(), radius, p0, p1, p2, v0.uv, v1.uv, v2.uv, uv, texGrad1, texGrad2);
    float4 baseColor = baseColorTexture.SampleGrad(textureSampler, uv, texGrad1, texGrad2);
    float3 emissive = emissiveTexture.SampleGrad(textureSampler, uv, texGrad1, texGrad2);
    if (normalTextureHeight != baseColorTextureHeight) {
        alpha = atan(2.0f * tan(renderInfo.cameraFovVertical * 0.5) / (float) normalTextureHeight);
        radius = RayTCurrent() * tan(alpha);
        anisotropicEllipseAxes(position, normal, WorldRayDirection(), radius, p0, p1, p2, v0.uv, v1.uv, v2.uv, uv, texGrad1, texGrad2);
    }
    float2 normalMapXY = normalTexture.SampleGrad(textureSampler, uv, texGrad1, texGrad2);
    normalMapXY = normalMapXY * 2.0 - 1.0;
    float3 normalMapNormal = float3(normalMapXY, sqrt(1.0 - normalMapXY.x * normalMapXY.x - normalMapXY.y * normalMapXY.y));
    float3x3 tbnMat = transpose(float3x3(tangent, bitangent, normal));
    normal = normalize(mul(tbnMat, normalMapNormal));
    
    float4 color;
    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        color = unpackRGBA(blasInstanceInfo.color);
    }
    else {
        color = baseColor * blasGeometryInfo.baseColorFactor;
        color.rgb += emissive * blasGeometryInfo.emissiveFactor;
    }
    if (blasInstanceInfo.flags & BLASInstanceFlagHighlightTriangleEdges) {
        if (barycentricsOnEdge(trigAttribs.barycentrics, 0.03)) {
            color = unpackRGBA(blasInstanceInfo.color);
        }
    }
    payload.color += color.rgb * dot(normal, normalize(1.xxx));
    
    //float4 ndc = mul(renderInfo.cameraViewProjectMat, float4(position, 1));
    //payload.depth = ndc.z / ndc.w;
    payload.depth = RayTCurrent() / CAMERA_Z_MAX;
    
    //RayPayloadShadow rayPayload;
    //RayDesc shadowRay;
    ////shadowRay.Origin = offsetRayShadow(position, p0, p1, p2, n0, n1, n2, 1.0 - trigAttribs.barycentrics.x - trigAttribs.barycentrics.y, trigAttribs.barycentrics.x, trigAttribs.barycentrics.y);
    //shadowRay.Origin = offsetRay(position, geometryNormal);
    //shadowRay.TMin = 0;
    //shadowRay.TMax = 1000;
    //shadowRay.Direction = normalize(float3(1, 1, 1));
    //TraceRay(bvh, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRay, rayPayload);
    //if (rayPayload.hit) {
    //    payload.color += 0.xxx;
    //}
    //else {
    //    payload.color += 0.5.xxx;
    //}
}

[shader("anyhit")]
void rayAnyHitPrimary(inout RayPayloadPrimary payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    BLASInstanceInfo blasInstanceInfo = blasInstancesInfos[InstanceIndex()];
    BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[blasInstanceInfo.blasGeometriesOffset + GeometryIndex()];

    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset)];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 1)];
    Texture2D<float4> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(blasGeometryInfo.descriptorsHeapOffset + 2)];

    uint triangleIndex = PrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
    Vertex v1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
    Vertex v2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
    float3 p0 = mul(ObjectToWorld3x4(), float4(v0.position, 1));
    float3 p1 = mul(ObjectToWorld3x4(), float4(v1.position, 1));
    float3 p2 = mul(ObjectToWorld3x4(), float4(v2.position, 1));
    float3 position = barycentricsLerp(trigAttribs.barycentrics, p0, p1, p2);
    float3 normal = barycentricsLerp(trigAttribs.barycentrics, v0.normal, v1.normal, v2.normal);
    normal = normalize(mul(blasInstanceInfo.transformNormalMat, normal));
    float2 uv = barycentricsLerp(trigAttribs.barycentrics, v0.uv, v1.uv, v2.uv);
    
    float4 color;
    if (blasInstanceInfo.flags & BLASInstanceFlagForcedColor) {
        color = unpackRGBA(blasInstanceInfo.color);
    }
    else {
        uint baseColorTextureWidth, baseColorTextureHeight;
        baseColorTexture.GetDimensions(baseColorTextureWidth, baseColorTextureHeight);
        float2 texGrad1, texGrad2;
        float alpha = atan(2.0 * tan(renderInfo.cameraFovVertical * 0.5) / (float) baseColorTextureHeight);
        float radius = RayTCurrent() * tan(alpha);
        anisotropicEllipseAxes(position, normal, WorldRayDirection(), radius, p0, p1, p2, v0.uv, v1.uv, v2.uv, uv, texGrad1, texGrad2);
        color = baseColorTexture.SampleGrad(textureSampler, uv, texGrad1, texGrad2) * blasGeometryInfo.baseColorFactor;
    }
    if (blasInstanceInfo.flags & BLASInstanceFlagHighlightTriangleEdges) {
        if (barycentricsOnEdge(trigAttribs.barycentrics, 0.03)) {
            color = float4(0, 1, 0, 1);
        }
    }
    if (blasGeometryInfo.alphaMode == AlphaModeOpaque) {
        payload.color += color.rgb * 0.25;
        IgnoreHit();
    }
    else if (blasGeometryInfo.alphaMode == AlphaModeMask) {
        if (color.a < blasGeometryInfo.alphaCutOff) {
            IgnoreHit();
        }
        else {
            return;
        }
    }
    else if (blasGeometryInfo.alphaMode == AlphaModeBlend) {
        if (color.a > 0.99) {
            return;
        }
        else if (color.a < 0.01) {
            IgnoreHit();
        }
        else {
            // TODO:: implement blend
            return;
        }
    }
}

[shader("miss")]
void rayMissShadow(inout RayPayloadShadow payload) {
    payload.hit = false;
}

[shader("closesthit")]
void rayClosestHitShadow(inout RayPayloadShadow payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.hit = true;
}
