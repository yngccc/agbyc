#include "shared.hlsli"
#include "../structsHLSL.h"

GlobalRootSignature globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
	"StaticSampler(s0, filter = FILTER_ANISOTROPIC, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, mipLODBias = 0, maxAnisotropy = 16)"
};

sampler sampler0 : register(s0);
RaytracingPipelineConfig pipelineConfig = {3 /*MaxTraceRecursionDepth*/};
RaytracingShaderConfig shaderConfig = {16 /*UINT MaxPayloadSizeInBytes*/, 8 /*UINT MaxAttributeSizeInBytes*/};
TriangleHitGroup primaryRayHitGroup = {"" /*AnyHitShader*/, "primaryRayClosestHit" /*ClosestHitShader*/};
TriangleHitGroup secondaryRayHitGroup = {"" /*AnyHitShader*/, "secondaryRayClosestHit" /*ClosestHitShader*/};

struct PrimaryRayPayload {
    float3 color;
};

struct SecondaryRayPayload {
    bool hit;
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

    RayDesc primaryRay = pinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjectMat);
    PrimaryRayPayload primaryRayPayload;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, primaryRay, primaryRayPayload);
    renderTexture[pixelIndex] = float4(primaryRayPayload.color, 1);
}

[shader("miss")]
void primaryRayMiss(inout PrimaryRayPayload payload) {
    SKYBOX_TEXTURE_DESCRIPTOR(skyboxTexture);
    float3 viewDir = WorldRayDirection();
    float theta = atan2(viewDir.z, viewDir.x);
    float phi = acos(viewDir.y);
    float3 skyboxColor = skyboxTexture.SampleLevel(sampler0, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
    payload.color = skyboxColor * 0.2f;
}

[shader("closesthit")]
void primaryRayClosestHit(inout PrimaryRayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    TLAS_INSTANCES_INFOS_DESCRIPTOR(instancesInfos);
    TLASInstanceInfo instanceInfo = instancesInfos[InstanceIndex()];
    if (instanceInfo.flags & TLASInstanceFlagSelected && barycentricsOnEdge(trigAttribs.barycentrics, 0.02)) {
        payload.color = float3(0, 1, 0);
    } else{
        BLAS_GEOMETRIES_INFOS_DESCRIPTOR(blasGeometriesInfos);
        BLASGeometryInfo blasGeometryInfo = blasGeometriesInfos[instanceInfo.blasGeometriesOffset + GeometryIndex()];
        uint verticesDescriptorIndex = InstanceID() + GeometryIndex() * 3;
        StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex)];
        StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex + 1)];
        Texture2D<float3> baseColorTexture = ResourceDescriptorHeap[NonUniformResourceIndex(verticesDescriptorIndex + 2)];

        uint triangleIndex = PrimitiveIndex() * 3;
        Vertex v0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
        Vertex v1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
        Vertex v2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];
        float3x4 transform = ObjectToWorld3x4();
        float3x3 normalTransform = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
        float3 p0 = mul(transform, float4(v0.position, 1)).xyz;
        float3 p1 = mul(transform, float4(v1.position, 1)).xyz;
        float3 p2 = mul(transform, float4(v2.position, 1)).xyz;
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
        
        float3 baseColor = baseColorTexture.SampleGrad(sampler0, uv, texGradient1, texGradient2) * blasGeometryInfo.baseColorFactor.xyz;
        float3 lightDir = normalize(float3(1, 1, 1));
        float ndotl = dot(normal, lightDir);
        payload.color = baseColor * ndotl;
        
        //BVH_DESCRIPTOR(bvh);
        //SecondaryRayPayload rayPayload;
        //RayDesc shadowRay;
        //shadowRay.Origin = position + normal * 0.005;
        //shadowRay.TMin = 0.0f;
        //shadowRay.TMax = 1000.0f;
        //shadowRay.Direction = lightDir;
        //TraceRay(bvh, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRay, rayPayload);
        //if (rayPayload.hit) {
        //    payload.color *= 0.1;
        //}
    }
}

[shader("miss")]
void secondaryRayMiss(inout SecondaryRayPayload payload) {
    payload.hit = false;
}

[shader("closesthit")]
void secondaryRayClosestHit(inout SecondaryRayPayload payload, in BuiltInTriangleIntersectionAttributes trigAttribs) {
    payload.hit = true;
}
