#include "shared.hlsli"
#include "../structsHLSL.h"

GlobalRootSignature globalRootSig = {
    "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),"
	"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP),"
	"StaticSampler(s1, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR)"
};

sampler sampler0 : register(s0);
sampler sampler1 : register(s1);
RaytracingPipelineConfig pipelineConfig = {3};
RaytracingShaderConfig shaderConfig = {52, 8};
TriangleHitGroup primaryRayHitGroup = {"", "primaryRayClosestHit"};
TriangleHitGroup secondaryRayHitGroup = {"", "secondaryRayClosestHit"};

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

    RayDesc primaryRay = generatePinholeCameraRay(pixelCoord, renderInfo.cameraViewMatInverseTranspose, renderInfo.cameraProjMat);
    PrimaryRayPayload primaryRayPayload;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, primaryRay, primaryRayPayload);
    renderTexture[pixelIndex] = float4(primaryRayPayload.color.xyz, 1);
}

[shader("miss")]
void primaryRayMiss(inout PrimaryRayPayload payload) {
    SKYBOX_TEXTURE_DESCRIPTOR(skyboxTexture);
    float3 viewDir = WorldRayDirection();
    float theta = atan2(viewDir.z, viewDir.x);
    float phi = acos(viewDir.y);
    float3 skyboxColor = skyboxTexture.SampleLevel(sampler0, float2((theta + PI) / (2.0 * PI), phi / PI), 0);
    payload.color = skyboxColor * 0.1f;
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
        Vertex vertex0 = vertices[indices[NonUniformResourceIndex(triangleIndex)]];
        Vertex vertex1 = vertices[indices[NonUniformResourceIndex(triangleIndex + 1)]];
        Vertex vertex2 = vertices[indices[NonUniformResourceIndex(triangleIndex + 2)]];

        float3x4 transform = ObjectToWorld3x4();
        float3x3 normalTransform = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
        float3 position = barycentricsLerp(trigAttribs.barycentrics, vertex0.position, vertex1.position, vertex2.position);
        position = mul(transform, float4(position, 1));
        float3 normal = barycentricsLerp(trigAttribs.barycentrics, vertex0.normal, vertex1.normal, vertex2.normal);
        normal = normalize(mul(normalTransform, normal));
        float2 uv = barycentricsLerp(trigAttribs.barycentrics, vertex0.uv, vertex1.uv, vertex2.uv);
        float3 diffuse = baseColorTexture.SampleLevel(sampler1, uv, 0) * blasGeometryInfo.baseColorFactor.xyz;

        float3 lightDir = normalize(float3(1, 1, 1));
        float ndotl = dot(normal, lightDir);
        payload.color = diffuse * ndotl;
        
        BVH_DESCRIPTOR(bvh);
        SecondaryRayPayload rayPayload;
        RayDesc shadowRay;
        shadowRay.Origin = position + normal * 0.005;
        shadowRay.TMin = 0.0f;
        shadowRay.TMax = 1000.0f;
        shadowRay.Direction = lightDir;
        TraceRay(bvh, RAY_FLAG_NONE, 0xff, 1, 0, 1, shadowRay, rayPayload);
        if (rayPayload.hit) {
            payload.color *= 0.1;
        }
    }
}

[shader("miss")]

    void secondaryRayMiss

    (inout
    SecondaryRayPayload payload) {
    payload.hit = false;
}

[shader("closesthit")]

    void secondaryRayClosestHit

    (inout
    SecondaryRayPayload payload, in BuiltInTriangleIntersectionAttributes

    trigAttribs) {
    payload.hit = true;
}
