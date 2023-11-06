#define PI 3.14159265358979323846
#define FLT_MAX 3.402823466e+38F
#define UINT_MAX 0xffffffff

#define HDR_SCALE_FACTOR (80.0f / 10000.0f)

#define RENDER_TEXTURE_SRV_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[0]
#define RENDER_TEXTURE_UAV_DESCRIPTOR(name) RWTexture2D<float4> name = ResourceDescriptorHeap[1]
#define RENDER_INFO_DESCRIPTOR(name) ConstantBuffer<RenderInfo> name = ResourceDescriptorHeap[2]
#define BVH_DESCRIPTOR(name) RaytracingAccelerationStructure name = ResourceDescriptorHeap[3]
#define TLAS_INSTANCES_INFOS_DESCRIPTOR(name) StructuredBuffer<TLASInstanceInfo> name = ResourceDescriptorHeap[4]
#define BLAS_GEOMETRIES_INFOS_DESCRIPTOR(name) StructuredBuffer<BLASGeometryInfo> name = ResourceDescriptorHeap[5]
#define SKYBOX_TEXTURE_DESCRIPTOR(name) Texture2D<float3> name = ResourceDescriptorHeap[6]
#define READBACK_BUFFER_DESCRIPTOR(name) RWStructuredBuffer<ReadBackBuffer> name = ResourceDescriptorHeap[7]
#define IMGUI_TEXTURE_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[8]
#define COLLISION_QUERIES_DESCRIPTOR(name) StructuredBuffer<CollisionQuery> name = ResourceDescriptorHeap[9]
#define COLLISION_QUERY_RESULTS_DESCRIPTOR(name) RWStructuredBuffer<CollisionQueryResult> name = ResourceDescriptorHeap[10]

float srgbToLinear(float e) {
    if (e <= 0.04045) {
        return e / 12.92;
    }
    else {
        return pow((e + 0.055) / 1.055, 2.4);
    }
}

float3 srgbToLinear(float3 srgb) {
    return float3(srgbToLinear(srgb.x), srgbToLinear(srgb.y), srgbToLinear(srgb.z));
}

float linearToSrgb(float e) {
    if (e <= 0.0031308) {
        return e * 12.92;
    }
    else {
        return 1.055 * pow(e, 1.0 / 2.4) - 0.055;
    }
}

float3 linearToSrgb(float3 rgb) {
    return float3(linearToSrgb(rgb.x), linearToSrgb(rgb.y), linearToSrgb(rgb.z));
}

float linearToPQ(float e) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float ym1 = pow(e, m1);
    return pow((c1 + c2 * ym1) / (1.0 + c3 * ym1), m2);
}

float3 linearToPQ(float3 rgb) {
    return float3(linearToPQ(rgb.x), linearToPQ(rgb.y), linearToPQ(rgb.z));
}

float3 bt709To2020(float3 rgb) {
    const float3x3 mat = {
        0.6274, 0.3293, 0.0433,
		0.0691, 0.9195, 0.0114,
		0.0164, 0.0880, 0.8956
    };
    return mul(mat, rgb);
}

float2 barycentricsLerp(in float2 barycentrics, in float2 vertAttrib0, in float2 vertAttrib1, in float2 vertAttrib2) {
    return vertAttrib0 + barycentrics.x * (vertAttrib1 - vertAttrib0) + barycentrics.y * (vertAttrib2 - vertAttrib0);
}

float3 barycentricsLerp(in float2 barycentrics, in float3 vertAttrib0, in float3 vertAttrib1, in float3 vertAttrib2) {
    return vertAttrib0 + barycentrics.x * (vertAttrib1 - vertAttrib0) + barycentrics.y * (vertAttrib2 - vertAttrib0);
}

bool barycentricsOnEdge(in float2 barycentrics, in float edgeThickness) {
    return (barycentrics.x < edgeThickness) || (barycentrics.y < edgeThickness) || ((1.0 - barycentrics.x - barycentrics.y) < edgeThickness);
}

RayDesc generatePinholeCameraRay(in float2 pixel, in float4x4 cameraViewMat, in float4x4 cameraProjMat) {
    RayDesc ray;
    ray.Origin = cameraViewMat[3].xyz;
    ray.TMin = 0.f;
    ray.TMax = FLT_MAX;
    float aspect = cameraProjMat[1][1] / cameraProjMat[0][0];
    float tanHalfFovY = 1.0f / cameraProjMat[1][1];
    ray.Direction = normalize(
        (pixel.x * cameraViewMat[0].xyz * tanHalfFovY * aspect) -
		(pixel.y * cameraViewMat[1].xyz * tanHalfFovY) +
        cameraViewMat[2].xyz
    );
    return ray;
}


