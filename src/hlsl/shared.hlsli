#define PI 3.14159265358979323846
#define FLT_MAX 3.402823466e+38F
#define UINT_MAX 0xffffffff

#define HDR_SCALE_FACTOR (80.0f / 10000.0f)

#define RENDER_TEXTURE_SRV_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[0];
#define RENDER_TEXTURE_UAV_DESCRIPTOR(name) RWTexture2D<float4> name = ResourceDescriptorHeap[1];
#define RENDER_INFO_DESCRIPTOR(name) ConstantBuffer<RenderInfo> name = ResourceDescriptorHeap[2];
#define BVH_DESCRIPTOR(name) RaytracingAccelerationStructure name = ResourceDescriptorHeap[3];
#define TLAS_INSTANCES_INFOS_DESCRIPTOR(name) StructuredBuffer<TLASInstanceInfo> name = ResourceDescriptorHeap[4];
#define BLAS_GEOMETRIES_INFOS_DESCRIPTOR(name) StructuredBuffer<BLASGeometryInfo> name = ResourceDescriptorHeap[5];
#define SKYBOX_TEXTURE_DESCRIPTOR(name) Texture2D<float3> name = ResourceDescriptorHeap[6];
#define IMGUI_IMAGE_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[7];
#define COLLISION_QUERIES_DESCRIPTOR(name) StructuredBuffer<CollisionQuery> name = ResourceDescriptorHeap[8];
#define COLLISION_QUERY_RESULTS_DESCRIPTOR(name) RWStructuredBuffer<CollisionQueryResult> name = ResourceDescriptorHeap[9];

float srgbToLinear(float e) {
    if (e <= 0.04045) {
        return e / 12.92;
    } else{
        return pow((e + 0.055) / 1.055, 2.4);
    }
}

float3 srgbToLinear(float3 srgb) {
    return float3(srgbToLinear(srgb.x), srgbToLinear(srgb.y), srgbToLinear(srgb.z));
}

float linearToSrgb(float e) {
    if (e <= 0.0031308) {
        return e * 12.92;
    } else{
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

RayDesc generatePinholeCameraRay(in float2 pixelCoord, in float4x4 cameraViewMat, in float4x4 cameraProjMat) {
    RayDesc ray;
    ray.Origin = cameraViewMat[3].xyz;
    ray.TMin = 0.0f;
    ray.TMax = FLT_MAX;
    float aspect = cameraProjMat[1][1] / cameraProjMat[0][0];
    float tanHalfFovY = 1.0f / cameraProjMat[1][1];
    ray.Direction = normalize((pixelCoord.x * cameraViewMat[0].xyz * tanHalfFovY * aspect) - (pixelCoord.y * cameraViewMat[1].xyz * tanHalfFovY) + cameraViewMat[2].xyz);
    return ray;
}

RayDesc generateShadowRay(in float3 a, in float3 b, in float3 c,
                          in float3 na, in float3 nb, in float3 nc,
                          in float u, in float v, in float w,
                          in float3 p, in float3 dir, in float length) {
    float3 tmpu = p - a;
    float3 tmpv = p - b;
    float3 tmpw = p - c;
    float dotu = min(0.0, dot(tmpu, na));
    float dotv = min(0.0, dot(tmpv, nb));
    float dotw = min(0.0, dot(tmpw, nc));
    tmpu -= dotu * na;
    tmpv -= dotv * nb;
    tmpw -= dotw * nc;
    float3 pp = p + u * tmpu + v * tmpv + w * tmpw;
    
    RayDesc ray;
    ray.Origin = pp;
    ray.TMax = 0.0f;
    ray.TMax = length;
    ray.Direction = dir;
    return ray;
}

void barycentricWorldDerivatives(float3 A1, float3 A2, out float3 du_dx, out float3 dv_dx) {
    float3 Nt = cross(A1, A2);
    float ntDotnt = dot(Nt, Nt);
    du_dx = cross(A2, Nt) / ntDotnt;
    dv_dx = cross(Nt, A1) / ntDotnt;
}

float3x3 worldScreenDerivatives(float4x4 worldToTargetMat, float4x4 targetToWorldMat, float4 x) {
    float3x3 dx_dxt = (float3x3)targetToWorldMat;
    dx_dxt[0] -= x.x * targetToWorldMat[3].xyz;
    dx_dxt[1] -= x.y * targetToWorldMat[3].xyz;
    dx_dxt[2] -= x.z * targetToWorldMat[3].xyz;
    return dx_dxt;
}

float2 depthGradient(float4 x, float3 n, float4x4 targetToWorldMat) {
    float4 n4 = float4(n, 0);
    n4.w = -dot(n4.xyz, x.xyz);
    n4 = mul(n4, targetToWorldMat);
    n4.z = max(abs(n4.r), 0.0001) * sign(n4.z);
    return n4.xy / -n4.z;
}

float2x2 barycentricDerivatives(float4 x, float3 n, float3 x0, float3 x1, float3 x2, float4x4 worldToTargetMat, float4x4 targetToWorldMat) {
    float3 du_dx, dv_dx;
    barycentricWorldDerivatives(x1 - x0, x2 - x0, du_dx, dv_dx);
    float3x3 dx_dxt = worldScreenDerivatives(worldToTargetMat, targetToWorldMat, x);
    float3 du_dxt = du_dx.x * dx_dxt[0] + du_dx.y * dx_dxt[1] + du_dx.z * dx_dxt[2];
    float3 dv_dxt = dv_dx.x * dx_dxt[0] + dv_dx.y * dx_dxt[1] + dv_dx.z * dx_dxt[2];
    float2 ddepth_dXY = depthGradient(x, n, targetToWorldMat);
    float wMx = dot(worldToTargetMat[3], x);
    float2 du_dXY = (du_dxt.xy + du_dxt.z * ddepth_dXY) * wMx;
    float2 dv_dXY = (dv_dxt.xy + dv_dxt.z * ddepth_dXY) * wMx;
    return float2x2(du_dXY, dv_dXY);
}

float2x2 texCoordDerivatives(float2x2 duv_dx1x2, float2 st0, float2 st1, float2 st2) {
    float2x2 dtc_duv = float2x2(st1.x - st0.x, st2.x - st0.x, st1.y - st0.y, st2.y - st0.y);
    return mul(dtc_duv, duv_dx1x2);    
}