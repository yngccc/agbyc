#define PI 3.14159265358979323846
#define FLT_MAX 3.402823466e+38F
#define UINT_MAX 0xffffffff
#define RADIAN(d) (d * (PI / 180.0))
#define DEGREE(r) (r * (180.0 / PI))

#define HDR_SCALE_FACTOR (80.0f / 10000.0f)

#define RENDER_TEXTURE_SRV_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[0];
#define RENDER_TEXTURE_UAV_DESCRIPTOR(name) RWTexture2D<float4> name = ResourceDescriptorHeap[1];
#define DEPTH_TEXTURE_UAV_DESCRIPTOR(name) RWTexture2D<float> name = ResourceDescriptorHeap[2];
#define MOTION_VECTOR_TEXTURE_UAV_DESCRIPTOR(name) RWTexture2D<float2> name = ResourceDescriptorHeap[3];
#define RENDER_INFO_DESCRIPTOR(name) ConstantBuffer<RenderInfo> name = ResourceDescriptorHeap[4];
#define BVH_DESCRIPTOR(name) RaytracingAccelerationStructure name = ResourceDescriptorHeap[5];
#define TLAS_INSTANCES_INFOS_DESCRIPTOR(name) StructuredBuffer<TLASInstanceInfo> name = ResourceDescriptorHeap[6];
#define BLAS_GEOMETRIES_INFOS_DESCRIPTOR(name) StructuredBuffer<BLASGeometryInfo> name = ResourceDescriptorHeap[7];
#define SKYBOX_TEXTURE_DESCRIPTOR(name) Texture2D<float3> name = ResourceDescriptorHeap[8];
#define IMGUI_IMAGE_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[9];
#define DIRECT_WRITE_IMAGE_DESCRIPTOR(name) Texture2D<float4> name = ResourceDescriptorHeap[10];
#define COLLISION_QUERIES_DESCRIPTOR(name) StructuredBuffer<CollisionQuery> name = ResourceDescriptorHeap[11];
#define COLLISION_QUERY_RESULTS_DESCRIPTOR(name) RWStructuredBuffer<CollisionQueryResult> name = ResourceDescriptorHeap[12];

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

float linearToSRGB(float e) {
    if (e <= 0.0031308) {
        return e * 12.92;
    } else{
        return 1.055 * pow(e, 1.0 / 2.4) - 0.055;
    }
}

float3 linearToSRGB(float3 rgb) {
    return float3(linearToSRGB(rgb.x), linearToSRGB(rgb.y), linearToSRGB(rgb.z));
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

RayDesc pinholeCameraRay(in float2 pixelCoord, in float4x4 cameraViewMat, in float4x4 cameraProjMat) {
    RayDesc ray;
    ray.Origin = cameraViewMat[3].xyz;
    ray.TMin = 0.0f;
    ray.TMax = FLT_MAX;
    float aspect = cameraProjMat[1][1] / cameraProjMat[0][0];
    float tanHalfFovY = 1.0f / cameraProjMat[1][1];
    ray.Direction = normalize((pixelCoord.x * cameraViewMat[0].xyz * tanHalfFovY * aspect) - (pixelCoord.y * cameraViewMat[1].xyz * tanHalfFovY) + cameraViewMat[2].xyz);
    return ray;
}

RayDesc shadowRay(in float3 a, in float3 b, in float3 c,
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

void anisotropicEllipseAxes(in float3 p, in float3 f, in float3 d, in float rayConeRadiusAtIntersection,
                            in float3 position0, in float3 position1, in float3 position2,
                            in float2 txcoord0, in float2 txcoord1, in float2 txcoord2,
                            in float2 interpolatedTexCoordsAtIntersection,
                            out float2 texGradient1, out float2 texGradient2) {
    // compute ellipse axes
    float3 a1 = d - dot(f, d) * f;
    float3 p1 = a1 - dot(d, a1) * d;
    a1 *= rayConeRadiusAtIntersection / max(0.0001, length(p1));
    
    float3 a2 = cross(f, a1);
    float3 p2 = a2 - dot(d, a2) * d;
    a2 *= rayConeRadiusAtIntersection / max(0.0001, length(p2));
    
    // compute texture coordinate gradients
    float3 eP, delta = p - position0;
    float3 e1 = position1 - position0;
    float3 e2 = position2 - position0;
    float oneOverAreaTriangle = 1.0 / dot(f, cross(e1, e2));
    eP = delta + a1;
    float u1 = dot(f, cross(eP, e2)) * oneOverAreaTriangle;
    float v1 = dot(f, cross(e1, eP)) * oneOverAreaTriangle;
    texGradient1 = (1.0 - u1 - v1) * txcoord0 + u1 * txcoord1 + v1 * txcoord2 - interpolatedTexCoordsAtIntersection;
    eP = delta + a2;
    float u2 = dot(f, cross(eP, e2)) * oneOverAreaTriangle;
    float v2 = dot(f, cross(e1, eP)) * oneOverAreaTriangle;
    texGradient2 = (1.0 - u2 - v2) * txcoord0 + u2 * txcoord1 + v2 * txcoord2 - interpolatedTexCoordsAtIntersection;
}
