#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <list>
#include <span>
#include <stack>
#include <streambuf>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <shellscalingapi.h>
#include <shlobj.h>
#include <cderr.h>
#include <commdlg.h>

#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <d3d11on12.h>
#include <dwrite.h>
#include <d2d1_3.h>

#undef near
#undef far

#include <xinput.h>

#define _XM_SSE4_INTRINSICS_
#include <directxmath.h>
#include <directxtex.h>
#include <directxcollision.h>
#include <directxpackedvector.h>
using namespace DirectX;

#include <pix3.h>

#include <rapidyaml/rapidyaml-0.5.0.hpp>

#include <cgltf/cgltf.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imguizmo.h>

#include <d3d12ma/d3d12memalloc.h>

#include <tracy/tracy/tracy.hpp>

#include <nvapi/nvapi.h>

typedef int8_t int8;
typedef int16_t int16;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint32_t uint;

static const float euler = 2.71828182845904523536f;
static const float pi = 3.14159265358979323846f;
static const float sqrt2 = 1.41421356237309504880f;

#define KILOBYTES(n) (1024 * (n))
#define MEGABYTES(n) (1024 * 1024 * (n))
#define GIGABYTES(n) (1024 * 1024 * 1024 * (n))
#define RADIAN(d) (d * (pi / 180.0f))
#define DEGREE(r) (r * (180.0f / pi))

template <typename T, uint32 N>
constexpr uint32 countof(const T (&)[N]) { return N; }

template <typename T>
uint64 vectorSizeof(const std::vector<T>& v) { return v.size() * sizeof(T); }

template <typename T, typename T2>
T align(T x, T2 n) {
    T remainder = x % (T)n;
    return remainder == 0 ? x : x + ((T)n - remainder);
}

bool getBit(uint32 n, uint32 index) {
    return (n >> index) & 1;
}

uint32 setBit(uint32 n, uint32 index) {
    return n |= (1 << index);
}

uint32 unsetBit(uint32 n, uint32 index) {
    return n &= ~(1 << index);
}

uint32 toggleBit(uint32 n, uint32 index) {
    return n ^= (1 << index);
}

struct int2 {
    int x = 0, y = 0;
};

struct uint8_4 {
    uint8 x = 0, y = 0, z = 0, w = 0;
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct uint16_4 {
    uint16 x = 0, y = 0, z = 0, w = 0;
    void operator=(uint8_4 v) { x = v.x, y = v.y, z = v.z, w = v.w; }
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct uint_4 {
    uint32 x = 0, y = 0, z = 0, w = 0;
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct float2 {
    float x = 0, y = 0;

    float2() = default;
    float2(float x, float y) : x(x), y(y) {}
    float2(const XMVECTOR& v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)) {}
    bool operator==(float2 v) const { return x == v.x && y == v.y; }
    bool operator!=(float2 v) const { return x != v.x || y != v.y; }
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
    float2 operator+(float v) const { return float2(x + v, y + v); }
    float2 operator+(float2 v) const { return float2(x + v.x, y + v.y); }
    float2 operator-(float v) const { return float2(x - v, y - v); }
    float2 operator-(float2 v) const { return float2(x - v.x, y - v.y); }
    float2 operator-() const { return float2(-x, -y); }
    float2 operator*(float v) const { return float2(x * v, y * v); }
    float2 operator*(float2 v) const { return float2(x * v.x, y * v.y); }
    float2 operator/(float v) const { return float2(x / v, y / v); }
    float2 operator/(float2 v) const { return float2(x / v.x, y / v.y); }
    std::string toString() const { return std::format("[{}, {}]", x, y); }
    float length() const { return sqrtf(x * x + y * y); }
    float2 normalize() const {
        float l = length();
        return (l > 0) ? float2(x / l, y / l) : float2(x, y);
    }
};

struct float3 {
    float x = 0, y = 0, z = 0;

    float3() = default;
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
    float3(const float* v) : x(v[0]), y(v[1]), z(v[2]) {}
    float3(const XMVECTOR& v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)) {}
    void operator=(const XMVECTOR& v) { x = XMVectorGetX(v), y = XMVectorGetY(v), z = XMVectorGetZ(v); }
    bool operator==(const float3& v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator!=(const float3& v) const { return x != v.x || y != v.y || z != v.z; }
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
    float3 operator+(float v) const { return float3(x + v, y + v, z + v); }
    float3 operator+(float3 v) const { return float3(x + v.x, y + v.y, z + v.z); }
    void operator+=(float3 v) { x += v.x, y += v.y, z += v.z; }
    float3 operator-() const { return float3(-x, -y, -z); }
    float3 operator-(float v) const { return float3(x - v, y - v, z - v); }
    float3 operator-(float3 v) const { return float3(x - v.x, y - v.y, z - v.z); }
    void operator-=(float3 v) { x -= v.x, y -= v.y, z -= v.z; }
    float3 operator*(float s) const { return float3(x * s, y * s, z * s); }
    float3 operator*(float3 s) const { return float3(x * s.x, y * s.y, z * s.z); }
    void operator*=(float s) { x *= s, y *= s, z *= s; }
    void operator*=(float3 s) { x *= s.x, y *= s.y, z *= s.z; }
    float3 operator/(float s) const { return float3(x / s, y / s, z / s); }
    float3 operator/(float3 s) const { return float3(x / s.x, y / s.y, z / s.z); }
    void operator/=(float s) { x /= s, y /= s, z /= s; }
    void operator/=(float3 s) { x /= s.x, y /= s.y, z /= s.z; }
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, 1.0f); }
    std::string toString() const { return std::format("[{}, {}, {}]", x, y, z); }
    float dot(float3 v) const { return x * v.x + y * v.y + z * v.z; }
    float3 cross(float3 v) const { return float3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    float length() const { return sqrtf(x * x + y * y + z * z); }
    float lengthSquared() const { return x * x + y * y + z * z; }
    float3 normalize() const {
        float l = length();
        return (l > 0) ? float3(x / l, y / l, z / l) : float3(x, y, z);
    }
    float3 orthogonal() const {
        float X = abs(x), Y = abs(y), Z = abs(z);
        float3 other = X < Y ? (X < Z ? float3(1, 0, 0) : float3(0, 0, 1)) : (Y < Z ? float3(0, 1, 0) : float3(0, 0, 1));
        return this->cross(other);
    }
};

struct float4 {
    float x = 0, y = 0, z = 0, w = 1;

    float4() = default;
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    float4(const float* v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}
    float4(const XMVECTOR& v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)), w(XMVectorGetW(v)) {}
    void operator=(const XMVECTOR& v) { x = XMVectorGetX(v), y = XMVectorGetY(v), z = XMVectorGetZ(v), w = XMVectorGetW(v); }
    void operator=(const float3& v) { x = v.x, y = v.y, z = v.z, w = 0; }
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
    void operator/=(float s) { x /= s, y /= s, z /= s; w /= s; }
    void operator/=(float4 s) { x /= s.x, y /= s.y, z /= s.z; w /= s.w; }
    float3 xyz() const { return float3(x, y, z); }
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, w); }
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

#define scale 10000.0f // 0.1mm precision
#define scaleInv 0.0001f
struct Position {
    int x = 0, y = 0, z = 0;

    Position() = default;
    Position(int px, int py, int pz) : x(px), y(py), z(pz) {}
    Position(float px, float py, float pz) : x(int(px * scale)), y(int(py * scale)), z(int(pz * scale)) {}
    Position(float3 p) : x(int(p.x * scale)), y(int(p.y * scale)), z(int(p.z * scale)) {}
    Position(XMVECTOR p) : x(int(XMVectorGetX(p) * scale)), y(int(XMVectorGetY(p) * scale)), z(int(XMVectorGetZ(p) * scale)) {}
    void operator=(float3 p) { x = int(p.x * scale), y = int(p.y * scale), z = int(p.z * scale); }
    void operator+=(float3 p) { x += int(p.x * scale), y += int(p.y * scale), z += int(p.z * scale); }
    Position operator+(float3 p) const { return Position(x + int(p.x * scale), y + int(p.y * scale), z + int(p.z * scale)); }
    Position operator-() const { return Position(-x, -y, -z); }
    float3 operator-(Position p) const { return float3((x - p.x) * scaleInv, (y - p.y) * scaleInv, (z - p.z) * scaleInv); }
    float3 toFloat3() const { return float3(x * scaleInv, y * scaleInv, z * scaleInv); }
    XMVECTOR toXMVector() const { return XMVectorSet(x * scaleInv, y * scaleInv, z * scaleInv, 0); }
};
#undef scale
#undef scaleInv

struct Transform {
    float3 s = {1, 1, 1};
    float4 r = {0, 0, 0, 1};
    float3 t = {0, 0, 0};

    XMMATRIX toMat() const { return XMMatrixAffineTransformation(s.toXMVector(), XMVectorSet(0, 0, 0, 0), r.toXMVector(), t.toXMVector()); }
};

struct Plane {
    float3 n;
    float d;
};

struct AABB {
    float3 min;
    float3 max;
};

float3 lerp(float3 a, float3 b, float t) {
    return a + ((b - a) * t);
}

float4 slerp(float4 a, float4 b, float t) {
    return float4(XMQuaternionSlerp(a.toXMVector(), b.toXMVector(), t));
}

std::string toString(XMVECTOR vec) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n", XMVectorGetX(vec), XMVectorGetY(vec), XMVectorGetZ(vec), XMVectorGetW(vec));
}

std::string toString(XMMATRIX mat) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n",
                       XMVectorGetX(mat.r[0]), XMVectorGetX(mat.r[1]), XMVectorGetX(mat.r[2]), XMVectorGetX(mat.r[3]),
                       XMVectorGetY(mat.r[0]), XMVectorGetY(mat.r[1]), XMVectorGetY(mat.r[2]), XMVectorGetY(mat.r[3]),
                       XMVectorGetZ(mat.r[0]), XMVectorGetZ(mat.r[1]), XMVectorGetZ(mat.r[2]), XMVectorGetZ(mat.r[3]),
                       XMVectorGetW(mat.r[0]), XMVectorGetW(mat.r[1]), XMVectorGetW(mat.r[2]), XMVectorGetW(mat.r[3]));
}

XMVECTOR quaternionBetween(float3 v1, float3 v2) {
    float c = v1.dot(v2);
    float k = sqrtf(v1.lengthSquared() * v2.lengthSquared());
    if (c / k == -1) {
        float3 u = v1.orthogonal().normalize();
        return XMVectorSet(u.x, u.y, u.z, 0);
    } else {
        float3 u = v1.cross(v2);
        return XMQuaternionNormalize(XMVectorSet(u.x, u.y, u.z, c + k));
    }
}

float3 worldToNDC(float3 position, const XMMATRIX& cameraProjViewMat) {
    float4 ndc = XMVector4Transform(XMVectorSet(position.x, position.y, position.z, 1.0f), cameraProjViewMat);
    ndc /= ndc.w;
    return float3(ndc.x, ndc.y, ndc.z);
}

bool insideNDC(float3 position) {
    return position.x >= -1.0f && position.x <= 1.0f &&
           position.y >= -1.0f && position.y <= 1.0f &&
           position.z >= 0.0f && position.z <= 1.0f;
}

bool insideAABB(float3 position, AABB aabb) {
    return position.x >= aabb.min.x && position.y >= aabb.min.y && position.z >= aabb.min.z &&
           position.x <= aabb.max.x && position.y <= aabb.max.y && position.z <= aabb.max.z;
}

bool intersectSegmentPlane(float3 a, float3 b, Plane p, float& t, float3& q) {
    float3 ab = b - a;
    t = (p.d - p.n.dot(a)) / p.n.dot(ab);
    if (t >= 0.0f && t <= 1.0f) {
        q = a + ab * t;
        return true;
    } else {
        return false;
    }
}

bool intersectRayAABB(float3 p, float3 d, AABB a, float& tmin, float3& q) {
    tmin = 0.0f;
    float tmax = FLT_MAX;
    for (int i = 0; i < 3; i++) {
        if (fabs(d[i]) < 0.00001f /*EPSILON*/) {
            if (p[i] < a.min[i] || p[i] > a.max[i]) return false;
        } else {
            float ood = 1.0f / d[i];
            float t1 = (a.min[i] - p[i]) * ood;
            float t2 = (a.max[i] - p[i]) * ood;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) tmin = t1;
            if (t2 > tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
    }
    q = p + d * tmin;
    return true;
}

#include "structsHLSL.h"

enum WindowMode {
    WindowModeWindowed,
    WindowModeBorderless,
    WindowModeFullscreen
};

struct Settings {
    WindowMode windowMode = WindowModeWindowed;
    uint32 windowX = 0, windowY = 0;
    uint32 windowW = 1920, windowH = 1080;
    uint32 renderW = 1920, renderH = 1080;
    DXGI_RATIONAL refreshRate = {60, 1};
    bool hdr = false;
};

struct Window {
    HWND hwnd;
};

struct DisplayMode {
    uint32 resolutionW;
    uint32 resolutionH;
    std::vector<DXGI_RATIONAL> refreshRates;
};

struct D3DDescriptor {
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

struct D3D {
    IDXGIFactory7* dxgiFactory;
    IDXGIOutput6* dxgiOutput;
    IDXGIAdapter4* dxgiAdapter;
    std::vector<DisplayMode> displayModes;
    ID3D12Device5* device;

    // bool gpuUploadHeapSupported;

    ID3D12CommandQueue* graphicsQueue;
    ID3D12CommandAllocator* graphicsCmdAllocator;
    ID3D12GraphicsCommandList4* graphicsCmdList;

    ID3D12Fence* renderDoneFence;
    HANDLE renderDoneFenceEvent;
    uint64 renderDoneFenceValue;

    ID3D12Fence* collisionQueriesFence;
    HANDLE collisionQueriesFenceEvent;
    uint64 collisionQueriesFenceValue;

    ID3D12CommandQueue* transferQueue;
    ID3D12CommandAllocator* transferCmdAllocator;
    ID3D12GraphicsCommandList4* transferCmdList;

    ID3D12Fence* transferDoneFence;
    HANDLE transferDoneFenceEvent;
    uint64 transferDoneFenceCounter;

    IDXGISwapChain4* swapChain;
    DXGI_FORMAT swapChainFormat;
    ID3D12Resource* swapChainImages[2];
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainImageRTVDescriptors[2];
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainImageUAVDescriptors[2];

    ID3D12DescriptorHeap* rtvDescriptorHeap;
    uint32 rtvDescriptorCount;
    ID3D12DescriptorHeap* cbvSrvUavDescriptorHeap;
    uint32 cbvSrvUavDescriptorSize;
    uint32 cbvSrvUavDescriptorCapacity;
    uint32 cbvSrvUavDescriptorCount;

    D3D12MA::Allocator* allocator;

    D3D12MA::Allocation* stagingBuffer;
    uint8* stagingBufferPtr;
    uint32 stagingBufferOffset = 0;

    D3D12MA::Allocation* constantsBuffer;
    uint8* constantsBufferPtr;
    uint32 constantsBufferOffset = 0;

    D3D12MA::Allocation* renderTexture;
    D3D12MA::Allocation* renderTexturePrevFrame;
    DXGI_FORMAT renderTextureFormat;

    D3D12MA::Allocation* shapeCirclesBuffer;
    D3D12MA::Allocation* shapeLinesBuffer;
    uint8* shapeCirclesBufferPtr;
    uint8* shapeLinesBufferPtr;

    D3D12MA::Allocation* imguiImage;
    D3D12MA::Allocation* imguiVertexBuffer;
    D3D12MA::Allocation* imguiIndexBuffer;
    uint8* imguiVertexBufferPtr;
    uint8* imguiIndexBufferPtr;

    D3D12MA::Allocation* directWriteImage;

    D3D12MA::Allocation* defaultMaterialBaseColorImage;
    D3D12_SHADER_RESOURCE_VIEW_DESC defaultMaterialBaseColorImageSRVDesc;

    D3D12MA::Allocation* tlasInstancesBuildInfosBuffer;
    D3D12MA::Allocation* tlasInstancesInfosBuffer;
    D3D12MA::Allocation* blasGeometriesInfosBuffer;
    uint8* tlasInstancesBuildInfosBufferPtr;
    uint8* tlasInstancesInfosBufferPtr;
    uint8* blasGeometriesInfosBufferPtr;

    D3D12MA::Allocation* tlasBuffer;
    D3D12MA::Allocation* tlasScratchBuffer;

    D3D12MA::Allocation* collisionQueriesBuffer;
    CollisionQuery* collisionQueriesBufferPtr;
    D3D12MA::Allocation* collisionQueryResultsUAVBuffer;
    D3D12MA::Allocation* collisionQueryResultsBuffer;
    CollisionQueryResult* collisionQueryResultsBufferPtr;

    ID3D12StateObject* renderScenePSO;
    ID3D12StateObjectProperties* renderScenePSOProps;
    ID3D12RootSignature* renderSceneRootSig;
    void* renderSceneRayGenID;
    void* renderScenePrimaryRayMissID;
    void* renderScenePrimaryRayHitGroupID;
    void* renderSceneSecondaryRayMissID;
    void* renderSceneSecondaryRayHitGroupID;

    ID3D12StateObject* collisionQueryPSO;
    ID3D12StateObjectProperties* collisionQueryProps;
    ID3D12RootSignature* collisionQueryRootSig;
    void* collisionQueryRayGenID;
    void* collisionQueryMissID;
    void* collisionQueryHitGroupID;

    ID3D12PipelineState* vertexSkinningPSO;
    ID3D12RootSignature* vertexSkinningRootSig;

    ID3D12PipelineState* compositePSO;
    ID3D12RootSignature* compositeRootSig;

    ID3D12PipelineState* shapesPSO;
    ID3D12RootSignature* shapesRootSig;

    ID3D12PipelineState* imguiPSO;
    ID3D12RootSignature* imguiRootSig;
};

struct DirectWrite {
    ID3D11Device* d3d11Device;
    ID3D11DeviceContext* d3d11DeviceContext;
    ID3D11On12Device* d3d11On12Device;
    IDXGIDevice* dxgiDevice;

    IDWriteFactory* dWriteFactory;
    IDWriteTextFormat* dWriteTextFormat;
    ID2D1SolidColorBrush* dWriteColorBrush;

    ID2D1Factory3* d2dFactory;
    ID2D1Device1* d2dDevice;
    ID2D1DeviceContext1* d2dDeviceContext;

    ID3D11Resource* image;
    IDXGISurface* imageSurface;
    ID2D1Bitmap1* imageRenderTarget;

    // ID2D1HwndRenderTarget* d2dRenderTarget;
    // ID2D1SolidColorBrush* d2dBrush;
};

struct ModelImage {
    D3D12MA::Allocation* gpuData;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
};

struct ModelTextureSampler {
};

struct ModelTexture {
    ModelImage* image;
    ModelTextureSampler sampler;
};

struct ModelMaterial {
    std::string name;
    float4 baseColorFactor = {1, 1, 1, 1};
    ModelTexture* baseColorTexture;
};

struct ModelPrimitive {
    uint32 verticesBufferOffset;
    uint32 verticesCount;
    uint32 indicesBufferOffset;
    uint32 indicesCount;
    ModelMaterial* material;
};

struct ModelMesh {
    std::string name;
    std::vector<ModelPrimitive> primitives;
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    D3D12MA::Allocation* verticesBuffer;
    D3D12MA::Allocation* indicesBuffer;
    D3D12MA::Allocation* blas;
    D3D12MA::Allocation* blasScratch;
};

struct ModelNode;

struct ModelJoint {
    ModelNode* node;
    XMMATRIX inverseBindMat;
};

struct ModelSkin {
    std::vector<ModelJoint> joints;
};

struct ModelNode {
    std::string name;
    ModelNode* parent;
    std::vector<ModelNode*> children;
    XMMATRIX globalTransform;
    XMMATRIX localTransform;
    ModelMesh* mesh;
    ModelSkin* skin;
};

enum ModelAnimationSamplerInterpolation {
    AnimationSamplerInterpolationLinear,
    AnimationSamplerInterpolationStep,
    AnimationSamplerInterpolationCubicSpline,
};

struct ModelAnimationSamplerKeyFrame {
    float time = 0;
    float4 xyzw = {0, 0, 0, 0};
};

struct ModelAnimationSampler {
    ModelAnimationSamplerInterpolation interpolation;
    std::vector<ModelAnimationSamplerKeyFrame> keyFrames;
};

enum ModelAnimationChannelType {
    AnimationChannelTypeTranslate,
    AnimationChannelTypeRotate,
    AnimationChannelTypeScale
};

struct ModelAnimationChannel {
    ModelNode* node;
    ModelAnimationSampler* sampler;
    ModelAnimationChannelType type;
};

struct ModelAnimation {
    std::string name;
    std::vector<ModelAnimationChannel> channels;
    std::vector<ModelAnimationSampler> samplers;
    double timeLength;
};

struct Model {
    std::filesystem::path filePath;
    cgltf_data* gltfData;
    std::vector<ModelNode> nodes;
    std::vector<ModelNode*> rootNodes;
    std::vector<ModelNode*> meshNodes;
    std::vector<ModelMesh> meshes;
    std::vector<ModelSkin> skins;
    ModelNode* skeletonRootNode;
    std::vector<ModelAnimation> animations;
    std::vector<ModelMaterial> materials;
    std::vector<ModelTexture> textures;
    std::vector<ModelImage> images;
};

struct ModelInstanceSkin {
    std::vector<XMMATRIX> mats;
    D3D12MA::Allocation* matsBuffer;
    uint8* matsBufferPtr;
};

struct ModelInstanceMeshNode {
    XMMATRIX transformMat;
    D3D12MA::Allocation* verticesBuffer;
    D3D12MA::Allocation* blas;
    D3D12MA::Allocation* blasScratch;
};

struct ModelInstance {
    Model* model;
    Transform transform;
    std::vector<ModelInstanceMeshNode> meshNodes;
    ModelAnimation* animation;
    double animationTime;
    std::vector<Transform> localTransforms;
    std::vector<XMMATRIX> globalTransformMats;
    std::vector<ModelInstanceSkin> skins;
};

struct CameraPlayer {
    Position position;
    Position lookAt;
    float3 lookAtOffset;
    float2 pitchYaw;
    float distance;
    float fovVertical = 50;
};

struct CameraEditor {
    Position position;
    Position lookAt;
    float2 pitchYaw;
    float fovVertical = 50;
    float moveSpeed = 1;
};

enum PlayerState {
    PlayerStateIdle,
    PlayerStateWalk,
    PlayerStateRun,
    PlayerStateJump,
};

struct PlayerStateTransition {
};

struct Player {
    ModelInstance modelInstance;
    Position spawnPosition;
    Position position;
    float3 PitchYawRoll;
    float walkSpeed;
    float runSpeed;
    float3 movement;
    PlayerState state;
    float3 velocity;
    float3 acceleration;
    uint32 idleAnimationIndex;
    uint32 walkAnimationIndex;
    uint32 runAnimationIndex;
    uint32 jumpAnimationIndex;
    CameraPlayer camera;
};

struct Skybox {
    std::filesystem::path hdriTextureFilePath;
    D3D12MA::Allocation* hdriTexture;
};

struct GameObject {
    std::string name;
    ModelInstance modelInstance;
    Position spawnPosition;
    Position position;
    bool toBeDeleted;
};

enum EditorUndoType {
    WorldEditorUndoTypeObjectDeletion
};

struct EditorUndoObjectDeletion {
    ObjectType objectType;
    void* object;
};

struct EditorUndo {
    EditorUndoType type;
    union {
        EditorUndoObjectDeletion* objectDeletion;
    };
};

struct Editor {
    bool active = true;
    CameraEditor camera;
    bool cameraMoving;
    ObjectType selectedObjectType;
    uint32 selectedObjectIndex;
    std::stack<EditorUndo> undos;
};

struct Controller {
    bool back, start;
    bool a, b, x, y;
    bool up, down, left, right;
    bool lb, rb;
    bool ls, rs;
    float lt, rt;
    float lsX, lsY, rsX, rsY;

    float backDownDuration, startDownDuration;
    float aDownDuration, bDownDuration, xDownDuration, yDownDuration;
    float upDownDuration, downDownDuration, leftDownDuration, rightDownDuration;
    float lbDownDuration, rbDownDuration;
    float lsDownDuration, rsDownDuration;
};
