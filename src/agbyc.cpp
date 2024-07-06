#include "pch.h"

#include <rapidyaml/rapidyaml-0.5.0.hpp>

#include <cgltf/cgltf.h>

#include <ufbx/ufbx.h>

#include <d3d12ma/d3d12memalloc.h>

#include <stb/stb_image.h>

#include <imgui/imconfig.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imguizmo.h>

#include <dlss/nvsdk_ngx.h>
#include <dlss/nvsdk_ngx_defs.h>
#include <dlss/nvsdk_ngx_params.h>
#include <dlss/nvsdk_ngx_helpers.h>

#define TRACY_ENABLE
#include <tracy/tracy/tracy.hpp>
#define EDITOR
#define LIVE_RELOAD_SHADERS
#define LIVE_RELOAD_FUNCS
#define PHYSX_PVD

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

static const XMMatrix xmMatrixIdentity = XMMatrixIdentity();
static const XMVector xmVectorZero = XMVectorSet(0, 0, 0, 0);
static const XMVector xmVectorOne = XMVectorSet(1, 1, 1, 1);
static const XMVector xmQuatIdentity = XMVectorSet(0, 0, 0, 1);

constexpr float radian(float d) { return d * (pi / 180.0f); }
constexpr float degree(float r) { return r * (180.0f / pi); }

constexpr uint64 kilobytes(uint64 n) { return n * 1024ll; }
constexpr uint64 megabytes(uint64 n) { return n * 1024ll * 1024ll; }
constexpr uint64 gigabytes(uint64 n) { return n * 1024ll * 1024ll * 1024ll; }

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

struct uint8_2 {
    uint8 x = 0, y = 0;
    std::string toString() const { return std::format("[{}, {}]", x, y); }
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
    float2(XMVector v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)) {}
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
    void operator*=(float v) { x *= v, y *= v; }
    void operator*=(float2 v) { x *= v.x, y *= v.y; }
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
    float3(XMVector v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)) {}
    float3(PxVec3 p) : x(p.x), y(p.y), z(p.z) {}
    float3(PxVec3d p) : x((float)p.x), y((float)p.y), z((float)p.z) {}
    void operator=(XMVector v) { x = XMVectorGetX(v), y = XMVectorGetY(v), z = XMVectorGetZ(v); }
    void operator=(PxVec3 v) { x = v.x, y = v.y, z = v.z; }
    bool operator==(float3 v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator!=(float3 v) const { return x != v.x || y != v.y || z != v.z; }
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
    float2 xy() const { return float2(x, y); }
    XMVector toXMVector() const { return XMVectorSet(x, y, z, 1.0f); }
    PxVec3 toPxVec3() const { return PxVec3(x, y, z); }
    PxVec3d toPxVec3d() const { return PxVec3d(x, y, z); }
    std::string toString() const { return std::format("[{}, {}, {}]", x, y, z); }
    float max() const { return std::max(std::max(x, y), z); }
    float min() const { return std::min(std::min(x, y), z); }
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
    float4(XMVector v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)), w(XMVectorGetW(v)) {}
    void operator=(XMVector v) { x = XMVectorGetX(v), y = XMVectorGetY(v), z = XMVectorGetZ(v), w = XMVectorGetW(v); }
    void operator=(PxVec4 v) { x = v.x, y = v.y, z = v.z, w = v.w; }
    void operator=(PxQuat v) { x = v.x, y = v.y, z = v.z, w = v.w; }
    void operator=(float3 v) { x = v.x, y = v.y, z = v.z, w = 0; }
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
    void operator/=(float s) { x /= s, y /= s, z /= s, w /= s; }
    void operator/=(float4 s) { x /= s.x, y /= s.y, z /= s.z, w /= s.w; }
    float2 xy() const { return float2(x, y); }
    float3 xyz() const { return float3(x, y, z); }
    XMVector toXMVector() const { return XMVectorSet(x, y, z, w); }
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct Transform {
    float3 s = {1, 1, 1};
    float4 r = {0, 0, 0, 1};
    float3 t = {0, 0, 0};

    Transform() = default;
    Transform(XMMatrix mat) {
        XMVector scaling, rotation, translation;
        XMMatrixDecompose(&scaling, &rotation, &translation, mat);
        s = scaling, r = rotation, t = translation;
    }
    Transform(PxTransform t) : s(1, 1, 1), r(t.q.x, t.q.y, t.q.z, t.q.w), t(t.p.x, t.p.y, t.p.z) {}
    PxTransform toPxTransform() const { return PxTransform(t.x, t.y, t.z, PxQuat(r.x, r.y, r.z, r.w)); }
    XMMatrix toMat() const { return XMMatrixAffineTransformation(s.toXMVector(), xmVectorZero, r.toXMVector(), t.toXMVector()); }
};

struct Plane {
    float3 n = {0, 1, 0};
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

std::string toString(XMVector vec) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n", XMVectorGetX(vec), XMVectorGetY(vec), XMVectorGetZ(vec), XMVectorGetW(vec));
}

std::string toString(XMMatrix mat) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n",
                       XMVectorGetX(mat.r[0]), XMVectorGetX(mat.r[1]), XMVectorGetX(mat.r[2]), XMVectorGetX(mat.r[3]),
                       XMVectorGetY(mat.r[0]), XMVectorGetY(mat.r[1]), XMVectorGetY(mat.r[2]), XMVectorGetY(mat.r[3]),
                       XMVectorGetZ(mat.r[0]), XMVectorGetZ(mat.r[1]), XMVectorGetZ(mat.r[2]), XMVectorGetZ(mat.r[3]),
                       XMVectorGetW(mat.r[0]), XMVectorGetW(mat.r[1]), XMVectorGetW(mat.r[2]), XMVectorGetW(mat.r[3]));
}

XMVector quaternionBetween(float3 v1, float3 v2) {
    float c = v1.dot(v2);
    float k = sqrtf(v1.lengthSquared() * v2.lengthSquared());
    if (((c / k) - (-1.0f)) < 0.000001f) {
        float3 u = v1.orthogonal().normalize();
        return XMVectorSet(u.x, u.y, u.z, 0);
    }
    else {
        float3 u = v1.cross(v2);
        return XMQuaternionNormalize(XMVectorSet(u.x, u.y, u.z, c + k));
    }
}

float4 quaternionFromEulerAngles(float3 eulerAngles) {
    float cr = cosf(eulerAngles.x * 0.5f);
    float sr = sinf(eulerAngles.x * 0.5f);
    float cp = cosf(eulerAngles.y * 0.5f);
    float sp = sinf(eulerAngles.y * 0.5f);
    float cy = cosf(eulerAngles.z * 0.5f);
    float sy = sinf(eulerAngles.z * 0.5f);

    float4 q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

float3 quaternionToEulerAngles(float4 q) {
    float3 angles;

    float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    angles.x = atan2f(sinr_cosp, cosr_cosp);

    float sinp = sqrtf(1.0f + 2.0f * (q.w * q.y - q.x * q.z));
    float cosp = sqrtf(1.0f - 2.0f * (q.w * q.y - q.x * q.z));
    angles.y = 2.0f * atan2f(sinp, cosp) - pi / 2.0f;

    float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    angles.z = atan2f(siny_cosp, cosy_cosp);

    return angles;
}

float2 ndcToScreen(float2 ndc, float2 screenSize) {
    float2 coord = (float2(ndc.x, -ndc.y) + 1.0f) * 0.5f;
    return coord * screenSize;
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

bool intersectSegmentPlane(float3 a, float3 b, Plane p, float* t, float3* q) {
    float3 ab = b - a;
    *t = (p.d - p.n.dot(a)) / p.n.dot(ab);
    if (*t >= 0.0f && *t <= 1.0f) {
        *q = a + ab * t;
        return true;
    }
    else {
        return false;
    }
}

bool intersectRayAABB(float3 p, float3 d, AABB a, float* tmin, float3* q) {
    *tmin = 0.0f;
    float tmax = FLT_MAX;
    for (int i = 0; i < 3; i++) {
        if (fabs(d[i]) < 0.00001f) {
            if (p[i] < a.min[i] || p[i] > a.max[i]) return false;
        }
        else {
            float ood = 1.0f / d[i];
            float t1 = (a.min[i] - p[i]) * ood;
            float t2 = (a.max[i] - p[i]) * ood;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > *tmin) *tmin = t1;
            if (t2 > tmax) tmax = t2;
            if (*tmin > tmax) return false;
        }
    }
    *q = p + d * *tmin;
    return true;
}

uint32 ARGBToABGR(uint32 argb) {
    uint32 a_g_ = argb & 0xff00ff00;
    uint32 _b__ = argb << 16 & 0x00ff0000;
    uint32 ___r = argb >> 16 & 0x000000ff;
    return a_g_ | _b__ | ___r;
}

std::string getLastErrorStr() {
    DWORD err = GetLastError();
    std::string message = std::system_category().message(err);
    return message;
}

std::string fileReadStr(const std::filesystem::path& path) {
    std::ifstream file;
    file.open(path, std::ios::in);
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return str;
}

std::vector<uint8> fileReadBytes(const std::filesystem::path& path) {
    HANDLE hwnd = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    assert(hwnd != INVALID_HANDLE_VALUE);
    DWORD size = GetFileSize(hwnd, nullptr);
    assert(size != INVALID_FILE_SIZE);
    std::vector<uint8> data(size);
    DWORD byteRead;
    assert(ReadFile(hwnd, data.data(), size, &byteRead, nullptr));
    assert(byteRead == size);
    CloseHandle(hwnd);
    return data;
}

void fileWriteStr(const std::filesystem::path& path, const std::string& str) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    file << str;
}

void fileWriteBytes(const std::filesystem::path& path, void* data, uint64 size) {
    std::ofstream file(path, std::ios::out | std::ios::trunc | std::ios::binary);
    file.write((char*)data, size);
}

bool commandLineContain(const wchar_t* arg) {
    int argsCount;
    LPWSTR* args = CommandLineToArgvW(GetCommandLineW(), &argsCount);
    for (int i = 1; i < argsCount; i++) {
        if (wcscmp(arg, args[i]) == 0) return true;
    }
    return false;
}

void showConsole() {
    if (AllocConsole()) {
        freopen_s((FILE**)stdin, "CONIN$", "r", stdin);
        freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
        freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);
        HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
        // HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        // HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
        assert(SetConsoleMode(hStdin, ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));
    }
}

template <typename T>
struct ArenaElementHandle {
    uint index;
    uint generation;
};

template <typename T>
struct ArenaElement {
    T element;
    bool valid;
    uint generation;
};

template <typename T>
struct Arena {
    std::vector<ArenaElement<T>> elements;
    std::stack<uint> freeSlots;

    ArenaElementHandle<T> add(const T& newElement) {
        if (freeSlots.empty()) {
            elements.push_back(ArenaElement{newElement, true, 0});
            return ArenaElementHandle<T>{(uint)(elements.size() - 1), 0};
        }
        else {
            uint index = freeSlots.top();
            freeSlots.pop();
            elements[index].element = newElement;
            elements[index].valid = true;
            return ArenaElementHandle<T>{index, elements[index].generation};
        }
    }
    void remove(uint index) {
        if (elements[index].valid) {
            elements[index].generation += 1;
            elements[index].valid = false;
            freeSlots.push(index);
        }
    }
    T* get(ArenaElementHandle<T> handle) {
        if (elements[handle.index].generation == handle.generation) {
            return &elements[handle.index].element;
        }
        else {
            return nullptr;
        }
    }
};

#include "hlsl/shared.h"

enum WindowMode {
    WindowModeWindowed,
    WindowModeBorderless,
};

struct Settings {
    WindowMode windowMode = WindowModeWindowed;
    uint32 windowX = 0, windowY = 0;
    uint32 windowW = 1920, windowH = 1080;
    uint32 renderW = 1920, renderH = 1080;
    bool hdr = false;
};

struct Window {
    HWND hwnd;
};

struct D3DDescriptor {
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

struct D3DFence {
    ID3D12Fence* fence;
    HANDLE event;
    uint64 value;
};

struct D3DDescriptorHeap {
    ID3D12DescriptorHeap* heap;
    uint32 size;
    uint32 capacity;
    uint32 descriptorSize;
};

struct D3DUploadBuffer {
    D3D12MA::Allocation* buffer;
    uint8* ptr;
    uint64 size;
    uint64 capacity;
};

struct D3DReadBackBuffer {
    D3D12MA::Allocation* bufferUAV;
    D3D12MA::Allocation* buffer;
    uint8* ptr;
    uint64 capacity;
};

struct D3D {
    IDXGIFactory7* dxgiFactory;
    IDXGIAdapter4* dxgiAdapter;
    ID3D12Device5* device;

    ID3D12CommandQueue* graphicsQueue;
    ID3D12CommandAllocator* graphicsCmdAllocator;
    ID3D12CommandAllocator* graphicsCmdAllocatorPrevFrame;
    ID3D12GraphicsCommandList4* graphicsCmdList;
    ID3D12GraphicsCommandList4* graphicsCmdListPrevFrame;

    D3DFence transferFence;
    D3DFence renderFence;
    D3DFence renderFencePrevFrame;
    D3DFence collisionQueriesFence;

    IDXGISwapChain4* swapChain;
    const DXGI_FORMAT swapChainFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
    ID3D12Resource* swapChainImages[2];
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainImageRTVDescriptors[2];

    D3DDescriptorHeap rtvDescriptorHeap;
    D3DDescriptorHeap cbvSrvUavDescriptorHeap;
    D3DDescriptorHeap cbvSrvUavDescriptorHeapPrevFrame;

    D3D12MA::Allocator* allocator;

    D3DUploadBuffer stagingBuffer;
    D3DUploadBuffer constantsBuffer;
    D3DUploadBuffer constantsBufferPrevFrame;

    D3D12MA::Allocation* renderTexture;
    D3D12MA::Allocation* renderTexturePrevFrame;
    const DXGI_FORMAT renderTextureFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

    D3D12MA::Allocation* pathTracerAccumulationTexture;
    const DXGI_FORMAT pathTracerAccumulationTextureFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uint pathTracerAccumulationCount;
    const uint pathTracerAccumulationCountMax = 10'000'000;

    D3D12MA::Allocation* depthTexture;
    const DXGI_FORMAT depthTextureFormat = DXGI_FORMAT_R32_FLOAT;

    D3D12MA::Allocation* motionVectorTexture;
    const DXGI_FORMAT motionVectorTextureFormat = DXGI_FORMAT_R32G32_FLOAT;

    D3D12MA::Allocation* imguiTexture;
    const DXGI_FORMAT imguiTextureFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    D3DUploadBuffer imguiVertexBuffer;
    D3DUploadBuffer imguiVertexBufferPrevFrame;
    D3DUploadBuffer imguiIndexBuffer;
    D3DUploadBuffer imguiIndexBufferPrevFrame;

    D3D12MA::Allocation* defaultEmissiveTexture;
    D3D12MA::Allocation* defaultBaseColorTexture;
    D3D12MA::Allocation* defaultMetallicRoughnessTexture;
    D3D12MA::Allocation* defaultNormalTexture;
    D3D12_SHADER_RESOURCE_VIEW_DESC defaultEmissiveTextureSRVDesc;
    D3D12_SHADER_RESOURCE_VIEW_DESC defaultBaseColorTextureSRVDesc;
    D3D12_SHADER_RESOURCE_VIEW_DESC defaultMetallicRoughnessTextureSRVDesc;
    D3D12_SHADER_RESOURCE_VIEW_DESC defaultNormalTextureSRVDesc;

    D3DUploadBuffer blasInstanceDescsBuffer;
    D3DUploadBuffer blasInstanceDescsBufferPrevFrame;
    D3DUploadBuffer blasInstancesInfosBuffer;
    D3DUploadBuffer blasInstancesInfosBufferPrevFrame;
    D3DUploadBuffer blasGeometriesInfosBuffer;
    D3DUploadBuffer blasGeometriesInfosBufferPrevFrame;

    D3D12MA::Allocation* tlasBuffer;
    D3D12MA::Allocation* tlasScratchBuffer;

    D3DUploadBuffer collisionQueriesBuffer;
    D3DReadBackBuffer collisionQueryResultsBuffer;

    ID3D12StateObject* renderScenePSO;
    ID3D12StateObjectProperties* renderScenePSOProps;
    ID3D12RootSignature* renderSceneRootSig;
    void* renderSceneRayGenID;
    void* renderSceneRayMissIDPrimary;
    void* renderSceneRayHitGroupIDPrimary;
    void* renderSceneRayMissIDShadow;
    void* renderSceneRayHitGroupIDShadow;

    ID3D12StateObject* pathTracerPSO;
    ID3D12StateObjectProperties* pathTracerPSOProps;
    ID3D12RootSignature* pathTracerRootSig;
    void* pathTracerRayGenID;
    void* pathTracerRayMissID;
    void* pathTracerRayHitGroupID;

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

    ID3D12PipelineState* imguiPSO;
    ID3D12RootSignature* imguiRootSig;
};

struct Sphere {
    float3 center;
    float radius;
    uint32 color;
};

struct Line {
    float3 p0, p1;
    float radius;
    uint32 color;
};

struct Triangle {
    float3 p0, p1, p2;
    uint32 color;
};

struct ModelImage {
    D3D12MA::Allocation* gpuData;
};

struct ModelTextureSampler {
};

struct ModelTexture {
    ModelImage* image;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    D3D12_TEXTURE_ADDRESS_MODE wrapU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    D3D12_TEXTURE_ADDRESS_MODE wrapV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
};

struct ModelMaterial {
    std::string name;
    float3 emissive = {0, 0, 0};
    float4 baseColor = {1, 1, 1, 1};
    float metallic;
    float roughness;
    ModelTexture* emissiveTexture;
    ModelTexture* baseColorTexture;
    ModelTexture* metallicRoughnessTexture;
    ModelTexture* normalTexture;
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
    uint blasGeometriesInfoOffset;
    D3D12MA::Allocation* verticesBuffer;
    D3D12MA::Allocation* indicesBuffer;
    D3D12MA::Allocation* blas;
    D3D12MA::Allocation* blasScratch;
};

struct ModelNode;

struct ModelJoint {
    ModelNode* node;
    XMMatrix inverseBindMat;
};

struct ModelSkin {
    std::vector<ModelJoint> joints;
};

struct ModelNode {
    std::string name;
    ModelNode* parent;
    std::vector<ModelNode*> children;
    XMMatrix globalTransform;
    XMMatrix localTransform;
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
    std::string filePathStr;
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
    PxConvexMesh* convexMesh;
    PxTriangleMesh* triangleMesh;
};

struct ModelInstanceSkin {
    std::vector<XMMatrix> mats;
    D3D12MA::Allocation* matsBuffer;
    uint8* matsBufferPtr;
};

struct ModelInstanceMeshNode {
    D3D12MA::Allocation* verticesBuffer;
    D3D12MA::Allocation* blas;
    D3D12MA::Allocation* blasScratch;
};

struct ModelInstance {
    Model* model;
    std::vector<ModelInstanceMeshNode> meshNodes;
    ModelAnimation* animation;
    double animationTime;
    std::vector<Transform> localTransforms;
    std::vector<XMMatrix> globalTransformMats;
    std::vector<ModelInstanceSkin> skins;
};

struct CameraPlayer {
    float3 position;
    float3 lookAt;
    float2 pitchYaw;
    float fovVertical = 50.0f;
    float3 lookAtOffset;
    float distance;
};

struct CameraEditor {
    float3 position;
    float3 lookAt;
    float2 pitchYaw;
    float fovVertical = 50.0f;
    bool moving;
    float moveSpeed = 5;
    float moveSpeedMax = 500;
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
    Transform transformDefault;
    Transform transform;
    Transform transformPrevFrame;
    float walkSpeed;
    float runSpeed;
    float3 velocity;
    PlayerState state;
    uint32 idleAnimationIndex;
    uint32 walkAnimationIndex;
    uint32 runAnimationIndex;
    uint32 jumpAnimationIndex;
    CameraPlayer camera;
    PxCapsuleController* pxController;
};

struct GameObject {
    std::string name;
    ModelInstance modelInstance;
    Transform transformDefault;
    Transform transform;
    Transform transformPrevFrame;
    PxRigidActor* rigidActor;
    bool toBeDeleted;
};

struct Skybox {
    std::filesystem::path hdriTextureFilePath;
    D3D12MA::Allocation* hdriTexture;
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

enum EditorUndoType {
    EditorUndoTypeObjectDeletion
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

enum EditorMode {
    EditorModeFreeCam,
    EditorModeEditObject,
};

struct Editor {
    EditorMode mode = EditorModeFreeCam;
    CameraEditor camera;
    CameraEditor editCamera;
    ObjectType selectedObjectType = ObjectTypeNone;
    uint selectedObjectIndex = UINT_MAX;
    bool selectedObjectXRay = false;
    uint renameObjectIndex = UINT_MAX;
    ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmoMode = ImGuizmo::LOCAL;
    bool beginDragDropGameObject;
    GameObject dragDropGameObject;
    uint selectedModelIndex = UINT_MAX;
    std::stack<EditorUndo> undos;
};

// GLOBALS
static std::filesystem::path exeDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, countof(buf));
    assert(n < countof(buf));
    std::filesystem::path path(buf);
    return path.parent_path();
}();

static std::filesystem::path saveDir = [] {
    wchar_t* documentFolderPathStr;
    HRESULT result = SHGetKnownFolderPath(FOLDERID_SavedGames, KF_FLAG_DEFAULT, nullptr, &documentFolderPathStr);
    assert(result == S_OK);
    std::filesystem::path documentFolderPath(documentFolderPathStr);
    CoTaskMemFree(documentFolderPathStr);
    documentFolderPath = documentFolderPath / "AGBY_GAME_SAVES";
    std::error_code err;
    bool createSuccess = std::filesystem::create_directory(documentFolderPath, err);
    if (!createSuccess) {
        assert(std::filesystem::exists(documentFolderPath));
    }
    return documentFolderPath;
}();

static std::filesystem::path worldFilePath = exeDir / "assets/worlds/world_1.yaml";
static std::filesystem::path gameSavePath = saveDir / "save.yaml";
static std::filesystem::path settingsPath = saveDir / "settings.yaml";

static bool quit;
static LARGE_INTEGER perfFrequency;
static LARGE_INTEGER perfCounterStart;
static LARGE_INTEGER perfCounterEnd;
static float frameTime;
static uint mouseSelectX = UINT_MAX;
static uint mouseSelectY = UINT_MAX;
static bool mouseSelectOngoing = false;
static int2 mouseDeltaRaw;
static float mouseWheel;
static float mouseSensitivity = 0.001f;
static float controllerSensitivity = 2.0f;

static HRESULT setDPIAwareness = SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
static int screenW = GetSystemMetrics(SM_CXSCREEN);
static int screenH = GetSystemMetrics(SM_CYSCREEN);
static WindowMode windowMode = WindowModeWindowed;
static int windowX = 0, windowY = 0;
static int windowW = 1920, windowH = 1080;
static int renderW = 1920, renderH = 1080;
static DXGI_RATIONAL refreshRate = {60, 1};
static bool hdr;
static Window window;

static D3D d3d;
static IGameInput* gameInput;
static Controller controller;
static float controllerDeadZone = 0.25f;
static HANDLE controllerDualSenseHID;

static ImFont* imguiFont;

static NVSDK_NGX_Parameter* ngxParameter;
static int dlssAvaliable;

static std::vector<Skybox> skyboxes = [] { std::vector<Skybox> skyboxes; skyboxes.reserve(16); return skyboxes; }();
static std::vector<Model> models = [] { std::vector<Model> models; models.reserve(2048); return models; }();
static Skybox* skybox;
static ModelInstance modelInstanceSphere;   /* diameter 1 meter */
static ModelInstance modelInstanceCube;     /* 1 meter w/h/d */
static ModelInstance modelInstanceCylinder; /* length 1 meter, diameter 1 meter */
static Player player;
static const uint gameObjectsMaxCount = 10'000;
static std::vector<GameObject> gameObjects = [] { std::vector<GameObject> gameObjects; gameObjects.reserve(gameObjectsMaxCount); return gameObjects; }();
static std::vector<GameObject*> gameObjectsWithDynamicRigidBody;
static std::vector<Light> lights;

static std::vector<D3D12_RAYTRACING_INSTANCE_DESC> blasInstancesDescs;
static std::vector<BLASInstanceInfo> blasInstancesInfos;
static std::vector<BLASGeometryInfo> blasGeometriesInfos;

static std::vector<Sphere> debugSpheres;
static std::vector<Line> debugLines;
static std::vector<Triangle> debugTriangles;

static bool pathTracer;

static XMMatrix cameraViewMat;
static XMMatrix cameraViewMatInverseTranspose;
static XMMatrix cameraProjectMat;
static XMMatrix cameraViewProjectMat;
static XMMatrix cameraViewProjectMatInverse;
static XMMatrix cameraViewProjectMatPrevFrame;
static uint64 frameCountWithoutCameraCut;

static bool showMenu;
static bool showRigidActorsGeometries;

static PxDefaultAllocator pxAllocator;
static PxDefaultErrorCallback pxErrorCallback;
static PxFoundation* pxFoundation;
static PxPhysics* pxPhysics;
static PxDefaultCpuDispatcher* pxDispatcher;
static PxScene* pxScene;
static float pxTimeStep = 1.0f / 60.0f;
static float pxTimeAccumulated = 0.0f;
static PxMaterial* pxDefaultMaterial;
static PxControllerManager* pxControllerManager;

#ifdef EDITOR
static Editor editor;
static bool editorActive = true;
#endif

void settingsInit() {
    if (std::filesystem::is_regular_file(settingsPath)) {
        std::string yamlStr = fileReadStr(settingsPath);
        ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
        ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
        yamlRoot["hdr"] >> hdr;
        yamlRoot["windowX"] >> windowX;
        yamlRoot["windowY"] >> windowY;
        yamlRoot["windowW"] >> windowW;
        yamlRoot["windowH"] >> windowH;
    }
}

void settingsSave() {
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;
    yamlRoot["hdr"] << hdr;
    yamlRoot["windowX"] << windowX;
    yamlRoot["windowY"] << windowY;
    yamlRoot["windowW"] << windowW;
    yamlRoot["windowH"] << windowH;
    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(settingsPath, yamlStr);
}

void windowUpdateSizes() {
    RECT windowRect;
    RECT clientRect;
    assert(GetWindowRect(window.hwnd, &windowRect));
    assert(GetClientRect(window.hwnd, &clientRect));
    windowX = windowRect.left;
    windowY = windowRect.top;
    windowW = windowRect.right - windowRect.left;
    windowH = windowRect.bottom - windowRect.top;
    renderW = clientRect.right - clientRect.left;
    renderH = clientRect.bottom - clientRect.top;
}

void windowInit() {
    HMODULE instanceHandle = GetModuleHandle(nullptr);
    WNDCLASSA windowClass = {};
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    LRESULT windowEventHandler(HWND hwnd, UINT eventType, WPARAM wparam, LPARAM lparam);
    windowClass.lpfnWndProc = windowEventHandler;
    windowClass.hInstance = instanceHandle;
    windowClass.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    windowClass.lpszClassName = "windowClassName";
    assert(RegisterClassA(&windowClass));

    window.hwnd = CreateWindowExA(0, windowClass.lpszClassName, nullptr, WS_OVERLAPPEDWINDOW, windowX, windowY, windowW, windowH, nullptr, nullptr, instanceHandle, nullptr);
    assert(window.hwnd);

    windowUpdateSizes();

    RAWINPUTDEVICE rawInputDeviceMouse = {.usUsagePage = 0x0001, .usUsage = 0x0002, .hwndTarget = window.hwnd};
    assert(RegisterRawInputDevices(&rawInputDeviceMouse, 1, sizeof(rawInputDeviceMouse)));

    RAWINPUTDEVICE rawInputDeviceController = {.usUsagePage = 0x0001, .usUsage = 0x0005, .dwFlags = RIDEV_DEVNOTIFY, .hwndTarget = window.hwnd};
    assert(RegisterRawInputDevices(&rawInputDeviceController, 1, sizeof(rawInputDeviceController)));
}

void windowShow() {
    ShowWindow(window.hwnd, SW_SHOW);
}

void windowClipCursor(bool clip) {
    if (clip) {
        RECT windowRect;
        assert(GetWindowRect(window.hwnd, &windowRect));
        POINT cursorP = {0, 0};
        GetCursorPos(&cursorP);
        if (cursorP.x <= windowRect.left || cursorP.x >= windowRect.right || cursorP.y <= windowRect.top || cursorP.y >= windowRect.bottom) {
            cursorP = {windowRect.left + (windowRect.right - windowRect.left) / 2, windowRect.top + (windowRect.bottom - windowRect.top) / 2};
            SetCursorPos(cursorP.x, cursorP.y);
        }
        RECT rect = {cursorP.x, cursorP.y, cursorP.x, cursorP.y};
        ClipCursor(&rect);
    }
    else {
        ClipCursor(nullptr);
    }
}

void windowHideCursor(bool hide) {
    CURSORINFO cursorInfo = {.cbSize = sizeof(CURSORINFO)};
    if (GetCursorInfo(&cursorInfo)) {
        if (hide) {
            windowClipCursor(true);
            if (cursorInfo.flags == CURSOR_SHOWING) {
                ShowCursor(false);
            }
        }
        else {
            windowClipCursor(false);
            if (cursorInfo.flags != CURSOR_SHOWING) {
                ShowCursor(true);
            }
        }
    }
}

void imguiInit() {
    assert(ImGui::CreateContext());
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();
    // ImGui::StyleColorsClassic();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = _strdup((exeDir / "imgui.ini").string().c_str());
    // assert(io.Fonts->AddFontDefault());
    imguiFont = io.Fonts->AddFontFromFileTTF((exeDir / "assets/fonts/NotoSerif.ttf").string().c_str(), 50);
    io.FontGlobalScale = (float)screenH / 3000.0f;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void gameInputInit() {
    assert(GameInputCreate(&gameInput) == S_OK);
};

void controllerApplyDeadZone(Controller* c) {
    float lDistance = sqrtf(c->lsX * c->lsX + c->lsY * c->lsY);
    if (lDistance > 0) {
        float lDistanceNew = std::max(0.0f, lDistance - controllerDeadZone) / (1.0f - controllerDeadZone);
        c->lsX = c->lsX / lDistance * lDistanceNew;
        c->lsY = c->lsY / lDistance * lDistanceNew;
    }
    float rDistance = sqrtf(c->rsX * c->rsX + c->rsY * c->rsY);
    if (rDistance > 0) {
        float rDistanceNew = std::max(0.0f, rDistance - controllerDeadZone) / (1.0f - controllerDeadZone);
        c->rsX = c->rsX / rDistance * rDistanceNew;
        c->rsY = c->rsY / rDistance * rDistanceNew;
    }
}

bool controllerStickMoved() {
    return controller.lsX != 0 || controller.lsY != 0 || controller.rsX != 0 || controller.rsY != 0;
}

std::string controllerToString() {
    std::string s = std::format("ls({}, {}) rs({}, {})\nlt({}) rt({})\n", controller.lsX, controller.lsY, controller.rsX, controller.rsY, controller.lt, controller.rt);
    if (controller.a) s += "A ";
    if (controller.b) s += "B ";
    if (controller.x) s += "X ";
    if (controller.y) s += "Y ";
    if (controller.up) s += "up ";
    if (controller.down) s += "down ";
    if (controller.left) s += "left ";
    if (controller.right) s += "right ";
    if (controller.lb) s += "lb ";
    if (controller.rb) s += "rb ";
    if (controller.ls) s += "ls ";
    if (controller.rs) s += "rs ";
    if (controller.back) s += "back ";
    if (controller.start) s += "start ";
    s += '\n';
    return s;
}

Controller controllerGetStateDualSense(uint8* packet, uint packetSize) {
    Controller c = {};
    uint n = (packetSize == 64) ? 0 : 1;
    c.lsX = (packet[n + 1] / 255.0f) * 2.0f - 1.0f;
    c.lsY = -((packet[n + 2] / 255.0f) * 2.0f - 1.0f);
    c.rsX = (packet[n + 3] / 255.0f) * 2.0f - 1.0f;
    c.rsY = -((packet[n + 4] / 255.0f) * 2.0f - 1.0f);
    c.lt = packet[n + 5] / 255.0f;
    c.rt = packet[n + 6] / 255.0f;
    switch (packet[n + 8] & 0x0f) {
        case 0x0: c.up = true; break;
        case 0x1: c.up = c.right = true; break;
        case 0x2: c.right = true; break;
        case 0x3: c.down = c.right = true; break;
        case 0x4: c.down = true; break;
        case 0x5: c.down = c.left = true; break;
        case 0x6: c.left = true; break;
        case 0x7: c.up = c.left = true; break;
    }
    c.x = packet[n + 8] & 0x10;
    c.a = packet[n + 8] & 0x20;
    c.b = packet[n + 8] & 0x40;
    c.y = packet[n + 8] & 0x80;
    c.lb = packet[n + 9] & 0x01;
    c.rb = packet[n + 9] & 0x02;
    c.back = packet[n + 9] & 0x10;
    c.start = packet[n + 9] & 0x20;
    c.ls = packet[n + 9] & 0x40;
    c.rs = packet[n + 9] & 0x80;
    return c;
}

Controller controllerGetStateXInput() {
    Controller c = {};
    XINPUT_STATE state;
    if (XInputGetState(0, &state) == ERROR_SUCCESS) {
        c.back = state.Gamepad.wButtons & XINPUT_GAMEPAD_BACK;
        c.start = state.Gamepad.wButtons & XINPUT_GAMEPAD_START;
        c.a = state.Gamepad.wButtons & XINPUT_GAMEPAD_A;
        c.b = state.Gamepad.wButtons & XINPUT_GAMEPAD_B;
        c.x = state.Gamepad.wButtons & XINPUT_GAMEPAD_X;
        c.y = state.Gamepad.wButtons & XINPUT_GAMEPAD_Y;
        c.up = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP;
        c.down = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN;
        c.left = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT;
        c.right = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT;
        c.lb = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER;
        c.rb = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER;
        c.lt = state.Gamepad.bLeftTrigger / 255.0f;
        c.rt = state.Gamepad.bRightTrigger / 255.0f;
        c.ls = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB;
        c.rs = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB;
        c.lsX = state.Gamepad.sThumbLX / 32767.0f;
        c.lsY = state.Gamepad.sThumbLY / 32767.0f;
        c.rsX = state.Gamepad.sThumbRX / 32767.0f;
        c.rsY = state.Gamepad.sThumbRY / 32767.0f;
    }
    return c;
}

Controller controllerGetStateGameInput() {
    Controller c = {};
    IGameInputReading* reading;
    if (SUCCEEDED(gameInput->GetCurrentReading(GameInputKindGamepad, nullptr, &reading))) {
        // IGameInputDevice *device;
        // reading->GetDevice(&device);
        // const GameInputDeviceInfo* info = device->GetDeviceInfo();
        GameInputGamepadState state;
        reading->GetGamepadState(&state);
        reading->Release();
        c.back = state.buttons & GameInputGamepadMenu;
        c.start = state.buttons & GameInputGamepadView;
        c.a = state.buttons & GameInputGamepadA;
        c.b = state.buttons & GameInputGamepadB;
        c.x = state.buttons & GameInputGamepadX;
        c.y = state.buttons & GameInputGamepadY;
        c.up = state.buttons & GameInputGamepadDPadUp;
        c.down = state.buttons & GameInputGamepadDPadDown;
        c.left = state.buttons & GameInputGamepadDPadLeft;
        c.right = state.buttons & GameInputGamepadDPadRight;
        c.lb = state.buttons & GameInputGamepadLeftShoulder;
        c.rb = state.buttons & GameInputGamepadRightShoulder;
        c.lt = state.leftTrigger / 255.0f;
        c.rt = state.rightTrigger / 255.0f;
        c.ls = state.buttons & GameInputGamepadLeftThumbstick;
        c.rs = state.buttons & GameInputGamepadRightThumbstick;
        c.lsX = state.leftThumbstickX / 32767.0f;
        c.lsY = state.leftThumbstickY / 32767.0f;
        c.rsX = state.rightThumbstickX / 32767.0f;
        c.rsY = state.rightThumbstickY / 32767.0f;
    }
    return c;
}

void controllerGetState() {
    Controller c = controllerGetStateXInput();
    // Controller c = controllerGetStateGameInput();
    //  Controller c = controllerGetStateDualSense();
    controllerApplyDeadZone(&c);
    if (controller.back && c.back) c.backDownDuration = controller.backDownDuration + frameTime;
    if (controller.start && c.start) c.startDownDuration = controller.startDownDuration + frameTime;
    if (controller.a && c.a) c.aDownDuration = controller.aDownDuration + frameTime;
    if (controller.b && c.b) c.bDownDuration = controller.bDownDuration + frameTime;
    if (controller.x && c.x) c.xDownDuration = controller.xDownDuration + frameTime;
    if (controller.y && c.y) c.yDownDuration = controller.yDownDuration + frameTime;
    if (controller.up && c.up) c.upDownDuration = controller.upDownDuration + frameTime;
    if (controller.down && c.down) c.downDownDuration = controller.downDownDuration + frameTime;
    if (controller.left && c.left) c.leftDownDuration = controller.leftDownDuration + frameTime;
    if (controller.right && c.right) c.rightDownDuration = controller.rightDownDuration + frameTime;
    if (controller.lb && c.lb) c.lbDownDuration = controller.lbDownDuration + frameTime;
    if (controller.rb && c.rb) c.rbDownDuration = controller.rbDownDuration + frameTime;
    if (controller.ls && c.ls) c.lsDownDuration = controller.lsDownDuration + frameTime;
    if (controller.rs && c.rs) c.rsDownDuration = controller.rsDownDuration + frameTime;
    controller = c;
}

void d3dMessageCallback(D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context) {
    if (id == D3D12_MESSAGE_ID_REFLECTSHAREDPROPERTIES_INVALIDOBJECT) return;
    if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR) {
        OutputDebugStringA(description);
        //__debugbreak();
    }
}

D3D12MA::Allocation* d3dCreateImage(const D3D12MA::ALLOCATION_DESC& allocDesc, D3D12_CLEAR_VALUE* clearValue, const D3D12_RESOURCE_DESC& resourceDesc, D3D12_SUBRESOURCE_DATA* imageMipsData, ID3D12GraphicsCommandList* cmdList, const wchar_t* name, D3D12_RESOURCE_STATES stateAfter) {
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, clearValue, &image, {}, nullptr)));
    image->GetResource()->SetName(name);
    if (imageMipsData) {
        d3d.stagingBuffer.size = align(d3d.stagingBuffer.size, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
        uint rowCounts[16];
        uint64 rowSizes[16];
        uint64 requiredSize;
        d3d.device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
        assert(d3d.stagingBuffer.size + requiredSize < d3d.stagingBuffer.capacity);
        for (uint mipIndex = 0; mipIndex < resourceDesc.MipLevels; mipIndex++) {
            mipFootprints[mipIndex].Offset += d3d.stagingBuffer.size;
        }
        assert(UpdateSubresources(cmdList, image->GetResource(), d3d.stagingBuffer.buffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, imageMipsData) == requiredSize);
        d3d.stagingBuffer.size += requiredSize;
    }
    if (stateAfter != D3D12_RESOURCE_STATE_COPY_DEST) {
        D3D12_RESOURCE_BARRIER transition = {
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Transition = {.pResource = image->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = stateAfter},
        };
        cmdList->ResourceBarrier(1, &transition);
    }
    return image;
}

D3D12MA::Allocation* d3dCreateImageSTB(const std::filesystem::path& ddsFilePath, ID3D12GraphicsCommandList* cmdList, const wchar_t* name, D3D12_RESOURCE_STATES stateAfter) {
    int width, height, channelCount;
    unsigned char* imageData = stbi_load(ddsFilePath.string().c_str(), &width, &height, &channelCount, 4);
    assert(imageData);
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint)width, .Height = (uint)height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr)));
    image->GetResource()->SetName(name);
    d3d.stagingBuffer.size = align(d3d.stagingBuffer.size, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprint;
    uint rowCount;
    uint64 rowSize;
    uint64 requiredSize;
    d3d.device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, &mipFootprint, &rowCount, &rowSize, &requiredSize);
    assert(d3d.stagingBuffer.size + requiredSize < d3d.stagingBuffer.capacity);
    mipFootprint.Offset += d3d.stagingBuffer.size;
    D3D12_SUBRESOURCE_DATA srcData = {.pData = imageData, .RowPitch = width * 4, .SlicePitch = width * height * 4};
    assert(UpdateSubresources(cmdList, image->GetResource(), d3d.stagingBuffer.buffer->GetResource(), 0, 1, requiredSize, &mipFootprint, &rowCount, &rowSize, &srcData) == requiredSize);
    d3d.stagingBuffer.size += requiredSize;
    stbi_image_free(imageData);
    D3D12_RESOURCE_BARRIER transition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = image->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = stateAfter}};
    cmdList->ResourceBarrier(1, &transition);
    return image;
}

D3D12MA::Allocation* d3dCreateImageDDS(const std::filesystem::path& ddsFilePath, ID3D12GraphicsCommandList* cmdList, const wchar_t* name, D3D12_RESOURCE_STATES stateAfter) {
    DirectX::ScratchImage scratchImage;
    assert(SUCCEEDED(LoadFromDDSFile(ddsFilePath.c_str(), DirectX::DDS_FLAGS_NONE, nullptr, scratchImage)));
    assert(scratchImage.GetImageCount() == scratchImage.GetMetadata().mipLevels);
    const DirectX::TexMetadata& scratchImageInfo = scratchImage.GetMetadata();
    DXGI_FORMAT format = scratchImageInfo.format;
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC resourceDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        .Width = (uint)scratchImageInfo.width,
        .Height = (uint)scratchImageInfo.height,
        .DepthOrArraySize = (uint16)scratchImageInfo.arraySize,
        .MipLevels = (uint16)scratchImageInfo.mipLevels,
        .Format = format,
        .SampleDesc = {.Count = 1},
    };
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr)));
    image->GetResource()->SetName(name);
    d3d.stagingBuffer.size = align(d3d.stagingBuffer.size, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
    uint rowCounts[16];
    uint64 rowSizes[16];
    uint64 requiredSize;
    D3D12_SUBRESOURCE_DATA srcData[16];
    d3d.device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
    assert(d3d.stagingBuffer.size + requiredSize < d3d.stagingBuffer.capacity);
    for (uint mipIndex = 0; mipIndex < scratchImageInfo.mipLevels; mipIndex++) {
        mipFootprints[mipIndex].Offset += d3d.stagingBuffer.size;
        const DirectX::Image& image = scratchImage.GetImages()[mipIndex];
        srcData[mipIndex] = {.pData = image.pixels, .RowPitch = (int64)image.rowPitch, .SlicePitch = (int64)image.slicePitch};
    }
    assert(UpdateSubresources(cmdList, image->GetResource(), d3d.stagingBuffer.buffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, srcData) == requiredSize);
    d3d.stagingBuffer.size += requiredSize;
    D3D12_RESOURCE_BARRIER transition = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = {.pResource = image->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = stateAfter},
    };
    cmdList->ResourceBarrier(1, &transition);
    return image;
}

void d3dCmdListReset() {
    assert(SUCCEEDED(d3d.graphicsCmdAllocator->Reset()));
    assert(SUCCEEDED(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocator, nullptr)));
}

void d3dCmdListExecute() {
    assert(SUCCEEDED(d3d.graphicsCmdList->Close()));
    d3d.graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.graphicsCmdList);
}

void d3dSignalFence(D3DFence* fence) {
    fence->value += 1;
    d3d.graphicsQueue->Signal(fence->fence, fence->value);
}

void d3dWaitFence(D3DFence* fence) {
    if (fence->fence->GetCompletedValue() < fence->value) {
        assert(SUCCEEDED(fence->fence->SetEventOnCompletion(fence->value, fence->event)));
        assert(WaitForSingleObjectEx(fence->event, INFINITE, false) == WAIT_OBJECT_0);
    }
}

bool d3dTryWaitFence(D3DFence* fence) {
    bool wait = fence->fence->GetCompletedValue() < fence->value;
    return !wait;
}

void d3dInit() {
    bool debug = commandLineContain(L"d3d_debug");
    uint factoryFlags = 0;
    if (debug) {
        factoryFlags = factoryFlags | DXGI_CREATE_FACTORY_DEBUG;
        ID3D12Debug1* debug;
        assert(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))));
        debug->EnableDebugLayer();
        // debug->SetEnableGPUBasedValidation(true);
        // debug->SetEnableSynchronizedCommandQueueValidation(true);
    }

    assert(SUCCEEDED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&d3d.dxgiFactory))));
    assert(SUCCEEDED(d3d.dxgiFactory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&d3d.dxgiAdapter))));
    DXGI_ADAPTER_DESC dxgiAdapterDesc = {};
    assert(SUCCEEDED(d3d.dxgiAdapter->GetDesc(&dxgiAdapterDesc)));
    assert(SUCCEEDED(D3D12CreateDevice(d3d.dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d.device))));
    if (debug) {
        ID3D12InfoQueue1* infoQueue;
        DWORD callbackCookie;
        assert(SUCCEEDED(d3d.device->QueryInterface(IID_PPV_ARGS(&infoQueue))));
        assert(SUCCEEDED(infoQueue->RegisterMessageCallback(d3dMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &callbackCookie)));
    }
    IDXGIOutput6* dxgiOutput;
    DXGI_OUTPUT_DESC1 dxgiOutputDesc;
    assert(SUCCEEDED(d3d.dxgiAdapter->EnumOutputs(0, (IDXGIOutput**)&dxgiOutput)));
    assert(SUCCEEDED(dxgiOutput->GetDesc1(&dxgiOutputDesc)));
    dxgiOutput->Release();
    hdr = (dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS resourceBindingTier = {};
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 rayTracing = {};
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &resourceBindingTier, sizeof(resourceBindingTier))));
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))));
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &rayTracing, sizeof(rayTracing))));
        assert(resourceBindingTier.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        assert(shaderModel.HighestShaderModel == D3D_SHADER_MODEL_6_6);
        assert(rayTracing.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
    }
    {
        D3D12_COMMAND_QUEUE_DESC graphicsQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
        assert(SUCCEEDED(d3d.device->CreateCommandQueue(&graphicsQueueDesc, IID_PPV_ARGS(&d3d.graphicsQueue))));
        assert(SUCCEEDED(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.graphicsCmdAllocator))));
        assert(SUCCEEDED(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.graphicsCmdAllocatorPrevFrame))));
        assert(SUCCEEDED(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.graphicsCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.graphicsCmdList))));
        assert(SUCCEEDED(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.graphicsCmdAllocatorPrevFrame, nullptr, IID_PPV_ARGS(&d3d.graphicsCmdListPrevFrame))));
        assert(SUCCEEDED(d3d.graphicsCmdList->Close()));
        assert(SUCCEEDED(d3d.graphicsCmdListPrevFrame->Close()));
        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.transferFence.fence))));
        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.renderFence.fence))));
        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.renderFencePrevFrame.fence))));
        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.collisionQueriesFence.fence))));
        assert(d3d.transferFence.event = CreateEventA(nullptr, false, false, nullptr));
        assert(d3d.renderFence.event = CreateEventA(nullptr, false, false, nullptr));
        assert(d3d.renderFencePrevFrame.event = CreateEventA(nullptr, false, false, nullptr));
        assert(d3d.collisionQueriesFence.event = CreateEventA(nullptr, false, false, nullptr));
    }
    {
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
            .Width = (uint)renderW,
            .Height = (uint)renderH,
            .Format = d3d.swapChainFormat,
            .SampleDesc = {.Count = 1},
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER,
            .BufferCount = countof(d3d.swapChainImages),
            .Scaling = DXGI_SCALING_NONE,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
            .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED,
        };
        assert(SUCCEEDED(d3d.dxgiFactory->CreateSwapChainForHwnd(d3d.graphicsQueue, window.hwnd, &swapChainDesc, nullptr, nullptr, (IDXGISwapChain1**)&d3d.swapChain)));

        DXGI_COLOR_SPACE_TYPE colorSpace = hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(colorSpace)));
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
            assert(SUCCEEDED(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image))));
            (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
        }
        d3d.dxgiFactory->MakeWindowAssociation(window.hwnd, DXGI_MWA_NO_WINDOW_CHANGES); // disable alt-enter
    }
    {
        d3d.rtvDescriptorHeap.size = 0;
        d3d.rtvDescriptorHeap.capacity = 16;
        d3d.rtvDescriptorHeap.descriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = d3d.rtvDescriptorHeap.capacity, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        assert(SUCCEEDED(d3d.device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&d3d.rtvDescriptorHeap.heap))));
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource* image = d3d.swapChainImages[imageIndex];
            d3d.swapChainImageRTVDescriptors[imageIndex] = {d3d.rtvDescriptorHeap.heap->GetCPUDescriptorHandleForHeapStart().ptr + d3d.rtvDescriptorHeap.descriptorSize * d3d.rtvDescriptorHeap.size};
            d3d.device->CreateRenderTargetView(image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
            d3d.rtvDescriptorHeap.size += 1;
        }

        d3d.cbvSrvUavDescriptorHeap.size = 0;
        d3d.cbvSrvUavDescriptorHeapPrevFrame.size = 0;
        d3d.cbvSrvUavDescriptorHeap.capacity = 1'000'000;
        d3d.cbvSrvUavDescriptorHeapPrevFrame.capacity = 1'000'000;
        d3d.cbvSrvUavDescriptorHeap.descriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        d3d.cbvSrvUavDescriptorHeapPrevFrame.descriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = d3d.cbvSrvUavDescriptorHeap.capacity, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        assert(SUCCEEDED(d3d.device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&d3d.cbvSrvUavDescriptorHeap.heap))));
        assert(SUCCEEDED(d3d.device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&d3d.cbvSrvUavDescriptorHeapPrevFrame.heap))));
    }
    {
        D3D12MA::ALLOCATOR_DESC allocatorDesc = {.Flags = D3D12MA::ALLOCATOR_FLAG_NONE, .pDevice = d3d.device, .pAdapter = d3d.dxgiAdapter};
        assert(SUCCEEDED(D3D12MA::CreateAllocator(&allocatorDesc, &d3d.allocator)));
    }
    {
        auto d3dUploadBufferInit = [](D3DUploadBuffer* buffer, uint64 size, D3D12_RESOURCE_STATES initStates, const wchar_t* name) {
            buffer->capacity = size;
            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, initStates, nullptr, &buffer->buffer, {}, nullptr)));
            buffer->buffer->GetResource()->SetName(name);
            assert(SUCCEEDED(buffer->buffer->GetResource()->Map(0, nullptr, (void**)&buffer->ptr)));
        };
        auto d3dReadBackBufferInit = [](D3DReadBackBuffer* buffer, uint64 size, D3D12_RESOURCE_STATES initStates, const wchar_t* name) {
            buffer->capacity = size;
            D3D12MA::ALLOCATION_DESC allocationDesc = {};
            D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            allocationDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, initStates, nullptr, &buffer->bufferUAV, {}, nullptr)));
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
            allocationDesc.HeapType = D3D12_HEAP_TYPE_READBACK;
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, initStates, nullptr, &buffer->buffer, {}, nullptr)));
            buffer->bufferUAV->GetResource()->SetName((std::wstring(name) + L"UAV").c_str());
            buffer->buffer->GetResource()->SetName(name);
            assert(SUCCEEDED(buffer->buffer->GetResource()->Map(0, nullptr, (void**)&buffer->ptr)));
        };
        d3dUploadBufferInit(&d3d.stagingBuffer, gigabytes(2), D3D12_RESOURCE_STATE_COPY_SOURCE, L"stagingBuffer");
        d3dUploadBufferInit(&d3d.constantsBuffer, megabytes(2), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer");
        d3dUploadBufferInit(&d3d.constantsBufferPrevFrame, megabytes(2), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBufferPrevFrame");
        d3dUploadBufferInit(&d3d.imguiVertexBuffer, megabytes(10), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer");
        d3dUploadBufferInit(&d3d.imguiVertexBufferPrevFrame, megabytes(10), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBufferPrevFrame");
        d3dUploadBufferInit(&d3d.imguiIndexBuffer, megabytes(10), D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer");
        d3dUploadBufferInit(&d3d.imguiIndexBufferPrevFrame, megabytes(10), D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBufferPrevFrame");
        d3dUploadBufferInit(&d3d.blasInstanceDescsBuffer, megabytes(32), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfosBuffer");
        d3dUploadBufferInit(&d3d.blasInstanceDescsBufferPrevFrame, megabytes(32), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfosBufferPrevFrame");
        d3dUploadBufferInit(&d3d.blasInstancesInfosBuffer, megabytes(16), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesInfosBuffer");
        d3dUploadBufferInit(&d3d.blasInstancesInfosBufferPrevFrame, megabytes(16), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesInfosBufferPrevFrame");
        d3dUploadBufferInit(&d3d.blasGeometriesInfosBuffer, megabytes(16), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"blasGeometriesInfosBuffer");
        d3dUploadBufferInit(&d3d.blasGeometriesInfosBufferPrevFrame, megabytes(16), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"blasGeometriesInfosBufferPrevFrame");
        d3dUploadBufferInit(&d3d.collisionQueriesBuffer, megabytes(1), D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueriesBuffer");
        d3dReadBackBufferInit(&d3d.collisionQueryResultsBuffer, megabytes(1), D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsBuffer");
        {
            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = megabytes(32), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &d3d.tlasBuffer, {}, nullptr)));
            d3d.tlasBuffer->GetResource()->SetName(L"tlasBuffer");
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &d3d.tlasScratchBuffer, {}, nullptr)));
            d3d.tlasScratchBuffer->GetResource()->SetName(L"tlasScratchBuffer");
        }
    }
    {
        D3D12_RESOURCE_DESC textureDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            .Width = (uint)renderW,
            .Height = (uint)renderH,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .SampleDesc = {.Count = 1},
            .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        };
        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};

        textureDesc.Format = d3d.renderTextureFormat;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &textureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr)));
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &textureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexturePrevFrame, {}, nullptr)));
        d3d.renderTexture->GetResource()->SetName(L"renderTexture");
        d3d.renderTexturePrevFrame->GetResource()->SetName(L"renderTexturePrevFrame");

        textureDesc.Format = d3d.depthTextureFormat;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &textureDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &d3d.depthTexture, {}, nullptr)));
        d3d.depthTexture->GetResource()->SetName(L"depthTexture");

        textureDesc.Format = d3d.motionVectorTextureFormat;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &textureDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &d3d.motionVectorTexture, {}, nullptr)));
        d3d.motionVectorTexture->GetResource()->SetName(L"motionVectorTexture");

        textureDesc.Format = d3d.pathTracerAccumulationTextureFormat;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &textureDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &d3d.pathTracerAccumulationTexture, {}, nullptr)));
        d3d.pathTracerAccumulationTexture->GetResource()->SetName(L"pathTracerAccumulationTexture");
    }
    {
        d3dCmdListReset();
        d3d.stagingBuffer.size = 0;
        {
            uint8* imguiTextureData;
            int imguiTextureWidth, imguiTextureHeight;
            ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&imguiTextureData, &imguiTextureWidth, &imguiTextureHeight);
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint64)imguiTextureWidth, .Height = (uint)imguiTextureHeight, .DepthOrArraySize = 1, .MipLevels = 1, .Format = d3d.imguiTextureFormat, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = imguiTextureData, .RowPitch = imguiTextureWidth * 4, .SlicePitch = imguiTextureWidth * imguiTextureHeight * 4};
            d3d.imguiTexture = d3dCreateImage(D3D12MA::ALLOCATION_DESC{.HeapType = D3D12_HEAP_TYPE_DEFAULT}, nullptr, desc, &data, d3d.graphicsCmdList, L"imguiTexture", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        {
            uint8_4 defaultEmissiveTextureData[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = defaultEmissiveTextureData, .RowPitch = 8, .SlicePitch = 16};
            d3d.defaultEmissiveTexture = d3dCreateImage(D3D12MA::ALLOCATION_DESC{.HeapType = D3D12_HEAP_TYPE_DEFAULT}, nullptr, desc, &data, d3d.graphicsCmdList, L"defaultEmissiveTexture", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            d3d.defaultEmissiveTextureSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
        }
        {
            uint8_4 defaultBaseColorTextureData[4] = {{255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}};
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = defaultBaseColorTextureData, .RowPitch = 8, .SlicePitch = 16};
            d3d.defaultBaseColorTexture = d3dCreateImage(D3D12MA::ALLOCATION_DESC{.HeapType = D3D12_HEAP_TYPE_DEFAULT}, nullptr, desc, &data, d3d.graphicsCmdList, L"defaultBaseColorTexture", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            d3d.defaultBaseColorTextureSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
        }
        {
            uint8_2 defaultMetallicRoughnessData[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = defaultMetallicRoughnessData, .RowPitch = 4, .SlicePitch = 8};
            d3d.defaultMetallicRoughnessTexture = d3dCreateImage(D3D12MA::ALLOCATION_DESC{.HeapType = D3D12_HEAP_TYPE_DEFAULT}, nullptr, desc, &data, d3d.graphicsCmdList, L"defaultMetallicRoughnessTexture", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            d3d.defaultMetallicRoughnessTextureSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
        }
        {
            uint8_4 defaultNormalTextureData[4] = {{128, 128, 255, 0}, {128, 128, 255, 0}, {128, 128, 255, 0}, {128, 128, 255, 0}};
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = defaultNormalTextureData, .RowPitch = 8, .SlicePitch = 16};
            d3d.defaultNormalTexture = d3dCreateImage(D3D12MA::ALLOCATION_DESC{.HeapType = D3D12_HEAP_TYPE_DEFAULT}, nullptr, desc, &data, d3d.graphicsCmdList, L"defaultNormalTexture", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            d3d.defaultNormalTextureSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
        }
        d3dCmdListExecute();
        d3dSignalFence(&d3d.transferFence);
        d3dWaitFence(&d3d.transferFence);
    }
}

void d3dCompileShaders() {
    {
        static std::filesystem::path shaderPath = exeDir / "renderScene.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.renderScenePSO) d3d.renderScenePSO->Release();
            if (d3d.renderScenePSOProps) d3d.renderScenePSOProps->Release();
            if (d3d.renderSceneRootSig) d3d.renderSceneRootSig->Release();
            std::vector<uint8> rtByteCode = fileReadBytes(shaderPath);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"rayMissPrimary"}, {L"rayHitGroupPrimary"}, {L"rayAnyHitPrimary"}, {L"rayClosestHitPrimary"}, {L"rayMissShadow"}, {L"rayHitGroupShadow"}, {L"rayClosestHitShadow"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(SUCCEEDED(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.renderScenePSO))));
            assert(SUCCEEDED(d3d.renderScenePSO->QueryInterface(IID_PPV_ARGS(&d3d.renderScenePSOProps))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.renderSceneRootSig))));
            assert(d3d.renderSceneRayGenID = d3d.renderScenePSOProps->GetShaderIdentifier(L"rayGen"));
            assert(d3d.renderSceneRayMissIDPrimary = d3d.renderScenePSOProps->GetShaderIdentifier(L"rayMissPrimary"));
            assert(d3d.renderSceneRayHitGroupIDPrimary = d3d.renderScenePSOProps->GetShaderIdentifier(L"rayHitGroupPrimary"));
            assert(d3d.renderSceneRayMissIDShadow = d3d.renderScenePSOProps->GetShaderIdentifier(L"rayMissShadow"));
            assert(d3d.renderSceneRayHitGroupIDShadow = d3d.renderScenePSOProps->GetShaderIdentifier(L"rayHitGroupShadow"));
        }
    }
    {
        static std::filesystem::path shaderPath = exeDir / "pathTracer.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.pathTracerPSO) d3d.pathTracerPSO->Release();
            if (d3d.pathTracerPSOProps) d3d.pathTracerPSOProps->Release();
            if (d3d.pathTracerRootSig) d3d.pathTracerRootSig->Release();
            std::vector<uint8> rtByteCode = fileReadBytes(shaderPath);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"rayMiss"}, {L"rayHitGroup"}, {L"rayAnyHit"}, {L"rayClosestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(SUCCEEDED(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.pathTracerPSO))));
            assert(SUCCEEDED(d3d.pathTracerPSO->QueryInterface(IID_PPV_ARGS(&d3d.pathTracerPSOProps))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.pathTracerRootSig))));
            assert(d3d.pathTracerRayGenID = d3d.pathTracerPSOProps->GetShaderIdentifier(L"rayGen"));
            assert(d3d.pathTracerRayMissID = d3d.pathTracerPSOProps->GetShaderIdentifier(L"rayMiss"));
            assert(d3d.pathTracerRayHitGroupID = d3d.pathTracerPSOProps->GetShaderIdentifier(L"rayHitGroup"));
        }
    }
    {
        static std::filesystem::path shaderPath = exeDir / "collisionQuery.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.collisionQueryPSO) d3d.collisionQueryPSO->Release();
            if (d3d.collisionQueryProps) d3d.collisionQueryProps->Release();
            if (d3d.collisionQueryRootSig) d3d.collisionQueryRootSig->Release();
            std::vector<uint8> rtByteCode = fileReadBytes(shaderPath);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"miss"}, {L"hitGroup"}, {L"closestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(SUCCEEDED(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.collisionQueryPSO))));
            assert(SUCCEEDED(d3d.collisionQueryPSO->QueryInterface(IID_PPV_ARGS(&d3d.collisionQueryProps))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.collisionQueryRootSig))));
            assert(d3d.collisionQueryRayGenID = d3d.collisionQueryProps->GetShaderIdentifier(L"rayGen"));
            assert(d3d.collisionQueryMissID = d3d.collisionQueryProps->GetShaderIdentifier(L"miss"));
            assert(d3d.collisionQueryHitGroupID = d3d.collisionQueryProps->GetShaderIdentifier(L"hitGroup"));
        }
    }
    {
        static std::filesystem::path shaderPath = exeDir / "vertexSkinningCS.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.vertexSkinningPSO) d3d.vertexSkinningPSO->Release();
            if (d3d.vertexSkinningRootSig) d3d.vertexSkinningRootSig->Release();
            std::vector<uint8> csByteCode = fileReadBytes(shaderPath);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {.pRootSignature = d3d.vertexSkinningRootSig, .CS = {.pShaderBytecode = csByteCode.data(), .BytecodeLength = csByteCode.size()}};
            assert(SUCCEEDED(d3d.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&d3d.vertexSkinningPSO))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, csByteCode.data(), csByteCode.size(), IID_PPV_ARGS(&d3d.vertexSkinningRootSig))));
        }
    }
    {
        static std::filesystem::path shaderPathVS = exeDir / "compositeVS.cso";
        static std::filesystem::path shaderPathPS = exeDir / "compositePS.cso";
        static std::filesystem::file_time_type prevLastWriteTimeVS = {};
        static std::filesystem::file_time_type prevLastWriteTimePS = {};
        std::filesystem::file_time_type lastWriteTimeVS = std::filesystem::last_write_time(shaderPathVS);
        std::filesystem::file_time_type lastWriteTimePS = std::filesystem::last_write_time(shaderPathPS);
        if (lastWriteTimeVS > prevLastWriteTimeVS || lastWriteTimePS > prevLastWriteTimePS) {
            prevLastWriteTimeVS = lastWriteTimeVS;
            prevLastWriteTimePS = lastWriteTimePS;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.compositePSO) d3d.compositePSO->Release();
            if (d3d.compositeRootSig) d3d.compositeRootSig->Release();
            std::vector<uint8> vsByteCode = fileReadBytes(shaderPathVS);
            std::vector<uint8> psByteCode = fileReadBytes(shaderPathPS);
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
                .VS = {vsByteCode.data(), vsByteCode.size()},
                .PS = {psByteCode.data(), psByteCode.size()},
                .BlendState = {.RenderTarget = {{.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL}}},
                .SampleMask = 0xffffffff,
                .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_BACK},
                .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                .NumRenderTargets = 1,
                .RTVFormats = {d3d.swapChainFormat},
                .SampleDesc = {.Count = 1},
            };
            assert(SUCCEEDED(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.compositePSO))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, psByteCode.data(), psByteCode.size(), IID_PPV_ARGS(&d3d.compositeRootSig))));
        }
    }
    {
        static std::filesystem::path shaderPathVS = exeDir / "ImGuiVS.cso";
        static std::filesystem::path shaderPathPS = exeDir / "ImGuiPS.cso";
        static std::filesystem::file_time_type prevLastWriteTimeVS = {};
        static std::filesystem::file_time_type prevLastWriteTimePS = {};
        std::filesystem::file_time_type lastWriteTimeVS = std::filesystem::last_write_time(shaderPathVS);
        std::filesystem::file_time_type lastWriteTimePS = std::filesystem::last_write_time(shaderPathPS);
        if (lastWriteTimeVS > prevLastWriteTimeVS || lastWriteTimePS > prevLastWriteTimePS) {
            prevLastWriteTimeVS = lastWriteTimeVS;
            prevLastWriteTimePS = lastWriteTimePS;
            d3dWaitFence(&d3d.renderFence);
            d3dWaitFence(&d3d.renderFencePrevFrame);
            if (d3d.imguiPSO) d3d.imguiPSO->Release();
            if (d3d.imguiRootSig) d3d.imguiRootSig->Release();
            std::vector<uint8> vsByteCode = fileReadBytes(shaderPathVS);
            std::vector<uint8> psByteCode = fileReadBytes(shaderPathPS);
            D3D12_INPUT_ELEMENT_DESC inputElemDescs[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            };
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
                .VS = {vsByteCode.data(), vsByteCode.size()},
                .PS = {psByteCode.data(), psByteCode.size()},
                .BlendState = {
                    .RenderTarget = {{
                        .BlendEnable = true,
                        .SrcBlend = D3D12_BLEND_SRC_ALPHA,
                        .DestBlend = D3D12_BLEND_INV_SRC_ALPHA,
                        .BlendOp = D3D12_BLEND_OP_ADD,
                        .SrcBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA,
                        .DestBlendAlpha = D3D12_BLEND_ZERO,
                        .BlendOpAlpha = D3D12_BLEND_OP_ADD,
                        .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL,
                    }}},
                .SampleMask = 0xffffffff,
                .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = true},
                .InputLayout = {inputElemDescs, countof(inputElemDescs)},
                .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                .NumRenderTargets = 1,
                .RTVFormats = {d3d.swapChainFormat},
                .SampleDesc = {.Count = 1},
            };
            assert(SUCCEEDED(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.imguiPSO))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, vsByteCode.data(), vsByteCode.size(), IID_PPV_ARGS(&d3d.imguiRootSig))));
        }
    }
}

void d3dApplySettings() {
    IDXGIOutput6* dxgiOutput;
    DXGI_OUTPUT_DESC1 dxgiOutputDesc;
    assert(SUCCEEDED(d3d.swapChain->GetContainingOutput((IDXGIOutput**)&dxgiOutput)));
    assert(SUCCEEDED(dxgiOutput->GetDesc1(&dxgiOutputDesc)));
    dxgiOutput->Release();
    if (hdr && dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)));
    }
    else {
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)));
    }
    static RECT windowRect = [] {  RECT r; assert(GetWindowRect(window.hwnd, &r)); return r; }();
    if (windowMode == WindowModeWindowed) {
        assert(SetWindowLong(window.hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW));
        assert(SetWindowPos(window.hwnd, HWND_NOTOPMOST, windowRect.left, windowRect.top, windowRect.right - windowRect.left, windowRect.bottom - windowRect.top, SWP_FRAMECHANGED | SWP_NOACTIVATE));
        ShowWindow(window.hwnd, SW_NORMAL);
    }
    else if (windowMode == WindowModeBorderless) {
        assert(GetWindowRect(window.hwnd, &windowRect));
        assert(SetWindowLong(window.hwnd, GWL_STYLE, GetWindowLong(window.hwnd, GWL_STYLE) & ~(WS_CAPTION | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU | WS_THICKFRAME)));
        RECT rect = dxgiOutputDesc.DesktopCoordinates;
        assert(SetWindowPos(window.hwnd, HWND_TOP, rect.left, rect.top, rect.right, rect.bottom, SWP_FRAMECHANGED | SWP_NOACTIVATE));
        ShowWindow(window.hwnd, SW_MAXIMIZE);
    }
}

D3DDescriptor d3dAppendCBVDescriptor(D3D12_CONSTANT_BUFFER_VIEW_DESC* constantBufferViewDesc) {
    assert(d3d.cbvSrvUavDescriptorHeap.size < d3d.cbvSrvUavDescriptorHeap.capacity);
    uint offset = d3d.cbvSrvUavDescriptorHeap.descriptorSize * d3d.cbvSrvUavDescriptorHeap.size;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateConstantBufferView(constantBufferViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorHeap.size += 1;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendSRVDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC* resourceViewDesc, ID3D12Resource* resource) {
    ZoneScopedN("d3dAppendSRVDescriptor");
    assert(d3d.cbvSrvUavDescriptorHeap.size < d3d.cbvSrvUavDescriptorHeap.capacity);
    uint offset = d3d.cbvSrvUavDescriptorHeap.descriptorSize * d3d.cbvSrvUavDescriptorHeap.size;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    {
        ZoneScopedN("CreateShaderResourceView");
        d3d.device->CreateShaderResourceView(resource, resourceViewDesc, cpuHandle);
    }
    d3d.cbvSrvUavDescriptorHeap.size += 1;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendUAVDescriptor(D3D12_UNORDERED_ACCESS_VIEW_DESC* unorderedAccessViewDesc, ID3D12Resource* resource) {
    assert(d3d.cbvSrvUavDescriptorHeap.size < d3d.cbvSrvUavDescriptorHeap.capacity);
    uint offset = d3d.cbvSrvUavDescriptorHeap.descriptorSize * d3d.cbvSrvUavDescriptorHeap.size;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap.heap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateUnorderedAccessView(resource, nullptr, unorderedAccessViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorHeap.size += 1;
    return {cpuHandle, gpuHandle};
}

void dlssInit() {
    int needsDriver = true;
    assert(NVSDK_NGX_D3D12_Init_with_ProjectID("1a632058-a49b-48d1-a742-9906f91912af", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", exeDir.c_str(), d3d.device) == NVSDK_NGX_Result_Success);
    assert(NVSDK_NGX_D3D12_GetCapabilityParameters(&ngxParameter) == NVSDK_NGX_Result_Success);
    assert(ngxParameter->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver) == NVSDK_NGX_Result_Success);
    assert(ngxParameter->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvaliable) == NVSDK_NGX_Result_Success);
    assert(!needsDriver);
    assert(dlssAvaliable);
}

bool projectCameraSpacePointToScreen(float3 p, float2* pScreen) {
    XMVector pViewSpace = XMVector4Transform(p.toXMVector(), cameraViewMat);
    DirectX::BoundingFrustum frustum(cameraProjectMat);
    DirectX::ContainmentType frustumContainsP = frustum.Contains(pViewSpace);
    if (frustumContainsP == DirectX::CONTAINS) {
        XMVector pClipSpace = XMVector4Transform(pViewSpace, cameraProjectMat);
        float4 pNDCSpace = pClipSpace;
        pNDCSpace /= pNDCSpace.w;
        *pScreen = ndcToScreen(pNDCSpace.xy(), float2((float)renderW, (float)renderH));
        return true;
    }
    else {
        return false;
    }
}

bool projectCameraSpaceLineToScreen(float3 p0, float3 p1, float2* p0Screen, float2* p1Screen) {
    XMVector p0ViewSpace = XMVector4Transform(p0.toXMVector(), cameraViewMat);
    XMVector p1ViewSpace = XMVector4Transform(p1.toXMVector(), cameraViewMat);
    XMVector p0ClipSpace = XMVector4Transform(p0ViewSpace, cameraProjectMat);
    XMVector p1ClipSpace = XMVector4Transform(p1ViewSpace, cameraProjectMat);
    float4 p0NDCSpace = p0ClipSpace;
    float4 p1NDCSpace = p1ClipSpace;
    p0NDCSpace /= p0NDCSpace.w;
    p1NDCSpace /= p1NDCSpace.w;
    DirectX::BoundingFrustum frustum(cameraProjectMat);
    DirectX::ContainmentType frustumContainsP0 = frustum.Contains(p0ViewSpace);
    DirectX::ContainmentType frustumContainsP1 = frustum.Contains(p1ViewSpace);
    float2 screenSize((float)renderW, (float)renderH);
    if (frustumContainsP0 && frustumContainsP1) {
        *p0Screen = ndcToScreen(p0NDCSpace.xy(), screenSize);
        *p1Screen = ndcToScreen(p1NDCSpace.xy(), screenSize);
        return true;
    }
    else if (!frustumContainsP0 && !frustumContainsP1) {
        float t;
        XMVector dir = XMVector3Normalize(XMVectorSubtract(p0ViewSpace, p1ViewSpace));
        if (frustum.Intersects(p1ViewSpace, dir, t)) {
            XMVector intersection0ViewSpace = XMVectorAdd(p1ViewSpace, XMVectorMultiply(dir, XMVectorSet(t, t, t, 0)));
            XMVector intersection0ClipSpace = XMVector4Transform(intersection0ViewSpace, cameraProjectMat);
            float4 intersection0NDCSpace = intersection0ClipSpace;
            intersection0NDCSpace /= intersection0NDCSpace.w;
            dir = XMVectorNegate(dir);
            if (frustum.Intersects(p0ViewSpace, dir, t)) {
                XMVector intersection1ViewSpace = XMVectorAdd(p0ViewSpace, XMVectorMultiply(dir, XMVectorSet(t, t, t, 0)));
                XMVector intersection1ClipSpace = XMVector4Transform(intersection1ViewSpace, cameraProjectMat);
                float4 intersection1NDCSpace = intersection1ClipSpace;
                intersection1NDCSpace /= intersection1NDCSpace.w;
                *p0Screen = ndcToScreen(intersection0NDCSpace.xy(), screenSize);
                *p1Screen = ndcToScreen(intersection1NDCSpace.xy(), screenSize);
                return true;
            }
        }
        return false;
    }
    else if (frustumContainsP0) {
        float t;
        XMVector dir = XMVector3Normalize(XMVectorSubtract(p0ViewSpace, p1ViewSpace));
        if (frustum.Intersects(p1ViewSpace, dir, t)) {
            XMVector intersectionViewSpace = XMVectorAdd(p1ViewSpace, XMVectorMultiply(dir, XMVectorSet(t, t, t, 0)));
            XMVector intersectionClipSpace = XMVector4Transform(intersectionViewSpace, cameraProjectMat);
            float4 intersectionNDCSpace = intersectionClipSpace;
            intersectionNDCSpace /= intersectionNDCSpace.w;
            *p0Screen = ndcToScreen(p0NDCSpace.xy(), screenSize);
            *p1Screen = ndcToScreen(intersectionNDCSpace.xy(), screenSize);
            return true;
        }
    }
    else if (frustumContainsP1) {
        float t;
        XMVector dir = XMVector3Normalize(XMVectorSubtract(p1ViewSpace, p0ViewSpace));
        if (frustum.Intersects(p0ViewSpace, dir, t)) {
            XMVector intersectionViewSpace = XMVectorAdd(p0ViewSpace, XMVectorMultiply(dir, XMVectorSet(t, t, t, 0)));
            XMVector intersectionClipSpace = XMVector4Transform(intersectionViewSpace, cameraProjectMat);
            float4 intersectionNDCSpace = intersectionClipSpace;
            intersectionNDCSpace /= intersectionNDCSpace.w;
            *p0Screen = ndcToScreen(intersectionNDCSpace.xy(), screenSize);
            *p1Screen = ndcToScreen(p1NDCSpace.xy(), screenSize);
            return true;
        }
    }
    return false;
}

bool projectCameraSpaceTriangleToScreen(float3 p0, float3 p1, float3 p2, float2* p0Screen, float2* p1Screen, float2* p2Screen) {
    DirectX::BoundingFrustum frustum(cameraProjectMat);
    XMVector p0ViewSpace = XMVector4Transform(p0.toXMVector(), cameraViewMat);
    XMVector p1ViewSpace = XMVector4Transform(p1.toXMVector(), cameraViewMat);
    XMVector p2ViewSpace = XMVector4Transform(p2.toXMVector(), cameraViewMat);
    if ((frustum.Contains(p0ViewSpace) == DirectX::CONTAINS) && (frustum.Contains(p1ViewSpace) == DirectX::CONTAINS) && (frustum.Contains(p2ViewSpace) == DirectX::CONTAINS)) {
        XMVector p0ClipSpace = XMVector4Transform(p0ViewSpace, cameraProjectMat);
        XMVector p1ClipSpace = XMVector4Transform(p1ViewSpace, cameraProjectMat);
        XMVector p2ClipSpace = XMVector4Transform(p2ViewSpace, cameraProjectMat);
        float4 p0NDCSpace = p0ClipSpace;
        float4 p1NDCSpace = p1ClipSpace;
        float4 p2NDCSpace = p2ClipSpace;
        p0NDCSpace /= p0NDCSpace.w;
        p1NDCSpace /= p1NDCSpace.w;
        p2NDCSpace /= p2NDCSpace.w;
        float2 screenSize = float2((float)renderW, (float)renderH);
        *p0Screen = ndcToScreen(p0NDCSpace.xy(), screenSize);
        *p1Screen = ndcToScreen(p1NDCSpace.xy(), screenSize);
        *p2Screen = ndcToScreen(p2NDCSpace.xy(), screenSize);
        return true;
    }
    return false;
}

Model* modelLoadGLTF(const std::filesystem::path& filePath) {
    const std::filesystem::path gltfFilePath = exeDir / filePath;
    const std::filesystem::path gltfFileFolderPath = gltfFilePath.parent_path();
    cgltf_options gltfOptions = {};
    cgltf_data* gltfData = nullptr;
    cgltf_result gltfParseFileResult = cgltf_parse_file(&gltfOptions, gltfFilePath.string().c_str(), &gltfData);
    assert(gltfParseFileResult == cgltf_result_success);
    cgltf_result gltfLoadBuffersResult = cgltf_load_buffers(&gltfOptions, gltfData, gltfFilePath.string().c_str());
    assert(gltfLoadBuffersResult == cgltf_result_success);
    assert(gltfData->scenes_count == 1);

    Model* model = &models.emplace_back();
    model->filePath = filePath;
    model->filePathStr = filePath.string();
    model->gltfData = gltfData;

    d3dCmdListReset();
    d3d.stagingBuffer.size = 0;

    model->nodes.resize(gltfData->nodes_count);
    model->meshes.resize(gltfData->meshes_count);
    model->skins.resize(gltfData->skins_count);
    model->animations.resize(gltfData->animations_count);
    model->materials.resize(gltfData->materials_count);
    model->textures.resize(gltfData->textures_count);
    model->images.resize(gltfData->images_count);

    for (uint nodeIndex = 0; nodeIndex < gltfData->nodes_count; nodeIndex++) {
        cgltf_node& gltfNode = gltfData->nodes[nodeIndex];
        ModelNode& node = model->nodes[nodeIndex];
        if (gltfNode.name) node.name = gltfNode.name;
        if (gltfNode.parent) {
            uint parentNodeIndex = (uint)(gltfNode.parent - gltfData->nodes);
            assert(parentNodeIndex >= 0 && parentNodeIndex < gltfData->nodes_count);
            node.parent = &model->nodes[parentNodeIndex];
        }
        else {
            node.parent = nullptr;
        }
        for (cgltf_node* child : std::span(gltfNode.children, gltfNode.children_count)) {
            uint childNodeIndex = (uint)(child - gltfData->nodes);
            assert(childNodeIndex >= 0 && childNodeIndex < gltfData->nodes_count);
            node.children.push_back(&model->nodes[childNodeIndex]);
        }
        float nodeTransform[16];
        cgltf_node_transform_world(&gltfNode, nodeTransform);
        node.globalTransform = XMMatrix(nodeTransform);
        if (gltfNode.has_matrix) {
            node.localTransform = XMMatrix(gltfNode.matrix);
        }
        else {
            Transform localTransform = {};
            if (gltfNode.has_scale) localTransform.s = float3(gltfNode.scale);
            if (gltfNode.has_rotation) localTransform.r = float4(gltfNode.rotation);
            if (gltfNode.has_translation) localTransform.t = float3(gltfNode.translation);
            node.localTransform = localTransform.toMat();
        }
        if (gltfNode.mesh) {
            uint meshIndex = (uint)(gltfNode.mesh - gltfData->meshes);
            assert(meshIndex >= 0 && meshIndex < gltfData->meshes_count);
            node.mesh = &model->meshes[meshIndex];
        }
        else {
            node.mesh = nullptr;
        }
        if (gltfNode.skin) {
            uint skinIndex = (uint)(gltfNode.skin - gltfData->skins);
            assert(skinIndex >= 0 && skinIndex < gltfData->skins_count);
            node.skin = &model->skins[skinIndex];
        }
        else {
            node.skin = nullptr;
        }
    }
    for (cgltf_node* gltfNode : std::span(gltfData->scenes[0].nodes, gltfData->scenes[0].nodes_count)) {
        model->rootNodes.push_back(&model->nodes[gltfNode - gltfData->nodes]);
    }
    for (ModelNode& node : model->nodes) {
        if (node.mesh) model->meshNodes.push_back(&node);
    }
    for (uint meshIndex = 0; meshIndex < gltfData->meshes_count; meshIndex++) {
        cgltf_mesh& gltfMesh = gltfData->meshes[meshIndex];
        ModelMesh& mesh = model->meshes[meshIndex];
        if (gltfMesh.name) mesh.name = gltfMesh.name;
        mesh.primitives.reserve(gltfMesh.primitives_count);
        for (cgltf_primitive& gltfPrimitive : std::span(gltfMesh.primitives, gltfMesh.primitives_count)) {
            cgltf_accessor* indices = gltfPrimitive.indices;
            cgltf_accessor* positions = nullptr;
            cgltf_accessor* normals = nullptr;
            cgltf_accessor* tangents = nullptr;
            cgltf_accessor* uvs = nullptr;
            cgltf_accessor* jointIndices = nullptr;
            cgltf_accessor* jointWeights = nullptr;
            for (cgltf_attribute& attribute : std::span(gltfPrimitive.attributes, gltfPrimitive.attributes_count)) {
                if (attribute.type == cgltf_attribute_type_position) {
                    positions = attribute.data;
                }
                else if (attribute.type == cgltf_attribute_type_normal) {
                    normals = attribute.data;
                }
                else if (attribute.type == cgltf_attribute_type_tangent) {
                    tangents = attribute.data;
                }
                else if (attribute.type == cgltf_attribute_type_texcoord) {
                    uvs = attribute.data;
                }
                else if (attribute.type == cgltf_attribute_type_joints) {
                    jointIndices = attribute.data;
                }
                else if (attribute.type == cgltf_attribute_type_weights) {
                    jointWeights = attribute.data;
                }
            }

            assert(gltfPrimitive.type == cgltf_primitive_type_triangles);
            assert(indices && positions && normals && uvs && tangents);
            assert(indices->count % 3 == 0 && indices->type == cgltf_type_scalar && (indices->component_type == cgltf_component_type_r_16u || indices->component_type == cgltf_component_type_r_32u));
            assert(positions->type == cgltf_type_vec3 && positions->component_type == cgltf_component_type_r_32f);
            assert(normals->count == positions->count && normals->type == cgltf_type_vec3 && normals->component_type == cgltf_component_type_r_32f);
            assert(uvs->count == positions->count && uvs->type == cgltf_type_vec2 && uvs->component_type == cgltf_component_type_r_32f);
            assert(tangents->count == positions->count && tangents->type == cgltf_type_vec4 && tangents->component_type == cgltf_component_type_r_32f);
            if (jointIndices) assert(jointIndices->count == positions->count && (jointIndices->component_type == cgltf_component_type_r_16u || jointIndices->component_type == cgltf_component_type_r_8u) && jointIndices->type == cgltf_type_vec4 && (jointIndices->stride == 8 || jointIndices->stride == 4));
            if (jointWeights) assert(jointWeights->count == positions->count && jointWeights->component_type == cgltf_component_type_r_32f && jointWeights->type == cgltf_type_vec4 && jointWeights->stride == 16);
            void* indicesBuffer = (uint8*)(indices->buffer_view->buffer->data) + indices->offset + indices->buffer_view->offset;
            float3* positionsBuffer = (float3*)((uint8*)(positions->buffer_view->buffer->data) + positions->offset + positions->buffer_view->offset);
            float3* normalsBuffer = (float3*)((uint8*)(normals->buffer_view->buffer->data) + normals->offset + normals->buffer_view->offset);
            float4* tangentsBuffer = (float4*)((uint8*)(tangents->buffer_view->buffer->data) + tangents->offset + tangents->buffer_view->offset);
            float2* uvsBuffer = (float2*)((uint8*)(uvs->buffer_view->buffer->data) + uvs->offset + uvs->buffer_view->offset);
            void* jointIndicesBuffer = jointIndices ? (uint8*)(jointIndices->buffer_view->buffer->data) + jointIndices->offset + jointIndices->buffer_view->offset : nullptr;
            float4* jointWeightsBuffer = jointWeights ? (float4*)((uint8*)(jointWeights->buffer_view->buffer->data) + jointWeights->offset + jointWeights->buffer_view->offset) : nullptr;

            ModelPrimitive& primitive = mesh.primitives.emplace_back();
            primitive.verticesBufferOffset = (uint)mesh.vertices.size();
            primitive.verticesCount = (uint)positions->count;
            primitive.indicesBufferOffset = (uint)mesh.indices.size();
            primitive.indicesCount = (uint)indices->count;
            for (uint vertexIndex = 0; vertexIndex < positions->count; vertexIndex++) {
                Vertex vertex = {.position = positionsBuffer[vertexIndex], .normal = normalsBuffer[vertexIndex], .tangent = tangentsBuffer[vertexIndex], .uv = uvsBuffer[vertexIndex]};
                if (jointIndicesBuffer) {
                    if (jointIndices->component_type == cgltf_component_type_r_16u) {
                        vertex.joints = ((uint16_4*)jointIndicesBuffer)[vertexIndex];
                    }
                    else {
                        vertex.joints = ((uint8_4*)jointIndicesBuffer)[vertexIndex];
                    }
                }
                if (jointWeightsBuffer) {
                    vertex.jointWeights = jointWeightsBuffer[vertexIndex];
                }
                mesh.vertices.push_back(vertex);
            }
            if (indices->component_type == cgltf_component_type_r_16u) {
                std::copy_n((uint16*)indicesBuffer, indices->count, std::back_inserter(mesh.indices));
            }
            else {
                std::copy_n((uint*)indicesBuffer, indices->count, std::back_inserter(mesh.indices));
            }
            if (gltfPrimitive.material) {
                primitive.material = &model->materials[gltfPrimitive.material - gltfData->materials];
            }
        }

        D3D12MA::ALLOCATION_DESC verticeIndicesBuffersAllocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC verticeIndicesBuffersDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.vertices);
        assert(SUCCEEDED(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.verticesBuffer, {}, nullptr)));
        mesh.verticesBuffer->GetResource()->SetName(std::format(L"{}Mesh{}VerticesBuffer", filePath.stem().wstring(), meshIndex).c_str());
        verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.indices);
        assert(SUCCEEDED(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.indicesBuffer, {}, nullptr)));
        mesh.verticesBuffer->GetResource()->SetName(std::format(L"{}Mesh{}IndicesBuffer", filePath.stem().wstring(), meshIndex).c_str());

        memcpy(d3d.stagingBuffer.ptr + d3d.stagingBuffer.size, mesh.vertices.data(), vectorSizeof(mesh.vertices));
        d3d.graphicsCmdList->CopyBufferRegion(mesh.verticesBuffer->GetResource(), 0, d3d.stagingBuffer.buffer->GetResource(), d3d.stagingBuffer.size, vectorSizeof(mesh.vertices));
        d3d.stagingBuffer.size += vectorSizeof(mesh.vertices);
        assert(d3d.stagingBuffer.size < d3d.stagingBuffer.capacity);
        memcpy(d3d.stagingBuffer.ptr + d3d.stagingBuffer.size, mesh.indices.data(), vectorSizeof(mesh.indices));
        d3d.graphicsCmdList->CopyBufferRegion(mesh.indicesBuffer->GetResource(), 0, d3d.stagingBuffer.buffer->GetResource(), d3d.stagingBuffer.size, vectorSizeof(mesh.indices));
        d3d.stagingBuffer.size += vectorSizeof(mesh.indices);
        assert(d3d.stagingBuffer.size < d3d.stagingBuffer.capacity);
        D3D12_RESOURCE_BARRIER bufferBarriers[2] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = mesh.verticesBuffer->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = mesh.indicesBuffer->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}},
        };
        d3d.graphicsCmdList->ResourceBarrier(countof(bufferBarriers), bufferBarriers);

        std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
        for (ModelPrimitive& primitive : mesh.primitives) {
            geometryDescs.push_back(D3D12_RAYTRACING_GEOMETRY_DESC{
                .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                .Triangles = {
                    .Transform3x4 = 0,
                    .IndexFormat = DXGI_FORMAT_R32_UINT,
                    .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
                    .IndexCount = (uint)primitive.indicesCount,
                    .VertexCount = (uint)primitive.verticesCount,
                    .IndexBuffer = mesh.indicesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.indicesBufferOffset * sizeof(uint),
                    .VertexBuffer = {mesh.verticesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.verticesBufferOffset * sizeof(struct Vertex), sizeof(Vertex)},
                },
            });
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE, .NumDescs = (uint)geometryDescs.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .pGeometryDescs = geometryDescs.data()};
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

        D3D12MA::ALLOCATION_DESC blasAllocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC blasDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        blasDesc.Width = prebuildInfo.ResultDataMaxSizeInBytes;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &mesh.blas, {}, nullptr)));
        mesh.blas->GetResource()->SetName(std::format(L"{}Mesh{}Blas", filePath.stem().wstring(), meshIndex).c_str());
        blasDesc.Width = prebuildInfo.ScratchDataSizeInBytes;
        assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &mesh.blasScratch, {}, nullptr)));
        mesh.blasScratch->GetResource()->SetName(std::format(L"{}Mesh{}BlasScratch", filePath.stem().wstring(), meshIndex).c_str());
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = mesh.blas->GetResource()->GetGPUVirtualAddress(), .Inputs = inputs, .ScratchAccelerationStructureData = mesh.blasScratch->GetResource()->GetGPUVirtualAddress()};
        d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
    }
    for (uint skinIndex = 0; skinIndex < gltfData->skins_count; skinIndex++) {
        cgltf_skin& gltfSkin = gltfData->skins[skinIndex];
        ModelSkin& skin = model->skins[skinIndex];
        assert(gltfSkin.inverse_bind_matrices->type == cgltf_type_mat4);
        assert(gltfSkin.inverse_bind_matrices->count == gltfSkin.joints_count);
        assert(gltfSkin.inverse_bind_matrices->stride == 16 * sizeof(float));
        assert(gltfSkin.inverse_bind_matrices->buffer_view->size == 64 * gltfSkin.joints_count);
        skin.joints.reserve(gltfSkin.joints_count);
        for (uint jointIndex = 0; jointIndex < gltfSkin.joints_count; jointIndex++) {
            cgltf_node* jointNode = gltfSkin.joints[jointIndex];
            ModelNode* node = &model->nodes[jointNode - gltfData->nodes];
            float* matsData = (float*)((uint8*)(gltfSkin.inverse_bind_matrices->buffer_view->buffer->data) + gltfSkin.inverse_bind_matrices->offset + gltfSkin.inverse_bind_matrices->buffer_view->offset);
            matsData += jointIndex * 16;
            skin.joints.push_back(ModelJoint{.node = node, .inverseBindMat = XMMatrix(matsData)});
        }
    }
    uint minParentCount = UINT_MAX;
    for (ModelSkin& skin : model->skins) {
        for (ModelJoint& joint : skin.joints) {
            ModelNode* jointNode = joint.node;
            ModelNode* parentNode = jointNode->parent;
            uint parentCount = 0;
            while (parentNode) {
                parentNode = parentNode->parent;
                parentCount++;
            }
            if (parentCount < minParentCount) {
                minParentCount = parentCount;
                model->skeletonRootNode = jointNode;
            }
        }
    }
    for (uint animationIndex = 0; animationIndex < gltfData->animations_count; animationIndex++) {
        cgltf_animation& gltfAnimation = gltfData->animations[animationIndex];
        ModelAnimation& animation = model->animations[animationIndex];
        if (gltfAnimation.name) animation.name = gltfAnimation.name;
        {
            cgltf_accessor* input = gltfAnimation.samplers[0].input;
            cgltf_size inputSize = gltfAnimation.samplers[0].input->count;
            float* keyFrameTimes = (float*)((uint8*)(input->buffer_view->buffer->data) + input->offset + input->buffer_view->offset);
            animation.timeLength = keyFrameTimes[inputSize - 1];
        }
        animation.samplers.reserve(gltfAnimation.samplers_count);
        for (cgltf_animation_sampler& gltfSampler : std::span(gltfAnimation.samplers, gltfAnimation.samplers_count)) {
            ModelAnimationSampler& sampler = animation.samplers.emplace_back();
            if (gltfSampler.interpolation == cgltf_interpolation_type_linear) {
                sampler.interpolation = AnimationSamplerInterpolationLinear;
            }
            else if (gltfSampler.interpolation == cgltf_interpolation_type_step) {
                sampler.interpolation = AnimationSamplerInterpolationStep;
            }
            else if (gltfSampler.interpolation == cgltf_interpolation_type_cubic_spline) {
                sampler.interpolation = AnimationSamplerInterpolationCubicSpline;
                assert(false);
            }
            else {
                assert(false);
            }
            assert(gltfSampler.input->component_type == cgltf_component_type_r_32f && gltfSampler.input->type == cgltf_type_scalar);
            assert(gltfSampler.output->component_type == cgltf_component_type_r_32f);
            assert((gltfSampler.output->type == cgltf_type_vec3 && gltfSampler.output->stride == sizeof(float3)) || (gltfSampler.output->type == cgltf_type_vec4 && gltfSampler.output->stride == sizeof(float4)));
            assert(gltfSampler.input->count >= 2 && gltfSampler.input->count == gltfSampler.output->count);
            float* inputs = (float*)((uint8*)(gltfSampler.input->buffer_view->buffer->data) + gltfSampler.input->offset + gltfSampler.input->buffer_view->offset);
            void* outputs = (uint8*)(gltfSampler.output->buffer_view->buffer->data) + gltfSampler.output->offset + gltfSampler.output->buffer_view->offset;
            assert(inputs[0] == 0.0f);
            assert(inputs[gltfSampler.input->count - 1] == animation.timeLength);
            sampler.keyFrames.reserve(gltfSampler.input->count);
            for (uint frameIndex = 0; frameIndex < gltfSampler.input->count; frameIndex++) {
                ModelAnimationSamplerKeyFrame& keyFrame = sampler.keyFrames.emplace_back();
                keyFrame.time = inputs[frameIndex];
                if (gltfSampler.output->type == cgltf_type_vec3) {
                    keyFrame.xyzw = ((float3*)outputs)[frameIndex];
                }
                else {
                    keyFrame.xyzw = ((float4*)outputs)[frameIndex];
                }
            }
        }
        animation.channels.reserve(gltfAnimation.channels_count);
        for (cgltf_animation_channel& gltfChannel : std::span(gltfAnimation.channels, gltfAnimation.channels_count)) {
            ModelAnimationChannel& channel = animation.channels.emplace_back();
            uint nodeIndex = (uint)(gltfChannel.target_node - gltfData->nodes);
            uint samplerIndex = (uint)(gltfChannel.sampler - gltfAnimation.samplers);
            assert(nodeIndex >= 0 && nodeIndex < gltfData->nodes_count);
            assert(samplerIndex >= 0 && samplerIndex < gltfAnimation.samplers_count);
            channel.node = &model->nodes[nodeIndex];
            channel.sampler = &animation.samplers[samplerIndex];
            if (gltfChannel.target_path == cgltf_animation_path_type_translation) {
                assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                channel.type = AnimationChannelTypeTranslate;
            }
            else if (gltfChannel.target_path == cgltf_animation_path_type_rotation) {
                assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec4);
                channel.type = AnimationChannelTypeRotate;
            }
            else if (gltfChannel.target_path == cgltf_animation_path_type_scale) {
                assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                channel.type = AnimationChannelTypeScale;
            }
            else {
                assert(false);
            }
        }
    }
    for (uint imageIndex = 0; imageIndex < gltfData->images_count; imageIndex++) {
        cgltf_image& gltfImage = gltfData->images[imageIndex];
        ModelImage& image = model->images[imageIndex];
        std::filesystem::path imageFilePath = gltfFileFolderPath / gltfImage.uri;
        std::filesystem::path imageDDSFilePath = imageFilePath;
        imageDDSFilePath.replace_extension(".dds");
        if (std::filesystem::exists(imageDDSFilePath)) {
            image.gpuData = d3dCreateImageDDS(imageDDSFilePath, d3d.graphicsCmdList, std::format(L"{}Image{}", filePath.stem().wstring(), imageIndex).c_str(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (std::filesystem::exists(imageFilePath)) {
            image.gpuData = d3dCreateImageSTB(imageFilePath, d3d.graphicsCmdList, std::format(L"{}Image{}", filePath.stem().wstring(), imageIndex).c_str(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else {
            assert(false);
        }
    }
    for (uint textureIndex = 0; textureIndex < gltfData->textures_count; textureIndex++) {
        cgltf_texture& gltfTexture = gltfData->textures[textureIndex];
        ModelTexture& texture = model->textures[textureIndex];
        assert(gltfTexture.image);
        texture.image = &model->images[gltfTexture.image - &gltfData->images[0]];
        if (gltfTexture.sampler) {
            switch (gltfTexture.sampler->wrap_s) {
                case 10497: texture.wrapU = D3D12_TEXTURE_ADDRESS_MODE_WRAP; break;
                case 33071: texture.wrapU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP; break;
                case 33648: texture.wrapU = D3D12_TEXTURE_ADDRESS_MODE_MIRROR; break;
                default: assert(false); break;
            }
            switch (gltfTexture.sampler->wrap_t) {
                case 10497: texture.wrapV = D3D12_TEXTURE_ADDRESS_MODE_WRAP; break;
                case 33071: texture.wrapV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP; break;
                case 33648: texture.wrapV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR; break;
                default: assert(false); break;
            }
        }
    }
    for (uint materialIndex = 0; materialIndex < gltfData->materials_count; materialIndex++) {
        cgltf_material& gltfMaterial = gltfData->materials[materialIndex];
        ModelMaterial& material = model->materials[materialIndex];
        if (gltfMaterial.name) material.name = gltfMaterial.name;
        material.emissive = float3(gltfMaterial.emissive_factor);
        if (gltfMaterial.emissive_texture.texture) {
            assert(gltfMaterial.emissive_texture.texcoord == 0);
            assert(!gltfMaterial.emissive_texture.has_transform);
            material.emissiveTexture = &model->textures[gltfMaterial.emissive_texture.texture - gltfData->textures];
            D3D12_RESOURCE_DESC imageDesc = material.emissiveTexture->image->gpuData->GetResource()->GetDesc();
            material.emissiveTexture->srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
        }
        assert(gltfMaterial.has_pbr_metallic_roughness);
        material.baseColor = float4(gltfMaterial.pbr_metallic_roughness.base_color_factor);
        material.metallic = gltfMaterial.pbr_metallic_roughness.metallic_factor;
        material.roughness = gltfMaterial.pbr_metallic_roughness.roughness_factor;
        if (gltfMaterial.pbr_metallic_roughness.base_color_texture.texture) {
            assert(gltfMaterial.pbr_metallic_roughness.base_color_texture.texcoord == 0);
            assert(!gltfMaterial.pbr_metallic_roughness.base_color_texture.has_transform);
            material.baseColorTexture = &model->textures[gltfMaterial.pbr_metallic_roughness.base_color_texture.texture - gltfData->textures];
            D3D12_RESOURCE_DESC imageDesc = material.baseColorTexture->image->gpuData->GetResource()->GetDesc();
            material.baseColorTexture->srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
            if (material.baseColorTexture->srvDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
                material.baseColorTexture->srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            }
            else if (material.baseColorTexture->srvDesc.Format == DXGI_FORMAT_BC7_UNORM) {
                material.baseColorTexture->srvDesc.Format = DXGI_FORMAT_BC7_UNORM_SRGB;
            }
        }
        if (gltfMaterial.pbr_metallic_roughness.metallic_roughness_texture.texture) {
            assert(gltfMaterial.pbr_metallic_roughness.metallic_roughness_texture.texcoord == 0);
            assert(!gltfMaterial.pbr_metallic_roughness.metallic_roughness_texture.has_transform);
            material.metallicRoughnessTexture = &model->textures[gltfMaterial.pbr_metallic_roughness.metallic_roughness_texture.texture - gltfData->textures];
            D3D12_RESOURCE_DESC imageDesc = material.metallicRoughnessTexture->image->gpuData->GetResource()->GetDesc();
            material.metallicRoughnessTexture->srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
        }
        if (gltfMaterial.normal_texture.texture) {
            assert(gltfMaterial.normal_texture.texcoord == 0);
            assert(!gltfMaterial.normal_texture.has_transform);
            material.normalTexture = &model->textures[gltfMaterial.normal_texture.texture - gltfData->textures];
            D3D12_RESOURCE_DESC imageDesc = material.normalTexture->image->gpuData->GetResource()->GetDesc();
            material.normalTexture->srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
        }
    }
    d3dCmdListExecute();
    d3dSignalFence(&d3d.transferFence);
    d3dWaitFence(&d3d.transferFence);
    {
        std::filesystem::path convexMeshPath = gltfFilePath;
        convexMeshPath.replace_extension("convexMesh");
        if (std::filesystem::exists(convexMeshPath)) {
            std::vector<uint8> data = fileReadBytes(convexMeshPath);
            PxDefaultMemoryInputData input(data.data(), (uint)data.size());
            model->convexMesh = pxPhysics->createConvexMesh(input);
            assert(model->convexMesh);
        }
    }
    {
        std::filesystem::path triangleMeshPath = gltfFilePath;
        triangleMeshPath.replace_extension("triangleMesh");
        if (std::filesystem::exists(triangleMeshPath)) {
            std::vector<uint8> data = fileReadBytes(triangleMeshPath);
            PxDefaultMemoryInputData input(data.data(), (uint)data.size());
            model->triangleMesh = pxPhysics->createTriangleMesh(input);
            assert(model->triangleMesh);
        }
    }
    return model;
}

Model* modelLoadFBX(const std::filesystem::path& filePath) {
    assert(false);
    return nullptr;
}

Model* modelLoad(const std::filesystem::path& filePath) {
    for (Model& m : models) {
        if (m.filePath == filePath) {
            return &m;
        }
    }
    if (filePath.extension() == ".gltf") {
        return modelLoadGLTF(filePath);
    }
    else if (filePath.extension() == ".fbx") {
        return modelLoadFBX(filePath);
    }
    else {
        assert(false);
        return nullptr;
    }
}

bool modelGenerateConvexMesh(Model* model) {
    std::vector<PxVec3> vertices;
    for (ModelNode* node : model->meshNodes) {
        for (Vertex& vertex : node->mesh->vertices) {
            XMVector p = XMVector3Transform(XMVector3Transform(vertex.position.toXMVector(), node->globalTransform), XMMatrixScaling(1, 1, -1));
            vertices.push_back(PxVec3(XMVectorGetX(p), XMVectorGetY(p), XMVectorGetZ(p)));
        }
    }
    PxConvexMeshDesc meshDesc;
    meshDesc.points.count = (uint)vertices.size();
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = vertices.data();
    meshDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxCookingParams params(PxTolerancesScale(1, 10));
    PxDefaultMemoryOutputStream output;
    PxConvexMeshCookingResult::Enum result;
    if (!PxCookConvexMesh(params, meshDesc, output, &result)) {
        return false;
    }
    else {
        PxDefaultMemoryInputData input(output.getData(), output.getSize());
        model->convexMesh = pxPhysics->createConvexMesh(input);
        std::filesystem::path convexMeshFilePath = model->filePath;
        convexMeshFilePath.replace_extension("convexMesh");
        fileWriteBytes(convexMeshFilePath, output.getData(), output.getSize());
        return true;
    }
}

bool modelGenerateTriangleMesh(Model* model) {
    std::vector<PxVec3> vertices;
    std::vector<uint32> indices;
    uint indicesOffset = 0;
    for (ModelNode* node : model->meshNodes) {
        for (Vertex& vertex : node->mesh->vertices) {
            XMVector p = XMVector3Transform(XMVector3Transform(vertex.position.toXMVector(), node->globalTransform), XMMatrixScaling(1, 1, -1));
            vertices.push_back(PxVec3(XMVectorGetX(p), XMVectorGetY(p), XMVectorGetZ(p)));
        }
        for (uint32 index : node->mesh->indices) {
            indices.push_back(index + indicesOffset);
        }
        indicesOffset += (uint)node->mesh->vertices.size();
    }
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = (uint)vertices.size();
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = vertices.data();
    meshDesc.triangles.count = (uint)indices.size() / 3;
    meshDesc.triangles.stride = 3 * 4;
    meshDesc.triangles.data = indices.data();
    PxCookingParams params(PxTolerancesScale(1, 10));
    PxDefaultMemoryOutputStream output;
    PxTriangleMeshCookingResult::Enum result;
    if (!PxCookTriangleMesh(params, meshDesc, output, &result)) {
        return false;
    }
    else {
        PxDefaultMemoryInputData input(output.getData(), output.getSize());
        model->triangleMesh = pxPhysics->createTriangleMesh(input);
        std::filesystem::path triangleMeshFilePath = model->filePath;
        triangleMeshFilePath.replace_extension("triangleMesh");
        fileWriteBytes(triangleMeshFilePath, output.getData(), output.getSize());
        return true;
    }
}

// bool modelCreateConvexMesh(Model& model, bool forceCooking = true) {
//     assert(!model.physxConvexMesh);
//     std::filesystem::path convexMeshFilePath = model.filePath;
//     convexMeshFilePath.replace_extension("convexMesh");
//     if (!forceCooking && std::filesystem::exists(exeDir / convexMeshFilePath)) {
//         std::vector<uint8> convexMeshData = fileReadBytes(exeDir / convexMeshFilePath);
//         PxDefaultMemoryInputData input(convexMeshData.data(), (uint)convexMeshData.size());
//         model.physxConvexMesh = pxPhysics->createConvexMesh(input);
//     } else {
//         std::vector<float3> points;
//         for (ModelNode* meshNode : model.meshNodes) {
//             meshNode->globalTransform;
//             XMMatrix transformMat = XMMatrixMultiply(meshNode->globalTransform, XMMatrixScaling(scaleFactor, scaleFactor, -scaleFactor));
//             for (Vertex& vertex : meshNode->mesh->vertices) {
//                 points.push_back(float3(XMVector3Transform(vertex.position.toXMVector(), transformMat)));
//             }
//         }
//         PxCookingParams cookingParams((PxTolerancesScale()));
//         PxConvexMeshDesc convexMeshDesc;
//         convexMeshDesc.points.count = (uint)points.size();
//         convexMeshDesc.points.stride = sizeof(float3);
//         convexMeshDesc.points.data = points.data();
//         convexMeshDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eDISABLE_MESH_VALIDATION;
//         PxDefaultMemoryOutputStream outputStream;
//         PxConvexMeshCookingResult::Enum cookingResult;
//         if (!PxCookConvexMesh(cookingParams, convexMeshDesc, outputStream, &cookingResult)) {
//             return false;
//         }
//         std::filesystem::path convexMeshFilePath = model.filePath;
//         convexMeshFilePath.replace_extension("convexMesh");
//         fileWriteBytes(exeDir / convexMeshFilePath, outputStream.getData(), outputStream.getSize());
//         PxDefaultMemoryInputData input(outputStream.getData(), outputStream.getSize());
//         model.physxConvexMesh = pxPhysics->createConvexMesh(input);
//     }
//     return true;
// }

// bool modelCreateTriangleMesh(Model& model, bool forceCooking = true) {
//     assert(!model.physxTriangleMesh);
//     std::filesystem::path triangleMeshFilePath = model.filePath;
//     triangleMeshFilePath.replace_extension("triangleMesh");
//     if (!forceCooking && std::filesystem::exists(exeDir / triangleMeshFilePath)) {
//         std::vector<uint8> triangleMeshData = fileReadBytes(exeDir / triangleMeshFilePath);
//         PxDefaultMemoryInputData input(triangleMeshData.data(), (uint)triangleMeshData.size());
//         model.physxTriangleMesh = physxPhysics->createTriangleMesh(input);
//     } else {
//         std::vector<float3> points;
//         for (ModelNode* meshNode : model.meshNodes) {
//             meshNode->globalTransform;
//             XMMatrix transformMat = XMMatrixMultiply(meshNode->globalTransform, XMMatrixScaling(scaleFactor, scaleFactor, -scaleFactor));
//             for (Vertex& vertex : meshNode->mesh->vertices) {
//                 points.push_back(float3(XMVector3Transform(vertex.position.toXMVector(), transformMat)));
//             }
//         }
//         PxCookingParams cookingParams(PxTolerancesScale(meters(1), meters(10)));
//         PxTriangleMeshDesc triangleMeshDesc;
//         //PxConvexMeshDesc convexMeshDesc;
//         //convexMeshDesc.points.count = (uint)points.size();
//         //convexMeshDesc.points.stride = sizeof(float3);
//         //convexMeshDesc.points.data = points.data();
//         //convexMeshDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eDISABLE_MESH_VALIDATION;
//         PxDefaultMemoryOutputStream outputStream;
//         PxTriangleMeshCookingResult::Enum cookingResult;
//         if (!PxCookTriangleMesh(cookingParams, triangleMeshDesc, outputStream, &cookingResult)) {
//             return false;
//         }
//         std::filesystem::path convexMeshFilePath = model.filePath;
//         convexMeshFilePath.replace_extension("convexMesh");
//         fileWriteBytes(exeDir / convexMeshFilePath, outputStream.getData(), outputStream.getSize());
//         PxDefaultMemoryInputData input(outputStream.getData(), outputStream.getSize());
//         model.physxConvexMesh = physxPhysics->createConvexMesh(input);
//     }
//     return true;
// }

void modelTraverseNodesImGui(ModelNode* node) {
    if (ImGui::TreeNode(node->name.c_str())) {
        for (ModelNode* childNode : node->children) {
            modelTraverseNodesImGui(childNode);
        }
        ImGui::TreePop();
    }
}

void modelTraverseNodesAndGetGlobalTransformMats(Model* model, ModelNode* node, const XMMatrix& parentMat, const std::vector<Transform>& nodeLocalTransforms, std::vector<XMMatrix>& nodeGlobalTransformMats) {
    int64 nodeIndex = node - &model->nodes[0];
    XMMatrix mat = XMMatrixMultiply(nodeLocalTransforms[nodeIndex].toMat(), parentMat);
    nodeGlobalTransformMats[nodeIndex] = mat;
    for (ModelNode* childNode : node->children) {
        modelTraverseNodesAndGetGlobalTransformMats(model, childNode, mat, nodeLocalTransforms, nodeGlobalTransformMats);
    }
}

ModelInstance modelInstanceInit(Model* model) {
    ModelInstance modelInstance = {};
    modelInstance.model = model;
    modelInstance.meshNodes.resize(model->meshNodes.size());
    for (uint meshNodeIndex = 0; meshNodeIndex < model->meshNodes.size(); meshNodeIndex++) {
        ModelNode* meshNode = model->meshNodes[meshNodeIndex];
        ModelInstanceMeshNode& instanceMeshNode = modelInstance.meshNodes[meshNodeIndex];
        if (!meshNode->skin || model->animations.size() == 0) {
            instanceMeshNode.verticesBuffer = meshNode->mesh->verticesBuffer;
            instanceMeshNode.blas = meshNode->mesh->blas;
            instanceMeshNode.blasScratch = meshNode->mesh->blasScratch;
        }
        else {
            D3D12MA::ALLOCATION_DESC verticesBufferAllocDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC verticesBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = meshNode->mesh->verticesBuffer->GetSize(), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            assert(SUCCEEDED(d3d.allocator->CreateResource(&verticesBufferAllocDesc, &verticesBufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &instanceMeshNode.verticesBuffer, {}, nullptr)));
            D3D12MA::ALLOCATION_DESC blasAllocDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC blasDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            blasDesc.Width = meshNode->mesh->blas->GetSize();
            assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocDesc, &blasDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &instanceMeshNode.blas, {}, nullptr)));
            blasDesc.Width = meshNode->mesh->blasScratch->GetSize();
            assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocDesc, &blasDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &instanceMeshNode.blasScratch, {}, nullptr)));
        }
    }
    if (model->animations.size() > 0) {
        modelInstance.animation = &model->animations[0];
        modelInstance.localTransforms.resize(model->nodes.size());
        modelInstance.globalTransformMats.resize(model->nodes.size());
        modelInstance.skins.resize(model->skins.size());
        for (uint skinIndex = 0; skinIndex < model->skins.size(); skinIndex++) {
            modelInstance.skins[skinIndex].mats.resize(model->skins[skinIndex].joints.size());
            D3D12MA::ALLOCATION_DESC jointBufferAllocDesc = {.HeapType = D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC jointBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = vectorSizeof(modelInstance.skins[skinIndex].mats), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            assert(SUCCEEDED(d3d.allocator->CreateResource(&jointBufferAllocDesc, &jointBufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &modelInstance.skins[skinIndex].matsBuffer, {}, nullptr)));
            assert(SUCCEEDED(modelInstance.skins[skinIndex].matsBuffer->GetResource()->Map(0, nullptr, (void**)&modelInstance.skins[skinIndex].matsBufferPtr)));
        }
    }
    return modelInstance;
}

ModelInstance modelInstanceInit(const std::filesystem::path& filePath) {
    Model* model = nullptr;
    for (Model& m : models) {
        if (m.filePath == filePath) model = &m;
    }
    assert(model);
    return modelInstanceInit(model);
}

void modelInstanceRelease(ModelInstance& modelInstance) {
    for (ModelInstanceSkin& skin : modelInstance.skins) {
        skin.matsBuffer->Release();
    }
    for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance.model->meshNodes.size(); meshNodeIndex++) {
        if (modelInstance.model->meshNodes[meshNodeIndex]->skin) {
            modelInstance.meshNodes[meshNodeIndex].verticesBuffer->Release();
            modelInstance.meshNodes[meshNodeIndex].blas->Release();
            modelInstance.meshNodes[meshNodeIndex].blasScratch->Release();
        }
    }
}

void modelInstanceGetSkeletonVisualization(ModelInstance* modelInstance, ModelNode* node, const XMMatrix& transformMat) {
    int64 nodeIndex = node - &modelInstance->model->nodes[0];
    XMVector nodePosition = XMVector3Transform(XMVector3Transform(xmVectorZero, modelInstance->globalTransformMats[nodeIndex]), transformMat);
    debugSpheres.push_back(Sphere{.center = float3(nodePosition), .radius = 0.03f, .color = 0xffffffff});
    for (ModelNode* childNode : node->children) {
        int64 childNodeIndex = childNode - &modelInstance->model->nodes[0];
        XMVector childNodePosition = XMVector3Transform(XMVector3Transform(xmVectorZero, modelInstance->globalTransformMats[childNodeIndex]), transformMat);
        debugLines.push_back(Line{.p0 = float3(nodePosition), .p1 = float3(childNodePosition), .radius = 0.01f, .color = 0xffffffff});
        modelInstanceGetSkeletonVisualization(modelInstance, childNode, transformMat);
    }
}

void modelInstanceImGui(ModelInstance* modelInstance) {
    if (ImGui::TreeNode("Model")) {
        ImGui::Text(std::format("File: {}", modelInstance->model->filePath.string()).c_str());
        if (ImGui::TreeNode("Animations")) {
            for (uint animationIndex = 0; animationIndex < modelInstance->model->animations.size(); animationIndex++) {
                ModelAnimation& modelAnimation = modelInstance->model->animations[animationIndex];
                ImGui::Text(std::format("#{}: {}", animationIndex, modelAnimation.name).c_str());
                ImGui::SameLine(ImGui::GetWindowWidth() * 0.8f);
                ImGui::PushID(animationIndex);
                if (ImGui::Button("play")) {
                    modelInstance->animation = &modelAnimation;
                    modelInstance->animationTime = 0;
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Hierarchy")) {
            for (ModelNode* rootNode : modelInstance->model->rootNodes) {
                modelTraverseNodesImGui(rootNode);
            }
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
}

void modelInstanceUpdateAnimation(ModelInstance* modelInstance, double time) {
    if (!modelInstance->animation || pathTracer) return;

    modelInstance->animationTime += time;
    if (modelInstance->animationTime > modelInstance->animation->timeLength) {
        modelInstance->animationTime -= modelInstance->animation->timeLength;
    }
    for (Transform& transform : modelInstance->localTransforms) transform = Transform{};
    {
        for (ModelAnimationChannel& channel : modelInstance->animation->channels) {
            float4 frame0 = channel.sampler->keyFrames[0].xyzw;
            float4 frame1 = channel.sampler->keyFrames[1].xyzw;
            float percentage = 0;
            for (uint frameIndex = 1; frameIndex < channel.sampler->keyFrames.size(); frameIndex++) {
                ModelAnimationSamplerKeyFrame& keyFrame = channel.sampler->keyFrames[frameIndex];
                if (modelInstance->animationTime <= keyFrame.time) {
                    ModelAnimationSamplerKeyFrame& keyFramePrevious = channel.sampler->keyFrames[frameIndex - 1];
                    frame0 = keyFramePrevious.xyzw;
                    frame1 = keyFrame.xyzw;
                    percentage = ((float)modelInstance->animationTime - keyFramePrevious.time) / (keyFrame.time - keyFramePrevious.time);
                    break;
                }
            }
            int64 nodeIndex = channel.node - &modelInstance->model->nodes[0];
            if (channel.type == AnimationChannelTypeTranslate) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    modelInstance->localTransforms[nodeIndex].t = lerp(frame0.xyz(), frame1.xyz(), percentage);
                }
                else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    modelInstance->localTransforms[nodeIndex].t = percentage < 1.0f ? frame0.xyz() : frame1.xyz();
                }
            }
            else if (channel.type == AnimationChannelTypeRotate) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    modelInstance->localTransforms[nodeIndex].r = slerp(frame0, frame1, percentage);
                }
                else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    modelInstance->localTransforms[nodeIndex].r = percentage < 1.0f ? frame0 : frame1;
                }
            }
            else if (channel.type == AnimationChannelTypeScale) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    modelInstance->localTransforms[nodeIndex].s = lerp(frame0.xyz(), frame1.xyz(), percentage);
                }
                else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    modelInstance->localTransforms[nodeIndex].s = percentage < 1.0f ? frame0.xyz() : frame1.xyz();
                }
            }
        }
    }
    for (ModelNode* rootNode : modelInstance->model->rootNodes) {
        modelTraverseNodesAndGetGlobalTransformMats(modelInstance->model, rootNode, xmMatrixIdentity, modelInstance->localTransforms, modelInstance->globalTransformMats);
    }
    for (uint skinIndex = 0; skinIndex < modelInstance->model->skins.size(); skinIndex++) {
        ModelSkin& skin = modelInstance->model->skins[skinIndex];
        ModelInstanceSkin& instanceSkin = modelInstance->skins[skinIndex];
        for (uint jointIndex = 0; jointIndex < skin.joints.size(); jointIndex++) {
            int64 nodeIndex = skin.joints[jointIndex].node - &modelInstance->model->nodes[0];
            instanceSkin.mats[jointIndex] = XMMatrixMultiply(skin.joints[jointIndex].inverseBindMat, modelInstance->globalTransformMats[nodeIndex]);
        }
        memcpy(instanceSkin.matsBufferPtr, instanceSkin.mats.data(), vectorSizeof(instanceSkin.mats));
    }
}

void modelsAppendDescriptorsAndBlasGeometriesInfos() {
    for (Model& model : models) {
        for (ModelNode* meshNode : model.meshNodes) {
            meshNode->mesh->blasGeometriesInfoOffset = (uint)blasGeometriesInfos.size();
            for (const ModelPrimitive& primitive : meshNode->mesh->primitives) {
                uint descriptorsHeapOffset = d3d.cbvSrvUavDescriptorHeap.size;
                D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.verticesBufferOffset, .NumElements = primitive.verticesCount, .StructureByteStride = sizeof(struct Vertex)}};
                D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.indicesBufferOffset, .NumElements = primitive.indicesCount, .StructureByteStride = sizeof(uint)}};
                d3dAppendSRVDescriptor(&vertexBufferDesc, meshNode->mesh->verticesBuffer->GetResource());
                d3dAppendSRVDescriptor(&indexBufferDesc, meshNode->mesh->indicesBuffer->GetResource());
                if (primitive.material) {
                    if (primitive.material->emissiveTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->emissiveTexture->srvDesc, primitive.material->emissiveTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultEmissiveTextureSRVDesc, d3d.defaultEmissiveTexture->GetResource());
                    }
                    if (primitive.material->baseColorTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->baseColorTexture->srvDesc, primitive.material->baseColorTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultBaseColorTextureSRVDesc, d3d.defaultBaseColorTexture->GetResource());
                    }
                    if (primitive.material->metallicRoughnessTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->metallicRoughnessTexture->srvDesc, primitive.material->metallicRoughnessTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultMetallicRoughnessTextureSRVDesc, d3d.defaultMetallicRoughnessTexture->GetResource());
                    }
                    if (primitive.material->normalTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->normalTexture->srvDesc, primitive.material->normalTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultNormalTextureSRVDesc, d3d.defaultNormalTexture->GetResource());
                    }
                    blasGeometriesInfos.push_back(BLASGeometryInfo{.descriptorsHeapOffset = descriptorsHeapOffset, .emissiveFactor = primitive.material->emissive, .metallicFactor = primitive.material->metallic, .baseColorFactor = primitive.material->baseColor.xyz(), .roughnessFactor = primitive.material->roughness});
                }
                else {
                    d3dAppendSRVDescriptor(&d3d.defaultEmissiveTextureSRVDesc, d3d.defaultEmissiveTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultBaseColorTextureSRVDesc, d3d.defaultBaseColorTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultMetallicRoughnessTextureSRVDesc, d3d.defaultMetallicRoughnessTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultNormalTextureSRVDesc, d3d.defaultNormalTexture->GetResource());
                    blasGeometriesInfos.push_back(BLASGeometryInfo{.descriptorsHeapOffset = descriptorsHeapOffset, .emissiveFactor = {0.0f, 0.0f, 0.0f}, .metallicFactor = 0.0f, .baseColorFactor = {0.7f, 0.7f, 0.7f}, .roughnessFactor = 0.7f});
                }
            }
        }
    }
}

void modelInstanceAddBLASInstancesToTLAS(ModelInstance& modelInstance, const XMMatrix& objectTransform, const XMMatrix& objectTransformPrevFrame, ObjectType objectType, uint objectIndex, uint instanceFlags, uint color) {
    ZoneScopedN("modelInstanceAddBLASInstancesToTLAS");
    for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance.meshNodes.size(); meshNodeIndex++) {
        ModelInstanceMeshNode* instanceMeshNode = &modelInstance.meshNodes[meshNodeIndex];
        ModelNode* meshNode = modelInstance.model->meshNodes[meshNodeIndex];
        XMMatrix transform = meshNode->globalTransform;
        if (modelInstance.globalTransformMats.size() > 0) {
            transform = XMMatrixMultiply(transform, modelInstance.globalTransformMats[meshNode - &modelInstance.model->nodes[0]]);
        }
        transform = XMMatrixMultiply(transform, XMMatrixScaling(1, 1, -1)); // convert RH to LH
        transform = XMMatrixMultiply(transform, objectTransform);
        XMMatrix transformT = XMMatrixTranspose(transform);
        D3D12_RAYTRACING_INSTANCE_DESC blasInstanceDesc = {.InstanceMask = objectType, .Flags = 0, .AccelerationStructure = instanceMeshNode->blas->GetResource()->GetGPUVirtualAddress()};
#ifdef EDITOR
        if (editor.mode == EditorModeFreeCam && objectType != ObjectTypeNone && editor.selectedObjectType == objectType && editor.selectedObjectIndex == objectIndex && editor.selectedObjectXRay) {
            blasInstanceDesc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
        }
#endif
        memcpy(blasInstanceDesc.Transform, &transformT, sizeof(blasInstanceDesc.Transform));
        BLASInstanceInfo blasInstanceInfo = {.flags = instanceFlags, .color = color, .objectType = objectType, .objectIndex = objectIndex};
        XMStoreFloat3x3(&blasInstanceInfo.transformNormalMat, transform);
        XMMatrix normalTransform = XMMatrixTranspose(XMMatrixInverse(nullptr, XMLoadFloat3x3(&blasInstanceInfo.transformNormalMat)));
        XMStoreFloat3x3(&blasInstanceInfo.transformNormalMat, normalTransform);
        if (instanceMeshNode->verticesBuffer == meshNode->mesh->verticesBuffer) {
            blasInstanceInfo.blasGeometriesOffset = meshNode->mesh->blasGeometriesInfoOffset;
        }
        else {
            blasInstanceInfo.blasGeometriesOffset = (uint)blasGeometriesInfos.size();
            for (const ModelPrimitive& primitive : meshNode->mesh->primitives) {
                uint descriptorsHeapOffset = d3d.cbvSrvUavDescriptorHeap.size;
                D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.verticesBufferOffset, .NumElements = primitive.verticesCount, .StructureByteStride = sizeof(struct Vertex)}};
                D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.indicesBufferOffset, .NumElements = primitive.indicesCount, .StructureByteStride = sizeof(uint)}};
                d3dAppendSRVDescriptor(&vertexBufferDesc, instanceMeshNode->verticesBuffer->GetResource());
                d3dAppendSRVDescriptor(&indexBufferDesc, meshNode->mesh->indicesBuffer->GetResource());
                if (primitive.material) {
                    if (primitive.material->emissiveTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->emissiveTexture->srvDesc, primitive.material->emissiveTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultEmissiveTextureSRVDesc, d3d.defaultEmissiveTexture->GetResource());
                    }
                    if (primitive.material->baseColorTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->baseColorTexture->srvDesc, primitive.material->baseColorTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultBaseColorTextureSRVDesc, d3d.defaultBaseColorTexture->GetResource());
                    }
                    if (primitive.material->metallicRoughnessTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->metallicRoughnessTexture->srvDesc, primitive.material->metallicRoughnessTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultMetallicRoughnessTextureSRVDesc, d3d.defaultMetallicRoughnessTexture->GetResource());
                    }
                    if (primitive.material->normalTexture) {
                        d3dAppendSRVDescriptor(&primitive.material->normalTexture->srvDesc, primitive.material->normalTexture->image->gpuData->GetResource());
                    }
                    else {
                        d3dAppendSRVDescriptor(&d3d.defaultNormalTextureSRVDesc, d3d.defaultNormalTexture->GetResource());
                    }
                    blasGeometriesInfos.push_back(BLASGeometryInfo{.descriptorsHeapOffset = descriptorsHeapOffset, .emissiveFactor = primitive.material->emissive, .metallicFactor = primitive.material->metallic, .baseColorFactor = primitive.material->baseColor.xyz(), .roughnessFactor = primitive.material->roughness});
                }
                else {
                    d3dAppendSRVDescriptor(&d3d.defaultEmissiveTextureSRVDesc, d3d.defaultEmissiveTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultBaseColorTextureSRVDesc, d3d.defaultBaseColorTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultMetallicRoughnessTextureSRVDesc, d3d.defaultMetallicRoughnessTexture->GetResource());
                    d3dAppendSRVDescriptor(&d3d.defaultNormalTextureSRVDesc, d3d.defaultNormalTexture->GetResource());
                    blasGeometriesInfos.push_back(BLASGeometryInfo{.descriptorsHeapOffset = descriptorsHeapOffset, .emissiveFactor = {0.0f, 0.0f, 0.0f}, .metallicFactor = 0.0f, .baseColorFactor = {0.7f, 0.7f, 0.7f}, .roughnessFactor = 0.7f});
                }
            }
        }
        blasInstancesDescs.push_back(blasInstanceDesc);
        blasInstancesInfos.push_back(blasInstanceInfo);
    }
}

void loadSimpleAssets() {
    modelInstanceSphere = modelInstanceInit("assets/models/sphere/gltf/sphere.gltf");
    modelInstanceCube = modelInstanceInit("assets/models/cube/gltf/cube.gltf");
    modelInstanceCylinder = modelInstanceInit("assets/models/cylinder/gltf/cylinder.gltf");
}

void physxInit() {
    pxFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, pxAllocator, pxErrorCallback);
    PxPvd* pvd = nullptr;
#ifdef PHYSX_PVD
    pvd = PxCreatePvd(*pxFoundation);
    assert(pvd);
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    assert(transport);
    pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
#endif
    pxPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *pxFoundation, PxTolerancesScale(1, 10), true, pvd);
    pxDispatcher = PxDefaultCpuDispatcherCreate(4);
    PxSceneDesc sceneDesc(pxPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0, -10, 0);
    sceneDesc.cpuDispatcher = pxDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    pxScene = pxPhysics->createScene(sceneDesc);
    pxScene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0f);
    pxScene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 2.0f);
    pxDefaultMaterial = pxPhysics->createMaterial(0.7f, 0.7f, 0.4f);
    pxControllerManager = PxCreateControllerManager(*pxScene);
    pxControllerManager->setOverlapRecoveryModule(false);
}

void physxScaleShapeGeometry(PxShape* shape, float3 scale) {
    if (scale == float3(1, 1, 1)) return;
    const PxGeometry& geometry = shape->getGeometry();
    PxGeometryType::Enum geometryType = geometry.getType();
    if (geometryType == PxGeometryType::eSPHERE) {
        PxSphereGeometry sphereGeomerty = (const PxSphereGeometry&)geometry;
        sphereGeomerty.radius *= ((abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0f);
        shape->setGeometry(sphereGeomerty);
    }
    else if (geometryType == PxGeometryType::eCAPSULE) {
        PxCapsuleGeometry capsuleGeomerty = (const PxCapsuleGeometry&)geometry;
        capsuleGeomerty.radius *= (abs(scale.x) + abs(scale.z)) / 2.0f;
        capsuleGeomerty.halfHeight *= abs(scale.y);
        shape->setGeometry(capsuleGeomerty);
    }
    else if (geometryType == PxGeometryType::eBOX) {
        PxBoxGeometry boxGeometry = (const PxBoxGeometry&)geometry;
        boxGeometry.halfExtents.x *= abs(scale.x);
        boxGeometry.halfExtents.y *= abs(scale.y);
        boxGeometry.halfExtents.z *= abs(scale.z);
        shape->setGeometry(boxGeometry);
    }
    else if (geometryType == PxGeometryType::eCONVEXMESH) {
        PxConvexMeshGeometry convexMeshGeometry = (const PxConvexMeshGeometry&)geometry;
        convexMeshGeometry.scale = PxMeshScale(scale.toPxVec3());
        shape->setGeometry(convexMeshGeometry);
    }
    else {
        assert(false);
    }
}

void playerCameraReset(bool keepPitchYaw) {
    player.camera.lookAt = player.transform.t + player.camera.lookAtOffset;
    player.camera.position = player.camera.lookAt + (float3(0, 0, 1) * player.camera.distance);
    if (!keepPitchYaw) {
        player.camera.pitchYaw = float2(0, 0);
    }
}

void playerCameraSetPitchYaw(float2 pitchYawNew) {
    player.camera.pitchYaw.x = std::clamp(pitchYawNew.x, -pi * 0.4f, pi * 0.09f);
    player.camera.pitchYaw.y = std::remainderf(pitchYawNew.y, pi * 2.0f);
    XMVector quaternion = XMQuaternionRotationRollPitchYaw(player.camera.pitchYaw.x, player.camera.pitchYaw.y, 0);
    float3 dir = float3(XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion)).normalize();
    player.camera.position = player.camera.lookAt + (dir * player.camera.distance);
}

void playerCameraTranslate(float3 translate) {
    player.camera.position += translate;
    player.camera.lookAt += translate;
}

#ifdef EDITOR
void editorCameraRotate(CameraEditor& editorCamera, float2 pitchYawDelta) {
    editorCamera.pitchYaw.x += pitchYawDelta.x;
    editorCamera.pitchYaw.y += pitchYawDelta.y;
    editorCamera.pitchYaw.x = std::clamp(editorCamera.pitchYaw.x, -pi * 0.4f, pi * 0.4f);
    editorCamera.pitchYaw.y = std::remainderf(editorCamera.pitchYaw.y, pi * 2.0f);
    XMVector quaternion = XMQuaternionRotationRollPitchYaw(editorCamera.pitchYaw.x, editorCamera.pitchYaw.y, 0);
    float3 dir = XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion);
    editorCamera.lookAt = editorCamera.position + dir;
}
void editorCameraTranslate(CameraEditor& editorCamera, float3 translate) {
    float3 dz = (editorCamera.lookAt - editorCamera.position).normalize();
    float3 dx = dz.cross(float3(0, 1, 0));
    float3 dy = dz.cross(float3(1, 0, 0));
    editorCamera.position += dx * translate.x;
    editorCamera.lookAt += dx * translate.x;
    editorCamera.position += dy * translate.y;
    editorCamera.lookAt += dy * translate.y;
    editorCamera.position += dz * translate.z;
    editorCamera.lookAt += dz * translate.z;
}
#endif

void operator>>(ryml::ConstNodeRef n, float2& v) { n[0] >> v.x, n[1] >> v.y; }
void operator>>(ryml::ConstNodeRef n, float3& v) { n[0] >> v.x, n[1] >> v.y, n[2] >> v.z; }
void operator>>(ryml::ConstNodeRef n, float4& v) { n[0] >> v.x, n[1] >> v.y, n[2] >> v.z, n[3] >> v.w; }
void operator>>(ryml::ConstNodeRef n, Transform& t) { n[0] >> t.s.x, n[1] >> t.s.y, n[2] >> t.s.z, n[3] >> t.r.x, n[4] >> t.r.y, n[5] >> t.r.z, n[6] >> t.r.w, n[7] >> t.t.x, n[8] >> t.t.y, n[9] >> t.t.z; }
void operator>>(ryml::ConstNodeRef n, PxTransform& t) { n[0] >> t.q.x, n[1] >> t.q.y, n[2] >> t.q.z, n[3] >> t.q.w, n[4] >> t.p.x, n[5] >> t.p.y, n[6] >> t.p.z; }

void operator<<(ryml::NodeRef n, float2 v) { n |= ryml::SEQ, n |= ryml::_WIP_STYLE_FLOW_SL, n.append_child() << v.x, n.append_child() << v.y; }
void operator<<(ryml::NodeRef n, float3 v) { n |= ryml::SEQ, n |= ryml::_WIP_STYLE_FLOW_SL, n.append_child() << v.x, n.append_child() << v.y, n.append_child() << v.z; }
void operator<<(ryml::NodeRef n, float4 v) { n |= ryml::SEQ, n |= ryml::_WIP_STYLE_FLOW_SL, n.append_child() << v.x, n.append_child() << v.y, n.append_child() << v.z, n.append_child() << v.w; }
void operator<<(ryml::NodeRef n, Transform t) { n |= ryml::SEQ, n |= ryml::_WIP_STYLE_FLOW_SL, n.append_child() << t.s.x, n.append_child() << t.s.y, n.append_child() << t.s.z, n.append_child() << t.r.x, n.append_child() << t.r.y, n.append_child() << t.r.z, n.append_child() << t.r.w, n.append_child() << t.t.x, n.append_child() << t.t.y, n.append_child() << t.t.z; }
void operator<<(ryml::NodeRef n, PxTransform t) { n |= ryml::SEQ, n |= ryml::_WIP_STYLE_FLOW_SL, n.append_child() << t.q.x, n.append_child() << t.q.y, n.append_child() << t.q.z, n.append_child() << t.q.w, n.append_child() << t.p.x, n.append_child() << t.p.y, n.append_child() << t.p.z; }

void worldInit() {
    if (!std::filesystem::exists(worldFilePath)) assert(false);
    std::string yamlStr = fileReadStr(worldFilePath);
    ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
    ryml::ConstNodeRef yamlRoot = yamlTree.rootref();

#ifdef EDITOR
    if (ryml::ConstNodeRef editorYaml = yamlRoot.find_child("editor"); editorYaml.valid()) {
        if (ryml::ConstNodeRef editorCameraYaml = editorYaml.find_child("camera"); editorCameraYaml.valid()) {
            editorCameraYaml["position"] >> editor.camera.position;
            editorCameraYaml["pitchYaw"] >> editor.camera.pitchYaw;
            editorCameraRotate(editor.camera, float2(0, 0));
            editorCameraYaml["moveSpeed"] >> editor.camera.moveSpeed;
            editor.camera.moveSpeed = std::clamp(editor.camera.moveSpeed, 0.0f, editor.camera.moveSpeedMax);
        }
        if (ryml::ConstNodeRef editorEditCameraYaml = editorYaml.find_child("editCamera"); editorEditCameraYaml.valid()) {
            editorEditCameraYaml["position"] >> editor.editCamera.position;
            editorEditCameraYaml["pitchYaw"] >> editor.editCamera.pitchYaw;
            editorCameraRotate(editor.editCamera, float2(0, 0));
            editorEditCameraYaml["moveSpeed"] >> editor.editCamera.moveSpeed;
            editor.editCamera.moveSpeed = std::clamp(editor.editCamera.moveSpeed, 0.0f, editor.editCamera.moveSpeedMax);
        }
    }
#endif
    {
        ryml::ConstNodeRef assetsYaml = yamlRoot["assets"];

        ryml::ConstNodeRef skyboxesYaml = assetsYaml["skyboxes"];
        d3dCmdListReset();
        d3d.stagingBuffer.size = 0;
        for (ryml::ConstNodeRef skyboxYaml : skyboxesYaml) {
            std::string file;
            skyboxYaml >> file;
            Skybox skybox;
            skybox.hdriTextureFilePath = file;
            skybox.hdriTexture = d3dCreateImageDDS(exeDir / skybox.hdriTextureFilePath, d3d.graphicsCmdList, L"SkyboxHDRI", D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            skyboxes.push_back(skybox);
        }
        d3dCmdListExecute();
        d3dSignalFence(&d3d.transferFence);
        d3dWaitFence(&d3d.transferFence);

        for (ryml::ConstNodeRef modelsYaml = assetsYaml["models"]; ryml::ConstNodeRef modelYaml : modelsYaml) {
            std::string modelFile;
            modelYaml["file"] >> modelFile;
            Model* model = modelLoad(modelFile);
        }
    }
    {
        ryml::ConstNodeRef skyboxYaml = yamlRoot["skybox"];
        std::string assetPath;
        skyboxYaml["asset"] >> assetPath;
        for (Skybox& sb : skyboxes) {
            if (sb.hdriTextureFilePath == assetPath) {
                skybox = &sb;
                break;
            }
        }
        assert(skybox);
    }
    {
        ryml::ConstNodeRef playerYaml = yamlRoot["player"];
        std::string file;
        playerYaml["asset"] >> file;
        player.modelInstance = modelInstanceInit(file);
        playerYaml["transform"] >> player.transformDefault;
        player.transform = player.transformDefault;
        player.transformPrevFrame = player.transform;
        player.state = PlayerStateIdle;
        playerYaml["walkSpeed"] >> player.walkSpeed;
        playerYaml["runSpeed"] >> player.runSpeed;
        playerYaml["idleAnimationIndex"] >> player.idleAnimationIndex;
        playerYaml["walkAnimationIndex"] >> player.walkAnimationIndex;
        playerYaml["runAnimationIndex"] >> player.runAnimationIndex;
        playerYaml["jumpAnimationIndex"] >> player.jumpAnimationIndex;
        player.modelInstance.animation = &player.modelInstance.model->animations[player.idleAnimationIndex];
        playerYaml["cameraLookAtOffset"] >> player.camera.lookAtOffset;
        playerYaml["cameraDistance"] >> player.camera.distance;
        playerCameraReset(false);
        PxCapsuleControllerDesc capsuleControllerDesc;
        playerYaml["capsuleRadius"] >> capsuleControllerDesc.radius;
        playerYaml["capsuleHeight"] >> capsuleControllerDesc.height;
        playerYaml["capsuleContactOffset"] >> capsuleControllerDesc.contactOffset;
        playerYaml["capsuleStepOffset"] >> capsuleControllerDesc.stepOffset;
        capsuleControllerDesc.slopeLimit = cosf(radian(35));
        capsuleControllerDesc.climbingMode = PxCapsuleClimbingMode::eCONSTRAINED;
        capsuleControllerDesc.material = pxPhysics->createMaterial(0.5f, 0.5f, 0.0f);
        assert(capsuleControllerDesc.isValid());
        player.pxController = (PxCapsuleController*)pxControllerManager->createController(capsuleControllerDesc);
        assert(player.pxController);
        assert(player.pxController->setFootPosition(player.transformDefault.t.toPxVec3d()));
    }
    for (ryml::ConstNodeRef gameObjectsYaml = yamlRoot["gameObjects"]; ryml::ConstNodeRef objYaml : gameObjectsYaml) {
        GameObject& obj = gameObjects.emplace_back();
        assert(gameObjects.size() < gameObjectsMaxCount);
        objYaml["name"] >> obj.name;
        std::string assetFile;
        objYaml["asset"] >> assetFile;
        obj.modelInstance = modelInstanceInit(assetFile);
        objYaml["transform"] >> obj.transformDefault;
        obj.transform = obj.transformDefault;
        obj.transformPrevFrame = obj.transform;
        if (ryml::ConstNodeRef rigidActorYaml = objYaml.find_child("rigidActor"); rigidActorYaml.valid()) {
            if (rigidActorYaml["type"] == "static") {
                obj.rigidActor = pxPhysics->createRigidStatic(obj.transform.toPxTransform());
                pxScene->addActor(*obj.rigidActor);
            }
            else if (rigidActorYaml["type"] == "dynamic") {
                obj.rigidActor = pxPhysics->createRigidDynamic(obj.transform.toPxTransform());
                float mass;
                rigidActorYaml["mass"] >> mass;
                ((PxRigidDynamic*)obj.rigidActor)->setMass(mass);
                pxScene->addActor(*obj.rigidActor);
            }
            else {
                assert(false);
            }
            if (ryml::ConstNodeRef shapesYaml = rigidActorYaml.find_child("shapes"); shapesYaml.valid()) {
                for (ryml::ConstNodeRef shapeYaml : shapesYaml) {
                    PxTransform transform(PxIdentity);
                    if (shapeYaml["type"] == "plane") {
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxPlaneGeometry(), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "box") {
                        float3 halfExtents;
                        shapeYaml["halfExtents"] >> halfExtents;
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxBoxGeometry(halfExtents.x, halfExtents.y, halfExtents.z), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "sphere") {
                        float radius;
                        shapeYaml["radius"] >> radius;
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxSphereGeometry(radius), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "capsule") {
                        float radius, halfHeight;
                        shapeYaml["radius"] >> radius;
                        shapeYaml["halfHeight"] >> halfHeight;
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCapsuleGeometry(radius, halfHeight), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "cylinder") {
                        float radius, height;
                        shapeYaml["radius"] >> radius;
                        shapeYaml["height"] >> height;
                        PxCustomGeometryExt::CylinderCallbacks* cylinder = new PxCustomGeometryExt::CylinderCallbacks(height, radius, 0, 0);
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCustomGeometry(*cylinder), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "cone") {
                        float radius, height;
                        shapeYaml["radius"] >> radius;
                        shapeYaml["height"] >> height;
                        PxCustomGeometryExt::ConeCallbacks* cone = new PxCustomGeometryExt::ConeCallbacks(height, radius, 0, 0);
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCustomGeometry(*cone), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "convexMesh") {
                        if (!obj.modelInstance.model->convexMesh) {
                            assert(modelGenerateConvexMesh(obj.modelInstance.model));
                        }
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxConvexMeshGeometry(obj.modelInstance.model->convexMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else if (shapeYaml["type"] == "triangleMesh") {
                        if (!obj.modelInstance.model->triangleMesh) {
                            assert(modelGenerateTriangleMesh(obj.modelInstance.model));
                        }
                        PxShape* shape = PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxTriangleMeshGeometry(obj.modelInstance.model->triangleMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial);
                        shapeYaml["transform"] >> transform;
                        shape->setLocalPose(transform);
                    }
                    else {
                        assert(false);
                    }
                }
            }
#ifdef EDITOR
            obj.rigidActor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, true);
#endif
        }
    }
}

void editorSave() {
#ifdef EDITOR
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;

    ryml::NodeRef editorYaml = yamlRoot["editor"];
    editorYaml |= ryml::MAP;
    ryml::NodeRef editorCameraYaml = editorYaml["camera"];
    editorCameraYaml |= ryml::MAP;
    editorCameraYaml["position"] << editor.camera.position;
    editorCameraYaml["pitchYaw"] << editor.camera.pitchYaw;
    editorCameraYaml["moveSpeed"] << editor.camera.moveSpeed;

    ryml::NodeRef editorEditCameraYaml = editorYaml["editCamera"];
    editorEditCameraYaml |= ryml::MAP;
    editorEditCameraYaml["position"] << editor.editCamera.position;
    editorEditCameraYaml["pitchYaw"] << editor.editCamera.pitchYaw;
    editorEditCameraYaml["moveSpeed"] << editor.editCamera.moveSpeed;

    ryml::NodeRef assetsYaml = yamlRoot["assets"];
    assetsYaml |= ryml::MAP;
    ryml::NodeRef skyboxesYaml = assetsYaml["skyboxes"];
    skyboxesYaml |= ryml::SEQ;
    for (Skybox& skybox : skyboxes) {
        skyboxesYaml.append_child() << skybox.hdriTextureFilePath.string();
    }
    ryml::NodeRef modelsYaml = assetsYaml["models"];
    modelsYaml |= ryml::SEQ;
    for (Model& model : models) {
        ryml::NodeRef modelYaml = modelsYaml.append_child();
        modelYaml |= ryml::MAP;
        modelYaml["file"] << model.filePath.string();
    }

    ryml::NodeRef skyboxYaml = yamlRoot["skybox"];
    skyboxYaml |= ryml::MAP;
    skyboxYaml["asset"] << skybox->hdriTextureFilePath.string();

    ryml::NodeRef playerYaml = yamlRoot["player"];
    playerYaml |= ryml::MAP;
    playerYaml["asset"] << player.modelInstance.model->filePath.string();
    playerYaml["transform"] << player.transformDefault;
    playerYaml["walkSpeed"] << player.walkSpeed;
    playerYaml["runSpeed"] << player.runSpeed;
    playerYaml["idleAnimationIndex"] << player.idleAnimationIndex;
    playerYaml["walkAnimationIndex"] << player.walkAnimationIndex;
    playerYaml["runAnimationIndex"] << player.runAnimationIndex;
    playerYaml["jumpAnimationIndex"] << player.jumpAnimationIndex;
    playerYaml["cameraLookAtOffset"] << player.camera.lookAtOffset;
    playerYaml["cameraDistance"] << player.camera.distance;
    playerYaml["capsuleRadius"] << player.pxController->getRadius();
    playerYaml["capsuleHeight"] << player.pxController->getHeight();
    playerYaml["capsuleContactOffset"] << player.pxController->getContactOffset();
    playerYaml["capsuleStepOffset"] << player.pxController->getStepOffset();

    ryml::NodeRef gameObjectsYaml = yamlRoot["gameObjects"];
    gameObjectsYaml |= ryml::SEQ;
    for (GameObject& obj : gameObjects) {
        ryml::NodeRef gameObjectYaml = gameObjectsYaml.append_child();
        gameObjectYaml |= ryml::MAP;
        gameObjectYaml["name"] << obj.name;
        gameObjectYaml["asset"] << obj.modelInstance.model->filePath.string();
        gameObjectYaml["transform"] << Transform(obj.transformDefault);
        if (obj.rigidActor) {
            ryml::NodeRef rigidActorYaml = gameObjectYaml["rigidActor"];
            rigidActorYaml |= ryml::MAP;
            PxActorType::Enum actorType = obj.rigidActor->getType();
            if (actorType == PxActorType::eRIGID_STATIC) {
                rigidActorYaml["type"] << "static";
            }
            else if (actorType == PxActorType::eRIGID_DYNAMIC) {
                rigidActorYaml["type"] << "dynamic";
                rigidActorYaml["mass"] << ((PxRigidDynamic*)obj.rigidActor)->getMass();
            }
            else {
                assert(false);
            }
            uint shapeCount = obj.rigidActor->getNbShapes();
            if (shapeCount > 0) {
                static std::vector<PxShape*> shapes;
                shapes.resize(shapeCount);
                obj.rigidActor->getShapes(shapes.data(), shapeCount);
                ryml::NodeRef shapesYaml = rigidActorYaml["shapes"];
                shapesYaml |= ryml::SEQ, shapesYaml |= ryml::_WIP_STYLE_FLOW_SL;
                for (PxShape* shape : shapes) {
                    ryml::NodeRef shapeYaml = shapesYaml.append_child();
                    shapeYaml |= ryml::MAP;
                    const PxGeometry& geometry = shape->getGeometry();
                    PxGeometryType::Enum geometryType = geometry.getType();
                    if (geometryType == PxGeometryType::ePLANE) {
                        shapeYaml["type"] = "plane";
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eBOX) {
                        shapeYaml["type"] = "box";
                        shapeYaml["halfExtents"] << float3(((const PxBoxGeometry&)geometry).halfExtents);
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eSPHERE) {
                        shapeYaml["type"] = "sphere";
                        shapeYaml["radius"] << ((const PxSphereGeometry&)geometry).radius;
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eCAPSULE) {
                        shapeYaml["type"] = "capsule";
                        shapeYaml["radius"] << ((const PxCapsuleGeometry&)geometry).radius;
                        shapeYaml["halfHeight"] << ((const PxCapsuleGeometry&)geometry).halfHeight;
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eCONVEXMESH) {
                        shapeYaml["type"] = "convexMesh";
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eTRIANGLEMESH) {
                        shapeYaml["type"] = "triangleMesh";
                        shapeYaml["transform"] << shape->getLocalPose();
                    }
                    else if (geometryType == PxGeometryType::eCUSTOM) {
                        PxCustomGeometry customGeometry = (const PxCustomGeometry&)geometry;
                        if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::CylinderCallbacks::TYPE()) {
                            PxCustomGeometryExt::CylinderCallbacks* cylinderCallback = (PxCustomGeometryExt::CylinderCallbacks*)customGeometry.callbacks;
                            shapeYaml["type"] = "cylinder";
                            shapeYaml["radius"] << cylinderCallback->getRadius();
                            shapeYaml["height"] << cylinderCallback->getHeight();
                            shapeYaml["transform"] << shape->getLocalPose();
                        }
                        else if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::ConeCallbacks::TYPE()) {
                            PxCustomGeometryExt::ConeCallbacks* coneCallback = (PxCustomGeometryExt::ConeCallbacks*)customGeometry.callbacks;
                            shapeYaml["type"] = "cone";
                            shapeYaml["radius"] << coneCallback->getRadius();
                            shapeYaml["height"] << coneCallback->getHeight();
                            shapeYaml["transform"] << shape->getLocalPose();
                        }
                        else {
                            assert(false);
                        }
                    }
                    else {
                        assert(false);
                    }
                    shape->release();
                }
            }
        }
    }
    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(worldFilePath, yamlStr);
#endif
}

void gameReadSave() {
#ifndef EDITOR
    if (!std::filesystem::exists(path)) return;

    std::string yamlStr = fileReadStr(path);
    ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
    ryml::ConstNodeRef yamlRoot = yamlTree.rootref();

    ryml::ConstNodeRef playerYaml = yamlRoot["player"];
    playerYaml["transform"] >> player.transform;
#endif
}

void gameSave() {
#ifndef EDITOR
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;

    ryml::NodeRef playerYaml = yamlRoot["player"];
    playerYaml |= ryml::MAP;
    playerYaml["transform"] << player.transform;

    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(gameSavePath, yamlStr);
#endif
}

void gameObjectRelease(GameObject& obj) {
    modelInstanceRelease(obj.modelInstance);
    if (obj.rigidActor) {
        obj.rigidActor->release();
    }
}

void setRigidActorsVisualization(bool b) {
    player.pxController->getActor()->setActorFlag(PxActorFlag::eVISUALIZATION, b);
    for (GameObject& obj : gameObjects) {
        if (obj.rigidActor) {
            obj.rigidActor->setActorFlag(PxActorFlag::eVISUALIZATION, b);
        }
    }
}

#ifdef EDITOR
void toggleBetweenEditorPlay() {
    editorActive = !editorActive;
    windowHideCursor(!editorActive);
    player.transform = player.transformDefault;
    player.pxController->setFootPosition(player.transform.t.toPxVec3d());
    playerCameraReset(true);
    for (GameObject& obj : gameObjects) {
        obj.transform = obj.transformDefault;
        if (obj.rigidActor) {
            obj.rigidActor->setGlobalPose(obj.transform.toPxTransform());
            obj.rigidActor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, editorActive);
            if (!editorActive && obj.rigidActor->getType() == PxActorType::eRIGID_DYNAMIC) {
                ((PxRigidDynamic*)obj.rigidActor)->wakeUp();
            }
        }
    }
}

void editorMainMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save")) {
                editorSave();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit")) { quit = true; }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Display")) {
            if (ImGui::MenuItem(hdr ? "HDR (On)" : "HDR (Off)")) {
                hdr = !hdr;
                d3dApplySettings();
            }
            else if (ImGui::MenuItem("Windowed")) {
                windowMode = WindowModeWindowed;
                d3dApplySettings();
            }
            else if (ImGui::MenuItem("Borderless")) {
                windowMode = WindowModeBorderless;
                d3dApplySettings();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::BeginMenu("Camera")) {
                ImGui::SliderFloat("speed(meters)", &editor.camera.moveSpeed, 0, editor.camera.moveSpeedMax);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("EditObjectCamera")) {
                ImGui::SliderFloat("speed(meters)", &editor.editCamera.moveSpeed, 0, editor.editCamera.moveSpeedMax);
                ImGui::EndMenu();
            }
            ImGui::Separator();
            ImGui::Checkbox("XRay", &editor.selectedObjectXRay);
            ImGui::Checkbox("RigidActors", &showRigidActorsGeometries);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("render")) {
            if (ImGui::Checkbox("pathTracer", &pathTracer)) {
                d3d.pathTracerAccumulationCount = 0;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void editorObjectsWindow() {
    if (ImGui::Begin("Objects")) {
        if (ImGui::Selectable("Player", editor.selectedObjectType == ObjectTypePlayer)) {
            editor.selectedObjectType = ObjectTypePlayer;
            editor.selectedObjectIndex = 0;
        }
        if (editor.selectedObjectType == ObjectTypePlayer && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("player edit");
        }
        if (editor.selectedObjectType == ObjectTypePlayer && ImGui::BeginPopup("player edit")) {
            ImGui::EndPopup();
        }
        bool gameObjectTreeOpen = ImGui::TreeNode("Game Objects");
        bool newGameObjectPopup = false;
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::Button("New Game Object")) {
                newGameObjectPopup = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (newGameObjectPopup) {
            ImGui::OpenPopup("New Game Object##NewGameObjectPopup");
        }
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("New Game Object##NewGameObjectPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            static char objectName[32] = {};
            static int modelIndex = -1;
            auto getModelFileName = [](void* userData, int index) {
                return models[index].filePathStr.c_str();
            };
            auto nameUnique = []() {
                for (GameObject& object : gameObjects) {
                    if (object.name == objectName) {
                        return false;
                    }
                }
                return true;
            };
            ImGui::InputText("Name", objectName, sizeof(objectName));
            ImGui::Combo("Models", &modelIndex, getModelFileName, nullptr, (int)models.size());
            if (ImGui::Button("Ok") || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                if (objectName[0] == '\0') {
                    ImGui::DebugLog("error: object name is empty\n");
                }
                else if (!nameUnique()) {
                    ImGui::DebugLog("error: object name \"%s\" already exists\n", objectName);
                }
                else if (modelIndex >= models.size() || modelIndex < 0) {
                    ImGui::DebugLog("error: invalid modelIndex %d\n", modelIndex);
                }
                else {
                    GameObject gameObject = {.name = objectName, .modelInstance = modelInstanceInit(&models[modelIndex])};
                    gameObjects.push_back(std::move(gameObject));
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (gameObjectTreeOpen) {
            for (uint objIndex = 0; objIndex < gameObjects.size(); objIndex++) {
                GameObject& obj = gameObjects[objIndex];
                ImGui::PushID(objIndex);
                if (objIndex == editor.renameObjectIndex) {
                    static char objectName[32] = {'\0'};
                    ImGui::SetKeyboardFocusHere();
                    ImGui::InputText("##newObjectName", objectName, sizeof(objectName));
                    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                        objectName[0] = '\0';
                        editor.renameObjectIndex = UINT_MAX;
                    }
                    else if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                        auto uniqueName = [](GameObject& obj) {
                            for (GameObject& object : gameObjects) {
                                if (object.name == objectName && &object != &obj) {
                                    return false;
                                }
                            }
                            return true;
                        };
                        if (objectName[0] == '\0') {
                            ImGui::DebugLog("error: object name empty\n");
                        }
                        else if (!uniqueName(obj)) {
                            ImGui::DebugLog("error: object name \"%s\" already exists\n", objectName);
                        }
                        else {
                            obj.name = objectName;
                            objectName[0] = '\0';
                            editor.renameObjectIndex = UINT_MAX;
                        }
                    }
                }
                else {
                    if (ImGui::Selectable(obj.name.c_str(), editor.selectedObjectType == ObjectTypeGameObject && editor.selectedObjectIndex == objIndex, ImGuiSelectableFlags_AllowDoubleClick)) {
                        editor.selectedObjectType = ObjectTypeGameObject;
                        editor.selectedObjectIndex = objIndex;
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                            editor.renameObjectIndex = objIndex;
                        }
                    }
                    if (ImGui::BeginPopupContextItem()) {
                        if (ImGui::Button("Rename")) {
                            editor.renameObjectIndex = objIndex;
                            ImGui::CloseCurrentPopup();
                        }
                        if (ImGui::Button("Delete")) {
                            obj.toBeDeleted = true;
                            editor.selectedObjectType = ObjectTypeNone;
                            ImGui::CloseCurrentPopup();
                        }
                        ImGui::EndPopup();
                    }
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

bool transformImGui(Transform& transform, bool hasT, bool hasR, bool hasS) {
    bool change = false;
    ImGui::Text("Transform");
    if (hasT) {
        if (ImGui::InputFloat3("T", &transform.t.x)) {
            change = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("reset##t")) {
            transform.t = float3(0, 0, 0);
            change = true;
        }
    }
    if (hasR) {
        float3 angles = quaternionToEulerAngles(transform.r);
        angles.x = degree(angles.x), angles.y = degree(angles.y), angles.z = degree(angles.z);
        if (ImGui::InputFloat3("R", &angles.x)) {
            angles.x = radian(angles.x), angles.y = radian(angles.y), angles.z = radian(angles.z);
            transform.r = quaternionFromEulerAngles(angles);
            change = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("reset##r")) {
            transform.r = float4(0, 0, 0, 1);
            change = true;
        }
    }
    if (hasS) {
        if (ImGui::InputFloat3("S", &transform.s.x)) {
            change = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("reset##s")) {
            transform.s = float3(1, 1, 1);
            change = true;
        }
    }
    return change;
}

bool transformImGui(PxTransform& pxTransform) {
    Transform transform(pxTransform);
    if (transformImGui(transform, true, true, false)) {
        PxTransform pt = transform.toPxTransform();
        if (pt.isValid()) {
            pxTransform = pt;
            return true;
        }
    }
    return false;
}

void editorPropertiesPlayer() {
    ImGui::Text("Player");
    modelInstanceImGui(&player.modelInstance);
    if (ImGui::TreeNode("Spawn")) {
        if (transformImGui(player.transformDefault, true, true, true)) {
            player.transform = player.transformDefault;
            player.pxController->setFootPosition(player.transform.t.toPxVec3d());
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Locomotion")) {
        ImGui::InputFloat("walkSpeed", &player.walkSpeed);
        ImGui::InputFloat("runSpeed", &player.runSpeed);
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("CapsuleController")) {
        float radius = player.pxController->getRadius();
        float height = player.pxController->getHeight();
        float contactOffset = player.pxController->getContactOffset();
        float stepOffset = player.pxController->getStepOffset();
        if (ImGui::InputFloat("radius", &radius)) {
            radius = std::clamp(radius, 0.0f, 10.0f);
            player.pxController->setRadius(radius);
            player.pxController->setFootPosition(player.transformDefault.t.toPxVec3d());
        }
        if (ImGui::InputFloat("height", &height)) {
            height = std::clamp(height, 0.0f, 10.0f);
            player.pxController->setHeight(height);
            player.pxController->setFootPosition(player.transformDefault.t.toPxVec3d());
        }
        if (ImGui::InputFloat("contactOffset", &contactOffset)) {
            contactOffset = std::clamp(contactOffset, 0.0f, 1.0f);
            player.pxController->setContactOffset(contactOffset);
            player.pxController->setFootPosition(player.transformDefault.t.toPxVec3d());
        }
        if (ImGui::InputFloat("stepOffset", &stepOffset)) {
            stepOffset = std::clamp(stepOffset, 0.0f, 1.0f);
            player.pxController->setStepOffset(stepOffset);
            player.pxController->setFootPosition(player.transformDefault.t.toPxVec3d());
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Animations")) {
        Model* model = player.modelInstance.model;
        if (model->animations.size() > 0) {
            auto selectAnimation = [model](const char* label, uint* index) {
                if (ImGui::BeginCombo(label, model->animations[*index].name.c_str())) {
                    for (uint animationIndex = 0; animationIndex < model->animations.size(); animationIndex++) {
                        if (ImGui::Selectable(model->animations[animationIndex].name.c_str())) {
                            *index = animationIndex;
                        }
                    }
                    ImGui::EndCombo();
                }
            };
            selectAnimation("idle", &player.idleAnimationIndex);
            selectAnimation("walk", &player.walkAnimationIndex);
            selectAnimation("run", &player.runAnimationIndex);
            selectAnimation("jump", &player.jumpAnimationIndex);
        }
        ImGui::TreePop();
    }
}

void editorUpdateGameObjectTransformDefault(GameObject& obj, Transform t) {
    obj.transformDefault = t;
    obj.transform = t;
    if (obj.rigidActor) {
        obj.rigidActor->setGlobalPose(t.toPxTransform());
        static std::vector<PxShape*> shapes;
        shapes.resize(obj.rigidActor->getNbShapes());
        obj.rigidActor->getShapes(shapes.data(), (uint)shapes.size());
        for (PxShape* shape : shapes) {
            const PxGeometry& geometry = shape->getGeometry();
            if (geometry.getType() == PxGeometryType::eCONVEXMESH) {
                shape->setGeometry(PxConvexMeshGeometry(((const PxConvexMeshGeometry&)geometry).convexMesh, PxMeshScale(t.s.toPxVec3())));
                break;
            }
        }
    }
}

void editorPropertiesGameObject() {
    GameObject& obj = gameObjects[editor.selectedObjectIndex];
    ImGui::Text("GameObject \"%s\"", obj.name.c_str());
    modelInstanceImGui(&obj.modelInstance);
    if (ImGui::TreeNode("Spawn")) {
        Transform transform = obj.transformDefault;
        if (transformImGui(transform, true, true, true)) {
            editorUpdateGameObjectTransformDefault(obj, transform);
        }
        ImGui::TreePop();
    }
    bool rigidActorTreeOpen = ImGui::TreeNode("RigidActor");
    if (ImGui::BeginPopupContextItem()) {
        if (obj.rigidActor) {
            if (ImGui::Button("delete")) {
                obj.rigidActor->release();
                obj.rigidActor = nullptr;
                ImGui::CloseCurrentPopup();
            }
        }
        else {
            if (ImGui::Button("new rigid static")) {
                obj.rigidActor = pxPhysics->createRigidStatic(obj.transform.toPxTransform());
                pxScene->addActor(*obj.rigidActor);
                obj.rigidActor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, true);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::Button("new rigid dynamic")) {
                obj.rigidActor = pxPhysics->createRigidDynamic(obj.transform.toPxTransform());
                pxScene->addActor(*obj.rigidActor);
                obj.rigidActor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, true);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    if (rigidActorTreeOpen) {
        if (obj.rigidActor) {
            PxActorType::Enum actorType = obj.rigidActor->getType();
            if (actorType == PxActorType::eRIGID_STATIC) {
                ImGui::Text("Type: Static");
            }
            else if (actorType == PxActorType::eRIGID_DYNAMIC) {
                ImGui::Text("Type: Dyanamic");
                float mass = ((PxRigidDynamic*)obj.rigidActor)->getMass();
                if (ImGui::InputFloat("mass", &mass)) {
                    ((PxRigidDynamic*)obj.rigidActor)->setMass(mass);
                }
            }
            else {
                assert(false);
            }
            bool shapesNodeOpen = ImGui::TreeNode("shapes");
            static std::vector<PxShape*> shapes;
            shapes.resize(obj.rigidActor->getNbShapes());
            obj.rigidActor->getShapes(shapes.data(), (uint)shapes.size());
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::Button("new plane")) {
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxPlaneGeometry(), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new box")) {
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxBoxGeometry(1, 1, 1), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new sphere")) {
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxSphereGeometry(1), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new capsule")) {
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCapsuleGeometry(1, 1), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new cylinder")) {
                    PxCustomGeometryExt::CylinderCallbacks* cylinder = new PxCustomGeometryExt::CylinderCallbacks(1, 0.5f, 0, 0);
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCustomGeometry(*cylinder), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new cone")) {
                    PxCustomGeometryExt::ConeCallbacks* cone = new PxCustomGeometryExt::ConeCallbacks(1, 1, 0, 0);
                    assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxCustomGeometry(*cone), *pxDefaultMaterial));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::Button("new convex mesh")) {
                    bool hasConvexMesh = false;
                    for (PxShape* shape : shapes) {
                        if (const PxGeometry& geometry = shape->getGeometry(); geometry.getType() == PxGeometryType::eCONVEXMESH) {
                            hasConvexMesh = true;
                            break;
                        }
                    }
                    if (hasConvexMesh) {
                        ImGui::DebugLog("a convex mesh already exists\n");
                    }
                    else {
                        if (obj.modelInstance.model->convexMesh) {
                            assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxConvexMeshGeometry(obj.modelInstance.model->convexMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial));
                        }
                        else {
                            if (modelGenerateConvexMesh(obj.modelInstance.model)) {
                                assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxConvexMeshGeometry(obj.modelInstance.model->convexMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial));
                            }
                            else {
                                ImGui::DebugLog("failed to generate convex mesh\n");
                            }
                        }
                    }
                    ImGui::CloseCurrentPopup();
                }
                if (actorType == PxActorType::eRIGID_STATIC && ImGui::Button("new triangle mesh")) {
                    bool hasTriangleMesh = false;
                    for (PxShape* shape : shapes) {
                        if (const PxGeometry& geometry = shape->getGeometry(); geometry.getType() == PxGeometryType::eTRIANGLEMESH) {
                            hasTriangleMesh = true;
                            break;
                        }
                    }
                    if (hasTriangleMesh) {
                        ImGui::DebugLog("a triangle mesh already exists\n");
                    }
                    else {
                        if (obj.modelInstance.model->triangleMesh) {
                            assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxTriangleMeshGeometry(obj.modelInstance.model->triangleMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial));
                        }
                        else {
                            if (modelGenerateTriangleMesh(obj.modelInstance.model)) {
                                assert(PxRigidActorExt::createExclusiveShape(*obj.rigidActor, PxTriangleMeshGeometry(obj.modelInstance.model->triangleMesh, PxMeshScale(obj.transform.s.toPxVec3())), *pxDefaultMaterial));
                            }
                            else {
                                ImGui::DebugLog("failed to generate a triangle mesh\n");
                            }
                        }
                    }
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            if (shapesNodeOpen) {
                PxShape* shapeToDelete = nullptr;
                for (uint shapeIndex = 0; shapeIndex < shapes.size(); shapeIndex++) {
                    PxShape* shape = shapes[shapeIndex];
                    const PxGeometry& geometry = shape->getGeometry();
                    PxGeometryType::Enum geometryType = geometry.getType();
                    ImGui::PushID(shapeIndex);
                    bool open = false;
                    if (geometryType == PxGeometryType::ePLANE) {
                        open = ImGui::TreeNode("plane");
                    }
                    else if (geometryType == PxGeometryType::eBOX) {
                        open = ImGui::TreeNode("box");
                    }
                    else if (geometryType == PxGeometryType::eSPHERE) {
                        open = ImGui::TreeNode("sphere");
                    }
                    else if (geometryType == PxGeometryType::eCAPSULE) {
                        open = ImGui::TreeNode("capsule");
                    }
                    else if (geometryType == PxGeometryType::eCUSTOM) {
                        PxCustomGeometry customGeometry = (const PxCustomGeometry&)geometry;
                        if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::CylinderCallbacks::TYPE()) {
                            open = ImGui::TreeNode("cylinder");
                        }
                        else if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::ConeCallbacks::TYPE()) {
                            open = ImGui::TreeNode("cone");
                        }
                    }
                    else if (geometryType == PxGeometryType::eCONVEXMESH) {
                        open = ImGui::TreeNode("convex mesh");
                    }
                    else if (geometryType == PxGeometryType::eTRIANGLEMESH) {
                        open = ImGui::TreeNode("triangle mesh");
                    }
                    else {
                        assert(false);
                    }
                    if (ImGui::BeginPopupContextItem()) {
                        if (ImGui::Button("delete")) {
                            shapeToDelete = shape;
                            ImGui::CloseCurrentPopup();
                        }
                        ImGui::EndPopup();
                    }
                    if (open) {
                        PxTransform pxTransform = shape->getLocalPose();
                        if (geometryType == PxGeometryType::ePLANE) {
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else if (geometryType == PxGeometryType::eBOX) {
                            PxBoxGeometry box = (const PxBoxGeometry&)geometry;
                            if (ImGui::InputFloat3("extents", &box.halfExtents.x) && box.isValid()) {
                                shape->setGeometry(box);
                            }
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else if (geometryType == PxGeometryType::eSPHERE) {
                            PxSphereGeometry sphere = (const PxSphereGeometry&)geometry;
                            if (ImGui::InputFloat("radius", &sphere.radius) && sphere.isValid()) {
                                shape->setGeometry(sphere);
                            }
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else if (geometryType == PxGeometryType::eCAPSULE) {
                            PxCapsuleGeometry capsule = (const PxCapsuleGeometry&)geometry;
                            bool b0 = ImGui::InputFloat("radius", &capsule.radius) && capsule.isValid();
                            bool b1 = ImGui::InputFloat("halfHeight", &capsule.halfHeight) && capsule.isValid();
                            if (b0 || b1) {
                                shape->setGeometry(capsule);
                            }
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else if (geometryType == PxGeometryType::eCUSTOM) {
                            PxCustomGeometry customGeometry = (const PxCustomGeometry&)geometry;
                            if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::CylinderCallbacks::TYPE()) {
                                PxCustomGeometryExt::CylinderCallbacks* cylinderCallback = (PxCustomGeometryExt::CylinderCallbacks*)customGeometry.callbacks;
                                float radius = cylinderCallback->getRadius();
                                float height = cylinderCallback->getHeight();
                                bool b0 = ImGui::InputFloat("radius", &radius);
                                bool b1 = ImGui::InputFloat("height", &height);
                                if (b0 || b1) {
                                    cylinderCallback->setRadius(radius);
                                    cylinderCallback->setHeight(height);
                                    shape->setGeometry(PxCustomGeometry(*cylinderCallback));
                                }
                                if (transformImGui(pxTransform)) {
                                    shape->setLocalPose(pxTransform);
                                }
                            }
                            else if (customGeometry.callbacks->getCustomType() == PxCustomGeometryExt::ConeCallbacks::TYPE()) {
                                PxCustomGeometryExt::ConeCallbacks* coneCallback = (PxCustomGeometryExt::ConeCallbacks*)customGeometry.callbacks;
                                float radius = coneCallback->getRadius();
                                float height = coneCallback->getHeight();
                                bool b0 = ImGui::InputFloat("radius", &radius);
                                bool b1 = ImGui::InputFloat("height", &height);
                                if (b0 || b1) {
                                    coneCallback->setRadius(radius);
                                    coneCallback->setHeight(height);
                                    shape->setGeometry(PxCustomGeometry(*coneCallback));
                                }
                                if (transformImGui(pxTransform)) {
                                    shape->setLocalPose(pxTransform);
                                }
                            }
                            else {
                                assert(false);
                            }
                        }
                        else if (geometryType == PxGeometryType::eCONVEXMESH) {
                            PxConvexMeshGeometry convexMesh = (const PxConvexMeshGeometry&)geometry;
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else if (geometryType == PxGeometryType::eTRIANGLEMESH) {
                            PxTriangleMeshGeometry triangleMesh = (const PxTriangleMeshGeometry&)geometry;
                            if (transformImGui(pxTransform)) {
                                shape->setLocalPose(pxTransform);
                            }
                        }
                        else {
                            assert(false);
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
                if (shapeToDelete) {
                    obj.rigidActor->detachShape(*shapeToDelete);
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
}

void editorPropertiesWindow() {
    if (ImGui::Begin("Properties")) {
        if (editor.selectedObjectType == ObjectTypePlayer) {
            editorPropertiesPlayer();
        }
        else if (editor.selectedObjectType == ObjectTypeGameObject) {
            editorPropertiesGameObject();
        }
    }
    ImGui::End();
}

void editorAssetsWindow() {
    if (ImGui::Begin("Assets")) {
        bool modelsTreeOpen = ImGui::TreeNode("Models");
        bool newModelPopup = false;
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::Button("New Model")) {
                newModelPopup = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (newModelPopup) {
            ImGui::OpenPopup("NewModelPopup");
        }
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("NewModelPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            static char filePath[256] = {};
            ImGui::InputText("File", filePath, sizeof(filePath)), ImGui::SameLine();
            if (ImGui::Button("Browse")) {
                OPENFILENAMEA openfileName = {.lStructSize = sizeof(OPENFILENAMEA), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                GetOpenFileNameA(&openfileName);
            }
            if (ImGui::Button("Okay") || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (modelsTreeOpen) {
            for (uint modelIndex = 0; modelIndex < models.size(); modelIndex++) {
                ImGui::PushID(modelIndex);
                if (ImGui::Selectable(models[modelIndex].filePath.string().c_str(), editor.selectedModelIndex == modelIndex)) {
                    editor.selectedModelIndex = modelIndex;
                }
                if (editor.mode == EditorModeFreeCam) {
                    if (ImGui::BeginDragDropSource()) {
                        ImGui::SetDragDropPayload("dragDropTypeModelIndex", (void*)&modelIndex, sizeof(modelIndex));
                        ImGui::EndDragDropSource();
                    }
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        bool skyBoxesTreeOpen = ImGui::TreeNode("Skyboxes");
        bool newSkyboxPopup = false;
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::Button("New Skybox")) {
                newSkyboxPopup = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (newSkyboxPopup) {
            ImGui::OpenPopup("NewSkyboxPopup");
        }
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("NewSkyboxPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            static char filePath[256] = {};
            ImGui::InputText("File", filePath, sizeof(filePath)), ImGui::SameLine();
            if (ImGui::Button("Browse")) {
                OPENFILENAMEA openfileName = {.lStructSize = sizeof(OPENFILENAMEA), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                GetOpenFileNameA(&openfileName);
            }
            if (ImGui::Button("Okay") || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (skyBoxesTreeOpen) {
            for (Skybox& skybox : skyboxes) {
                ImGui::Text("%s\n", skybox.hdriTextureFilePath.string().c_str());
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload(); payload) {
        if (payload->IsDataType("dragDropTypeModelIndex")) {
            uint modelIndex = *(uint*)payload->Data;
            auto generateNewGameObjectName = []() {
                static int n = 0;
                while (true) {
                    std::string name = std::format("gameObject{}", n++);
                    bool unique = true;
                    for (GameObject& obj : gameObjects) {
                        if (obj.name == name) {
                            unique = false;
                            break;
                        }
                    }
                    if (unique) {
                        return name;
                    }
                }
            };
            if (!editor.beginDragDropGameObject) {
                editor.beginDragDropGameObject = true;
                editor.dragDropGameObject = GameObject{.name = generateNewGameObjectName(), .modelInstance = modelInstanceInit(&models[modelIndex])};
            }
            float3 pos = XMVector3Unproject(XMVectorSet(ImGui::GetMousePos().x, ImGui::GetMousePos().y, 0, 0), 0, 0, (float)renderW, (float)renderH, 0.0f, 1.0f, cameraProjectMat, cameraViewMat, xmMatrixIdentity);
            float3 dir = (pos - editor.camera.position).normalize();
            editor.dragDropGameObject.transformDefault.t = editor.camera.position + (dir * 10.0f);
            editor.dragDropGameObject.transform = editor.dragDropGameObject.transformDefault;
        }
    }
    else if (editor.beginDragDropGameObject) {
        editor.beginDragDropGameObject = false;
        gameObjects.push_back(editor.dragDropGameObject);
    }
}

void editorUpdate() {
    ZoneScopedN("editorUpdate");
    if (mouseSelectOngoing && d3dTryWaitFence(&d3d.collisionQueriesFence)) {
        mouseSelectOngoing = false;
        CollisionQueryResult collisionQueryResult = ((CollisionQueryResult*)d3d.collisionQueryResultsBuffer.ptr)[0];
        if (editor.selectedObjectType == collisionQueryResult.objectType && editor.selectedObjectIndex == collisionQueryResult.objectIndex) {
            editor.selectedObjectType = ObjectTypeNone;
            editor.selectedObjectIndex = UINT_MAX;
        }
        else {
            editor.selectedObjectType = collisionQueryResult.objectType;
            editor.selectedObjectIndex = collisionQueryResult.objectIndex;
        }
    }
    {
        pxTimeAccumulated += frameTime;
        while (pxTimeAccumulated >= pxTimeStep) {
            pxScene->simulate(pxTimeStep);
            pxScene->fetchResults(true);
            pxTimeAccumulated -= pxTimeStep;
        }
    }
    {
        player.transformPrevFrame = player.transform;
        for (GameObject& obj : gameObjects) {
            obj.transformPrevFrame = obj.transform;
        }
    }
    {
        modelInstanceUpdateAnimation(&player.modelInstance, frameTime);
        for (GameObject& obj : gameObjects) {
            modelInstanceUpdateAnimation(&obj.modelInstance, frameTime);
        }
    }

    static ImVec2 mousePosPrev = ImGui::GetMousePos();
    ImVec2 mousePos = ImGui::GetMousePos();
    ImVec2 mouseDelta = {mousePos.x - mousePosPrev.x, mousePos.y - mousePosPrev.y};
    mousePosPrev = mousePos;

    editorMainMenuBar();
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    editorObjectsWindow();
    editorPropertiesWindow();
    editorAssetsWindow();

    if (editor.mode == EditorModeFreeCam) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
            mouseSelectX = (uint)mousePos.x;
            mouseSelectY = (uint)mousePos.y;
        }
        else {
            mouseSelectX = UINT_MAX;
            mouseSelectY = UINT_MAX;
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && editor.selectedObjectType != ObjectTypeNone && !ImGui::GetIO().WantCaptureMouse) {
            ImGui::OpenPopup("selected object popup");
        }
        if (ImGui::BeginPopup("selected object popup")) {
            if (ImGui::Selectable("Edit Object")) {
                editor.mode = EditorModeEditObject;
                setRigidActorsVisualization(false);
                if (editor.selectedObjectType == ObjectTypePlayer) {
                    player.transform.t = float3(0, 0, 0);
                    player.pxController->setFootPosition(PxVec3d(0, 0, 0));
                    player.pxController->getActor()->setActorFlag(PxActorFlag::eVISUALIZATION, true);
                }
                else if (editor.selectedObjectType == ObjectTypeGameObject) {
                    GameObject& obj = gameObjects[editor.selectedObjectIndex];
                    obj.transform.t = float3(0, 0, 0);
                    obj.transform.r = float4(0, 0, 0, 1);
                    if (obj.rigidActor) {
                        obj.rigidActor->setGlobalPose(PxTransform(PxIdentity));
                        obj.rigidActor->setActorFlag(PxActorFlag::eVISUALIZATION, true);
                    }
                }
                ImGui::CloseCurrentPopup();
            }
            if (editor.selectedObjectType == ObjectTypeGameObject) {
                if (ImGui::Selectable("Delete Object")) {
                    gameObjects[editor.selectedObjectIndex].toBeDeleted = true;
                    editor.selectedObjectType = ObjectTypeNone;
                    editor.selectedObjectIndex = UINT_MAX;
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::EndPopup();
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && !ImGui::GetIO().WantCaptureMouse) {
            editor.camera.moving = true;
            windowHideCursor(true);
        }
        if (editor.camera.moving && ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
            editor.camera.moving = false;
            windowHideCursor(false);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            editor.selectedObjectType = ObjectTypeNone;
        }
    }
    else if (editor.mode == EditorModeEditObject) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && !ImGui::GetIO().WantCaptureMouse) {
            editor.editCamera.moving = true;
            windowHideCursor(true);
        }
        if (editor.editCamera.moving && ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
            editor.editCamera.moving = false;
            windowHideCursor(false);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            setRigidActorsVisualization(true);
            if (editor.selectedObjectType == ObjectTypePlayer) {
                player.transform = player.transformDefault;
                player.pxController->setFootPosition(player.transform.t.toPxVec3d());
            }
            else if (editor.selectedObjectType == ObjectTypeGameObject) {
                GameObject& obj = gameObjects[editor.selectedObjectIndex];
                obj.transform = obj.transformDefault;
                if (obj.rigidActor) {
                    obj.rigidActor->setGlobalPose(obj.transform.toPxTransform());
                }
            }
            editor.mode = EditorModeFreeCam;
        }
    }
    {
        CameraEditor& camera = editor.mode == EditorModeFreeCam ? editor.camera : editor.editCamera;
        if (camera.moving || controllerStickMoved()) {
            float pitch = (mouseDeltaRaw.y * mouseSensitivity) - (controller.rsY * controllerSensitivity * frameTime);
            float yaw = (mouseDeltaRaw.x * mouseSensitivity) + (controller.rsX * controllerSensitivity * frameTime);
            float distance = frameTime * camera.moveSpeed;
            float3 translate = {-controller.lsX * distance, 0, controller.lsY * distance};
            if (ImGui::IsKeyDown(ImGuiKey_W)) translate.z = distance;
            if (ImGui::IsKeyDown(ImGuiKey_S)) translate.z = -distance;
            if (ImGui::IsKeyDown(ImGuiKey_A)) translate.x = distance;
            if (ImGui::IsKeyDown(ImGuiKey_D)) translate.x = -distance;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) translate.y = distance;
            if (ImGui::IsKeyDown(ImGuiKey_E)) translate.y = -distance;
            editorCameraRotate(camera, float2(pitch, yaw));
            editorCameraTranslate(camera, translate);
            if (pitch != 0 || yaw != 0 || translate != float3(0, 0, 0)) {
                if (pathTracer) {
                    d3d.pathTracerAccumulationCount = 0;
                }
            }
        }
        {
            cameraViewProjectMatPrevFrame = cameraViewProjectMat;
            cameraViewMat = XMMatrixLookAtLH(camera.position.toXMVector(), camera.lookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
            cameraViewMatInverseTranspose = XMMatrixTranspose(XMMatrixInverse(nullptr, cameraViewMat));
            cameraProjectMat = XMMatrixPerspectiveFovLH(radian(camera.fovVertical), (float)renderW / (float)renderH, 0.01f, 1000.0f);
            cameraViewProjectMat = XMMatrixMultiply(cameraViewMat, cameraProjectMat);
            cameraViewProjectMatInverse = XMMatrixInverse(nullptr, cameraViewProjectMat);
        }
    }
    ImGuizmo::SetRect(0, 0, (float)renderW, (float)renderH);
    ImGuizmo::BeginFrame();
    auto transformGizmo = [](XMMatrix* transformMat) -> bool {
        if (ImGui::IsKeyPressed(ImGuiKey_Space) && ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
            ImGui::OpenPopup("gizmo properties");
        }
        if (ImGui::BeginPopup("gizmo properties")) {
            if (ImGui::Selectable("local", editor.gizmoMode == ImGuizmo::LOCAL)) {
                editor.gizmoMode = ImGuizmo::LOCAL;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::Selectable("world", editor.gizmoMode == ImGuizmo::WORLD)) {
                editor.gizmoMode = ImGuizmo::WORLD;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (!ImGui::IsAnyItemActive() && !editor.camera.moving && !editor.editCamera.moving) {
            if (ImGui::IsKeyPressed(ImGuiKey_T)) {
                editor.gizmoOperation = ImGuizmo::TRANSLATE;
            }
            else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                editor.gizmoOperation = ImGuizmo::ROTATE;
            }
            else if (ImGui::IsKeyPressed(ImGuiKey_S)) {
                editor.gizmoOperation = ImGuizmo::SCALE;
            }
        }
        return ImGuizmo::Manipulate((const float*)&cameraViewMat, (const float*)&cameraProjectMat, editor.gizmoOperation, editor.gizmoMode, (float*)transformMat);
    };
    if (editor.mode == EditorModeFreeCam) {
        if (editor.selectedObjectType == ObjectTypePlayer) {
            XMMatrix transformMat = player.transformDefault.toMat();
            if (transformGizmo(&transformMat)) {
                player.transformDefault = Transform(transformMat);
                player.transform = player.transformDefault;
                player.pxController->setFootPosition(player.transform.t.toPxVec3d());
            }
        }
        else if (editor.selectedObjectType == ObjectTypeGameObject && editor.selectedObjectIndex < gameObjects.size()) {
            GameObject& obj = gameObjects[editor.selectedObjectIndex];
            XMMatrix transformMat = obj.transformDefault.toMat();
            if (transformGizmo(&transformMat)) {
                editorUpdateGameObjectTransformDefault(obj, Transform(transformMat));
            }
        }
    }
    else if (editor.mode == EditorModeEditObject) {
    }
    // else if (editor.mode == EditorModeEditModel) {
    //     if (editor.selectedModelCollisionShape) {
    //         XMMatrix transformMat = Transform(editor.selectedModelCollisionShape->getLocalPose()).toMat();
    //         if (transformGizmo(&transformMat)) {
    //             Transform transform(transformMat);
    //             const PxGeometry& pxGeometry = editor.selectedModelCollisionShape->getGeometry();
    //             auto shapeSetGeometryAndTransform = [&](const PxGeometry& geometry, PxTransform transform) {
    //                 editor.selectedModelTempRigidActor->detachShape(*editor.selectedModelCollisionShape);
    //                 editor.selectedModelCollisionShape->setGeometry(geometry);
    //                 editor.selectedModelCollisionShape->setLocalPose(transform);
    //                 editor.selectedModelTempRigidActor->attachShape(*editor.selectedModelCollisionShape);
    //                 modelShapesSyncGameObjects(&models[editor.selectedModelIndex]);
    //             };
    //             if (pxGeometry.getType() == PxGeometryType::ePLANE) {
    //                 PxPlaneGeometry planeGeometry = (PxPlaneGeometry&)pxGeometry;
    //                 shapeSetGeometryAndTransform(planeGeometry, Transform(transformMat).toPxTransform());
    //             }
    //             else if (pxGeometry.getType() == PxGeometryType::eBOX) {
    //                 PxBoxGeometry boxGeometry = (PxBoxGeometry&)pxGeometry;
    //                 boxGeometry.halfExtents.x *= std::clamp(transform.s.x, 0.98f, 1.02f);
    //                 boxGeometry.halfExtents.y *= std::clamp(transform.s.y, 0.98f, 1.02f);
    //                 boxGeometry.halfExtents.z *= std::clamp(transform.s.z, 0.98f, 1.02f);
    //                 shapeSetGeometryAndTransform(boxGeometry, Transform(transformMat).toPxTransform());
    //             }
    //             else if (pxGeometry.getType() == PxGeometryType::eSPHERE) {
    //                 PxSphereGeometry sphereGeometry = (PxSphereGeometry&)pxGeometry;
    //                 float s = std::clamp((transform.s.x + transform.s.y + transform.s.z) / 3.0f, 0.98f, 1.02f);
    //                 sphereGeometry.radius *= s;
    //                 shapeSetGeometryAndTransform(sphereGeometry, Transform(transformMat).toPxTransform());
    //             }
    //             else if (pxGeometry.getType() == PxGeometryType::eCAPSULE) {
    //                 PxCapsuleGeometry capsuleGeometry = (PxCapsuleGeometry&)pxGeometry;
    //                 capsuleGeometry.halfHeight *= std::clamp(transform.s.x, 0.98f, 1.02f);
    //                 capsuleGeometry.radius *= std::clamp((transform.s.y + transform.s.z) / 2.0f, 0.98f, 1.02f);
    //                 shapeSetGeometryAndTransform(capsuleGeometry, Transform(transformMat).toPxTransform());
    //             }
    //             else if (pxGeometry.getType() == PxGeometryType::eCONVEXMESH) {
    //                 PxConvexMeshGeometry convexMeshGeometry = (PxConvexMeshGeometry&)pxGeometry;
    //                 shapeSetGeometryAndTransform(convexMeshGeometry, Transform(transformMat).toPxTransform());
    //             }
    //             else {
    //                 assert(false);
    //             }
    //         }
    //     }
    // }
    {
        auto objIter = gameObjects.begin();
        while (objIter != gameObjects.end()) {
            if (objIter->toBeDeleted) {
                gameObjectRelease(*objIter);
                objIter = gameObjects.erase(objIter);
            }
            else {
                objIter++;
            }
        }
    }
}
#endif

void liveReloadFuncs() {
    static HMODULE gameDLL = nullptr;
    static std::filesystem::path gameDLLPath = exeDir / "game.dll";
    static std::filesystem::path gameDLLCopyPath = exeDir / "game.dll";
    static std::filesystem::file_time_type gameDLLPrevLastWriteTime = {};
    if (std::filesystem::is_regular_file(gameDLLPath)) {
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(gameDLLPath);
        if (lastWriteTime > gameDLLPrevLastWriteTime) {
            gameDLLPrevLastWriteTime = lastWriteTime;
            FreeLibrary(gameDLL);
            CopyFileW(gameDLLPath.c_str(), gameDLLCopyPath.c_str(), false);
            gameDLL = LoadLibraryW(gameDLLCopyPath.c_str());
            assert(gameDLL);
        }
    }
}

void imguiGameMenu() {
    ImGuiIO& imguiIO = ImGui::GetIO();
    ImVec2 size = imguiIO.DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(size);
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs;
    ImGui::Begin("GameMenuWindow", nullptr, windowFlags);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    imguiFont->RenderText(drawList, imguiFont->FontSize, ImVec2(size.x * 0.2f, size.y * 0.2f), 0xffffffff, ImVec4(0, 0, size.x, size.y), "test0 test0 test0", nullptr);
    imguiFont->RenderText(drawList, imguiFont->FontSize, ImVec2(size.x * 0.2f, size.y * 0.2f + imguiFont->FontSize), 0xffffffff, ImVec4(0, 0, size.x, size.y), "test1 test1 test1", nullptr);
    ImGui::End();
}

void gameUpdate() {
    ZoneScopedN("gameUpdate");
    if (!showMenu && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        showMenu = true;
        windowHideCursor(false);
    }
    else if (showMenu) {
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            showMenu = false;
            windowHideCursor(true);
        }
        else {
            imguiGameMenu();
            return;
        }
    }
    {
        pxTimeAccumulated += frameTime;
        while (pxTimeAccumulated >= pxTimeStep) {
            pxScene->simulate(pxTimeStep);
            pxScene->fetchResults(true);
            pxTimeAccumulated -= pxTimeStep;
        }
        for (GameObject& obj : gameObjects) {
            obj.transformPrevFrame = obj.transform;
            if (obj.rigidActor && obj.rigidActor->getType() == PxActorType::eRIGID_DYNAMIC) {
                PxTransform pxTransform = obj.rigidActor->getGlobalPose();
                obj.transform.t = pxTransform.p;
                obj.transform.r = pxTransform.q;
            }
        }
    }
    {
        float3 playerMoveDir(0, 0, 0);
        bool playerJump = false;
        {
            float3 forwardDir = player.camera.lookAt - player.camera.position;
            forwardDir.y = 0;
            forwardDir = forwardDir.normalize();
            float3 sideDir = forwardDir.cross(float3(0, 1, 0));
            sideDir.y = 0;
            if (ImGui::IsKeyDown(ImGuiKey_W)) playerMoveDir += forwardDir;
            if (ImGui::IsKeyDown(ImGuiKey_S)) playerMoveDir += -forwardDir;
            if (ImGui::IsKeyDown(ImGuiKey_A)) playerMoveDir += sideDir;
            if (ImGui::IsKeyDown(ImGuiKey_D)) playerMoveDir += -sideDir;
            playerMoveDir += forwardDir * controller.lsY;
            playerMoveDir += sideDir * -controller.lsX;
            playerMoveDir = playerMoveDir.normalize();

            playerJump = ImGui::IsKeyPressed(ImGuiKey_Space);
        }
        if (playerMoveDir == float3(0, 0, 0)) {
            player.state = PlayerStateIdle;
            player.velocity = float3(0, 0, 0);
            player.modelInstance.animation = &player.modelInstance.model->animations[player.idleAnimationIndex];
        }
        else {
            if (!ImGui::IsKeyDown(ImGuiKey_LeftShift) && (controller.lsX * controller.lsX + controller.lsY * controller.lsY) < 0.6f) {
                player.state = PlayerStateWalk;
                player.velocity = playerMoveDir * player.walkSpeed;
                player.modelInstance.animation = &player.modelInstance.model->animations[player.walkAnimationIndex];
            }
            else {
                player.state = PlayerStateRun;
                player.velocity = playerMoveDir * player.runSpeed;
                player.modelInstance.animation = &player.modelInstance.model->animations[player.runAnimationIndex];
            }
        }
        player.velocity.y = -5.0f;

        float3 playerOldPosition = player.transform.t;
        player.pxController->move(player.velocity.toPxVec3() * frameTime, 0.001f, frameTime, PxControllerFilters());
        player.transformPrevFrame = player.transform;
        player.transform.t = float3(player.pxController->getFootPosition());
        playerCameraTranslate(player.transform.t - playerOldPosition);
        if (player.velocity.x != 0.0f && player.velocity.y != 0.0f) {
            float3 velocity(player.velocity.x, 0, player.velocity.z);
            float angle = acosf(velocity.normalize().dot(float3(0, 0, -1)));
            if (velocity.x > 0) angle = -angle;
            player.transform.r = float4(XMQuaternionRotationRollPitchYaw(0, angle, 0));
        }
    }
    {
        modelInstanceUpdateAnimation(&player.modelInstance, frameTime);
        for (GameObject& obj : gameObjects) {
            modelInstanceUpdateAnimation(&obj.modelInstance, frameTime);
        }
    }
    {
        float pitch = (-mouseDeltaRaw.y * mouseSensitivity) + (controller.rsY * controllerSensitivity * frameTime);
        float yaw = (mouseDeltaRaw.x * mouseSensitivity) + (controller.rsX * controllerSensitivity * frameTime);
        playerCameraSetPitchYaw(player.camera.pitchYaw + float2(pitch, yaw));

        cameraViewProjectMatPrevFrame = cameraViewProjectMat;
        cameraViewMat = XMMatrixLookAtLH(player.camera.position.toXMVector(), player.camera.lookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
        cameraViewMatInverseTranspose = XMMatrixTranspose(XMMatrixInverse(nullptr, cameraViewMat));
        cameraProjectMat = XMMatrixPerspectiveFovLH(radian(player.camera.fovVertical), (float)renderW / (float)renderH, 0.01f, 1000.0f);
        cameraViewProjectMat = XMMatrixMultiply(cameraViewMat, cameraProjectMat);
        cameraViewProjectMatInverse = XMMatrixInverse(nullptr, cameraViewProjectMat);
    }
}

void update() {
    ZoneScopedN("update");
    ImGui::GetIO().DeltaTime = frameTime;
    ImGui::GetIO().DisplaySize = ImVec2((float)renderW, (float)renderH);
    ImGui::NewFrame();
    if (ImGui::IsKeyDown(ImGuiKey_F4) && ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
        quit = true;
    }
#ifdef EDITOR
    if (ImGui::IsKeyPressed(ImGuiKey_P) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
        toggleBetweenEditorPlay();
    }
    if (editorActive) {
        editorUpdate();
    }
    else {
        gameUpdate();
    }
    {
        ZoneScopedN("SkeletonVisualization");
        if (editor.selectedObjectType == ObjectTypePlayer) {
            if (player.modelInstance.model->skeletonRootNode) {
                XMMatrix playerTransformMat = (editor.mode == EditorModeEditObject) ? xmMatrixIdentity : player.transform.toMat();
                XMMatrix transformMat = XMMatrixMultiply(XMMatrixMultiply(player.modelInstance.model->meshNodes[0]->globalTransform, XMMatrixScaling(1, 1, -1)), playerTransformMat);
                modelInstanceGetSkeletonVisualization(&player.modelInstance, player.modelInstance.model->skeletonRootNode, transformMat);
            }
        }
        else if (editor.selectedObjectType == ObjectTypeGameObject) {
            GameObject& gameObject = gameObjects[editor.selectedObjectIndex];
            if (gameObject.modelInstance.model->skeletonRootNode) {
                XMMatrix objTransformMat = (editor.mode == EditorModeEditObject) ? xmMatrixIdentity : gameObject.transform.toMat();
                XMMatrix transformMat = XMMatrixMultiply(XMMatrixMultiply(gameObject.modelInstance.model->meshNodes[0]->globalTransform, XMMatrixScaling(1, 1, -1)), objTransformMat);
                modelInstanceGetSkeletonVisualization(&gameObject.modelInstance, gameObject.modelInstance.model->skeletonRootNode, transformMat);
            }
        }
    }
#else
    gameUpdate();
#endif
    {
        debugSpheres.resize(0);
        debugLines.resize(0);
        debugTriangles.resize(0);
    }
    if (showRigidActorsGeometries) {
        ZoneScopedN("RigidActorsVisualization");
        const PxRenderBuffer& physxRenderBuffer = pxScene->getRenderBuffer();
        for (PxU32 i = 0; i < physxRenderBuffer.getNbPoints(); i++) {
            const PxDebugPoint& pxPoint = physxRenderBuffer.getPoints()[i];
            debugSpheres.push_back(Sphere{.center = float3(pxPoint.pos), .radius = 0.02f, .color = 0xffffffff});
        }
        for (PxU32 i = 0; i < physxRenderBuffer.getNbLines(); i++) {
            const PxDebugLine& pxLine = physxRenderBuffer.getLines()[i];
            debugLines.push_back(Line{.p0 = float3(pxLine.pos0), .p1 = float3(pxLine.pos1), .radius = 0.01f, .color = 0xffffffff});
        }
        for (PxU32 i = 0; i < physxRenderBuffer.getNbTriangles(); i++) {
            const PxDebugTriangle& pxTriangle = physxRenderBuffer.getTriangles()[i];
            debugTriangles.push_back(Triangle{.p0 = float3(pxTriangle.pos0), .p1 = float3(pxTriangle.pos1), .p2 = float3(pxTriangle.pos2), .color = 0xffffffff});
        }
    }
    ImGui::ShowDebugLogWindow();
    ImGui::Render();
}

D3D12_DISPATCH_RAYS_DESC fillRayTracingShaderTable(D3DUploadBuffer* buffer, void* rayGenID, std::span<void*> missIDs, std::span<void*> hitGroupIDs) {
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    buffer->size = align(buffer->size, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    memcpy(buffer->ptr + buffer->size, rayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    dispatchDesc.RayGenerationShaderRecord = {buffer->buffer->GetResource()->GetGPUVirtualAddress() + buffer->size, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
    buffer->size = align(buffer->size + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.MissShaderTable = {buffer->buffer->GetResource()->GetGPUVirtualAddress() + buffer->size, missIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* missID : missIDs) {
        memcpy(buffer->ptr + buffer->size, missID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        buffer->size = align(buffer->size + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    buffer->size = align(buffer->size, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.HitGroupTable = {buffer->buffer->GetResource()->GetGPUVirtualAddress() + buffer->size, hitGroupIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* hitGroupID : hitGroupIDs) {
        memcpy(buffer->ptr + buffer->size, hitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        buffer->size = align(buffer->size + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    return dispatchDesc;
}

void d3dRender() {
    ZoneScopedN("d3dRender");
    d3dWaitFence(&d3d.renderFence);
    blasInstancesDescs.resize(0);
    blasInstancesInfos.resize(0);
    blasGeometriesInfos.resize(0);
    d3d.stagingBuffer.size = 0;
    d3d.constantsBuffer.size = 0;
    d3d.cbvSrvUavDescriptorHeap.size = 0;
    d3dCmdListReset();
    d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap.heap);
    {
        RenderInfo renderInfo = {
            .cameraViewMat = cameraViewMat,
            .cameraViewMatInverseTranspose = cameraViewMatInverseTranspose,
            .cameraProjectMat = cameraProjectMat,
            .cameraViewProjectMat = cameraViewProjectMat,
            .cameraViewProjectMatInverse = cameraViewProjectMatInverse,
            .cameraViewProjectMatPrevFrame = cameraViewProjectMatPrevFrame,
        };
        if (pathTracer) {
            d3d.pathTracerAccumulationCount += 1;
            if (d3d.pathTracerAccumulationCount > d3d.pathTracerAccumulationCountMax) {
                d3d.pathTracerAccumulationCount = d3d.pathTracerAccumulationCountMax;
            }
            renderInfo.pathTracerAccumulationFrameCount = d3d.pathTracerAccumulationCount;
        }
        assert(d3d.constantsBuffer.size == 0);
        memcpy(d3d.constantsBuffer.ptr + d3d.constantsBuffer.size, &renderInfo, sizeof(renderInfo));
        d3d.constantsBuffer.size += sizeof(renderInfo);
    }
    {
        ZoneScopedN("vertexSkinning + BLAS build");
        struct MeshSkinningInfo {
            ModelNode* meshNode;
            ModelInstanceMeshNode* instanceMeshNode;
            D3D12_GPU_VIRTUAL_ADDRESS matsBuffer;
        };
        static std::vector<ModelInstance*> skinnedModelInstances;
        static std::vector<MeshSkinningInfo> meshSkinningInfos;
        static std::vector<D3D12_RESOURCE_BARRIER> verticeBufferBarriers;
        static std::vector<D3D12_RESOURCE_BARRIER> blasBarriers;
        skinnedModelInstances.resize(0);
        meshSkinningInfos.resize(0);
        verticeBufferBarriers.resize(0);
        blasBarriers.resize(0);

        skinnedModelInstances.push_back(&player.modelInstance);
        for (GameObject& object : gameObjects) {
            if (object.modelInstance.skins.size() > 0 && object.modelInstance.animation) {
                skinnedModelInstances.push_back(&object.modelInstance);
            }
        }
        for (ModelInstance* modelInstance : skinnedModelInstances) {
            for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance->model->meshNodes.size(); meshNodeIndex++) {
                ModelNode* meshNode = modelInstance->model->meshNodes[meshNodeIndex];
                ModelInstanceMeshNode* instanceMeshNode = &modelInstance->meshNodes[meshNodeIndex];
                if (!meshNode->skin) continue;
                int skinIndex = (int)(meshNode->skin - &modelInstance->model->skins[0]);
                verticeBufferBarriers.push_back(D3D12_RESOURCE_BARRIER{.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = instanceMeshNode->verticesBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}});
                meshSkinningInfos.push_back(MeshSkinningInfo{.meshNode = meshNode, .instanceMeshNode = instanceMeshNode, .matsBuffer = modelInstance->skins[skinIndex].matsBuffer->GetResource()->GetGPUVirtualAddress()});
            }
        }
        PIXSetMarker(d3d.graphicsCmdList, 0, "vertexSkinning");
        d3d.graphicsCmdList->SetPipelineState(d3d.vertexSkinningPSO);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.vertexSkinningRootSig);
        d3d.graphicsCmdList->ResourceBarrier((uint)verticeBufferBarriers.size(), verticeBufferBarriers.data());
        for (MeshSkinningInfo& info : meshSkinningInfos) {
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(0, info.matsBuffer);
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, info.meshNode->mesh->verticesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRootUnorderedAccessView(2, info.instanceMeshNode->verticesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRoot32BitConstant(3, (uint)info.meshNode->mesh->vertices.size(), 0);
            d3d.graphicsCmdList->Dispatch((uint)info.meshNode->mesh->vertices.size() / 32 + 1, 1, 1);
        }
        for (D3D12_RESOURCE_BARRIER& barrier : verticeBufferBarriers) std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier((uint)verticeBufferBarriers.size(), verticeBufferBarriers.data());

        PIXSetMarker(d3d.graphicsCmdList, 0, "BLAS build");
        for (MeshSkinningInfo& info : meshSkinningInfos) {
            static std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
            geometryDescs.resize(0);
            for (ModelPrimitive& primitive : info.meshNode->mesh->primitives) {
                geometryDescs.push_back(D3D12_RAYTRACING_GEOMETRY_DESC{
                    .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                    .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                    .Triangles = {
                        .Transform3x4 = 0,
                        .IndexFormat = DXGI_FORMAT_R32_UINT,
                        .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
                        .IndexCount = (uint)primitive.indicesCount,
                        .VertexCount = (uint)primitive.verticesCount,
                        .IndexBuffer = info.meshNode->mesh->indicesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.indicesBufferOffset * sizeof(uint32),
                        .VertexBuffer = {.StartAddress = info.instanceMeshNode->verticesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.verticesBufferOffset * sizeof(struct Vertex), .StrideInBytes = sizeof(struct Vertex)},
                    },
                });
            }
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD, .NumDescs = (uint)geometryDescs.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .pGeometryDescs = geometryDescs.data()};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {.DestAccelerationStructureData = info.instanceMeshNode->blas->GetResource()->GetGPUVirtualAddress(), .Inputs = blasInputs, .ScratchAccelerationStructureData = info.instanceMeshNode->blasScratch->GetResource()->GetGPUVirtualAddress()};
            d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);
            blasBarriers.push_back(D3D12_RESOURCE_BARRIER{.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = info.instanceMeshNode->blas->GetResource()}});
        }
        d3d.graphicsCmdList->ResourceBarrier((uint)blasBarriers.size(), blasBarriers.data());
    }
    {
        ZoneScopedN("TLAS build");
        PIXSetMarker(d3d.graphicsCmdList, 0, "TLAS build");
        modelsAppendDescriptorsAndBlasGeometriesInfos();
#ifdef EDITOR
        if (editor.mode == EditorModeFreeCam) {
            modelInstanceAddBLASInstancesToTLAS(player.modelInstance, player.transform.toMat(), player.transformPrevFrame.toMat(), ObjectTypePlayer, 0, 0, 0);
            for (uint objIndex = 0; objIndex < gameObjects.size(); objIndex++) {
                GameObject& obj = gameObjects[objIndex];
                modelInstanceAddBLASInstancesToTLAS(obj.modelInstance, obj.transform.toMat(), obj.transformPrevFrame.toMat(), ObjectTypeGameObject, objIndex, 0, 0);
            }
            if (editor.beginDragDropGameObject) {
                editor.dragDropGameObject.transformPrevFrame = editor.dragDropGameObject.transform;
                modelInstanceAddBLASInstancesToTLAS(editor.dragDropGameObject.modelInstance, editor.dragDropGameObject.transform.toMat(), editor.dragDropGameObject.transformPrevFrame.toMat(), ObjectTypeGameObject, UINT_MAX, 0, 0);
            }
        }
        else if (editor.mode == EditorModeEditObject) {
            if (editor.selectedObjectType == ObjectTypePlayer) {
                modelInstanceAddBLASInstancesToTLAS(player.modelInstance, player.transform.toMat(), player.transformPrevFrame.toMat(), ObjectTypePlayer, 0, 0, 0);
            }
            else if (editor.selectedObjectType == ObjectTypeGameObject) {
                GameObject& obj = gameObjects[editor.selectedObjectIndex];
                modelInstanceAddBLASInstancesToTLAS(obj.modelInstance, obj.transform.toMat(), obj.transformPrevFrame.toMat(), ObjectTypeGameObject, editor.selectedObjectIndex, 0, 0);
            }
        }
        for (Sphere& sphere : debugSpheres) {
            float s = sphere.radius;
            XMMatrix transformMat = XMMatrixAffineTransformation(XMVectorSet(s, s, s, 0), xmVectorZero, xmQuatIdentity, sphere.center.toXMVector());
            modelInstanceAddBLASInstancesToTLAS(modelInstanceSphere, transformMat, transformMat, ObjectTypeNone, 0, BLASInstanceFlagForcedColor, 0xff000000);
        }
        for (Line& line : debugLines) {
            float3 dir = line.p1 - line.p0;
            float s = line.radius;
            float sY = dir.length();
            float4 r = quaternionBetween(float3(0, 0.01f, 0), dir);
            XMMatrix transformMat = XMMatrixAffineTransformation(XMVectorSet(s, sY, s, 0), xmVectorZero, r.toXMVector(), line.p0.toXMVector());
            modelInstanceAddBLASInstancesToTLAS(modelInstanceCube, transformMat, transformMat, ObjectTypeNone, 0, BLASInstanceFlagForcedColor, 0xff000000);
        }
#else
        modelInstanceAddBLASInstancesToTLAS(player.modelInstance, player.transform.toMat(), ObjectTypePlayer, 0, 0, 0);
        for (uint objIndex = 0; objIndex < gameObjects.size(); objIndex++) {
            GameObject& obj = gameObjects[objIndex];
            modelInstanceAddBLASInstancesToTLAS(obj.modelInstance, obj.transform.toMat(), ObjectTypeGameObject, objIndex, 0, 0);
        }
        for (Sphere& sphere : debugSpheres) {
            float s = sphere.radius / scaleFactor;
            XMMatrix transformMat = XMMatrixAffineTransformation(XMVectorSet(s, s, s, 0), xmVectorZero, xmQuatIdentity, sphere.center.toXMVector());
            modelInstanceAddBLASInstancesToTLAS(modelInstanceSphere, transformMat, ObjectTypeOthers, 0, BLASInstanceFlagForcedColor, 0xff000000);
        }
        for (Cylinder& cylinder : debugCylinders) {
            float3 dir = cylinder.p1 - cylinder.p0;
            float s = cylinder.radius / scaleFactor;
            float sY = dir.length() / scaleFactor;
            float4 r = quaternionBetween(float3(0, 0.001f, 0), dir);
            XMMatrix transformMat = XMMatrixAffineTransformation(XMVectorSet(s, sY, s, 0), xmVectorZero, r.toXMVector(), cylinder.p0.toXMVector());
            modelInstanceAddBLASInstancesToTLAS(modelInstanceCylinder, transformMat, ObjectTypeOthers, 0, BLASInstanceFlagForcedColor, 0xff000000);
        }
#endif
        assert(vectorSizeof(blasInstancesDescs) < d3d.blasInstanceDescsBuffer.capacity);
        assert(vectorSizeof(blasInstancesInfos) < d3d.blasInstancesInfosBuffer.capacity);
        assert(vectorSizeof(blasGeometriesInfos) < d3d.blasGeometriesInfosBuffer.capacity);
        memcpy(d3d.blasInstanceDescsBuffer.ptr, blasInstancesDescs.data(), vectorSizeof(blasInstancesDescs));
        memcpy(d3d.blasInstancesInfosBuffer.ptr, blasInstancesInfos.data(), vectorSizeof(blasInstancesInfos));
        memcpy(d3d.blasGeometriesInfosBuffer.ptr, blasGeometriesInfos.data(), vectorSizeof(blasGeometriesInfos));
        {
            ZoneScopedN("BuildRaytracingAccelerationStructure");
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {
                .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                .NumDescs = (uint)blasInstancesDescs.size(),
                .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
                .InstanceDescs = d3d.blasInstanceDescsBuffer.buffer->GetResource()->GetGPUVirtualAddress(),
            };
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
            d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &prebuildInfo);
            assert(prebuildInfo.ResultDataMaxSizeInBytes < d3d.tlasBuffer->GetSize());
            assert(prebuildInfo.ScratchDataSizeInBytes < d3d.tlasScratchBuffer->GetSize());

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress(), .Inputs = tlasInputs, .ScratchAccelerationStructureData = d3d.tlasScratchBuffer->GetResource()->GetGPUVirtualAddress()};
            d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            D3D12_RESOURCE_BARRIER tlasBarrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = d3d.tlasBuffer->GetResource()}};
            d3d.graphicsCmdList->ResourceBarrier(1, &tlasBarrier);
        }
    }
#ifdef EDITOR
    if (!mouseSelectOngoing && mouseSelectX != UINT_MAX && mouseSelectY != UINT_MAX) {
        mouseSelectOngoing = true;
        XMFloat4x4 viewMat;
        XMFloat4x4 projMat;
        XMStoreFloat4x4(&viewMat, cameraViewMatInverseTranspose);
        XMStoreFloat4x4(&projMat, cameraProjectMat);
        float2 pixelCoord = ((float2((float)mouseSelectX, (float)mouseSelectY) + 0.5f) / float2((float)renderW, (float)renderH)) * 2.0f - 1.0f;
        RayDesc rayDesc = {.origin = {viewMat.m[0][3], viewMat.m[1][3], viewMat.m[2][3]}, .min = 0.0f, .max = FLT_MAX};
        float aspect = projMat.m[1][1] / projMat.m[0][0];
        float tanHalfFovY = 1.0f / projMat.m[1][1];
        rayDesc.dir = (float3(viewMat.m[0][0], viewMat.m[1][0], viewMat.m[2][0]) * pixelCoord.x * tanHalfFovY * aspect) - (float3(viewMat.m[0][1], viewMat.m[1][1], viewMat.m[2][1]) * pixelCoord.y * tanHalfFovY) + (float3(viewMat.m[0][2], viewMat.m[1][2], viewMat.m[2][2]));
        rayDesc.dir = rayDesc.dir.normalize();
        ((CollisionQuery*)d3d.collisionQueriesBuffer.ptr)[0] = {.rayDesc = rayDesc, .instanceInclusionMask = 0xff & ~ObjectTypeNone};

        D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.collisionQueryResultsBuffer.bufferUAV->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        d3d.graphicsCmdList->ResourceBarrier(1, &barrier);

        void* missIDs[1] = {d3d.collisionQueryMissID};
        void* hitGroupIDs[1] = {d3d.collisionQueryHitGroupID};
        D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(&d3d.constantsBuffer, d3d.collisionQueryRayGenID, missIDs, hitGroupIDs);
        dispatchDesc.Width = 2, dispatchDesc.Height = 1, dispatchDesc.Depth = 1;
        assert(d3d.constantsBuffer.size < d3d.constantsBuffer.capacity);

        PIXSetMarker(d3d.graphicsCmdList, 0, "collisionQuery");
        d3d.graphicsCmdList->SetPipelineState1(d3d.collisionQueryPSO);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.collisionQueryRootSig);
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(0, d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, d3d.collisionQueriesBuffer.buffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(2, d3d.blasInstancesInfosBuffer.buffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootUnorderedAccessView(3, d3d.collisionQueryResultsBuffer.bufferUAV->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(1, &barrier);
        d3d.graphicsCmdList->CopyBufferRegion(d3d.collisionQueryResultsBuffer.buffer->GetResource(), 0, d3d.collisionQueryResultsBuffer.bufferUAV->GetResource(), 0, d3d.collisionQueryResultsBuffer.capacity);

        d3dCmdListExecute();
        d3dSignalFence(&d3d.collisionQueriesFence);
        assert(SUCCEEDED(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocator, nullptr)));
        d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap.heap);
    }
#endif
    if (pathTracer) {
        if (d3d.pathTracerAccumulationCount < d3d.pathTracerAccumulationCountMax) {
            void* rayMissIDs[] = {d3d.pathTracerRayMissID};
            void* rayHitGroupIDs[] = {d3d.pathTracerRayHitGroupID};
            D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(&d3d.constantsBuffer, d3d.pathTracerRayGenID, rayMissIDs, rayHitGroupIDs);
            dispatchDesc.Width = renderW, dispatchDesc.Height = renderH, dispatchDesc.Depth = 1;
            assert(d3d.constantsBuffer.size < d3d.constantsBuffer.capacity);

            D3D12_RESOURCE_BARRIER transitions[] = {
                {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
                {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.depthTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
            };
            d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);

            PIXSetMarker(d3d.graphicsCmdList, 0, "pathTracer");
            d3d.graphicsCmdList->SetPipelineState1(d3d.pathTracerPSO);
            d3d.graphicsCmdList->SetComputeRootSignature(d3d.pathTracerRootSig);
            d3d.graphicsCmdList->SetComputeRootConstantBufferView(0, d3d.constantsBuffer.buffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(2, d3d.blasInstancesInfosBuffer.buffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(3, d3d.blasGeometriesInfosBuffer.buffer->GetResource()->GetGPUVirtualAddress());
            D3D12_UNORDERED_ACCESS_VIEW_DESC renderTextureUAVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
            D3D12_UNORDERED_ACCESS_VIEW_DESC pathTracerAccumulationTextureUAVDesc = {.Format = d3d.pathTracerAccumulationTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
            D3DDescriptor renderTextureUAVDescriptor = d3dAppendUAVDescriptor(&renderTextureUAVDesc, d3d.renderTexture->GetResource());
            D3DDescriptor pathTracerAccumulationTextureUAVDescriptor = d3dAppendUAVDescriptor(&pathTracerAccumulationTextureUAVDesc, d3d.pathTracerAccumulationTexture->GetResource());
            D3DDescriptor skyboxTextureDescriptor = d3dAppendSRVDescriptor(nullptr, skybox->hdriTexture->GetResource());
            d3d.graphicsCmdList->SetComputeRootDescriptorTable(4, renderTextureUAVDescriptor.gpuHandle);
            d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

            std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
            std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
            d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);
        }
    }
    else {
        void* missIDs[] = {d3d.renderSceneRayMissIDPrimary, d3d.renderSceneRayMissIDShadow};
        void* hitGroupIDs[] = {d3d.renderSceneRayHitGroupIDPrimary, d3d.renderSceneRayHitGroupIDShadow};
        D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(&d3d.constantsBuffer, d3d.renderSceneRayGenID, missIDs, hitGroupIDs);
        dispatchDesc.Width = renderW, dispatchDesc.Height = renderH, dispatchDesc.Depth = 1;
        assert(d3d.constantsBuffer.size < d3d.constantsBuffer.capacity);

        D3D12_RESOURCE_BARRIER transitions[] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.depthTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
        };
        d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);

        PIXSetMarker(d3d.graphicsCmdList, 0, "renderScene");
        d3d.graphicsCmdList->SetPipelineState1(d3d.renderScenePSO);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.renderSceneRootSig);
        d3d.graphicsCmdList->SetComputeRootConstantBufferView(0, d3d.constantsBuffer.buffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(2, d3d.blasInstancesInfosBuffer.buffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(3, d3d.blasGeometriesInfosBuffer.buffer->GetResource()->GetGPUVirtualAddress());
        D3D12_UNORDERED_ACCESS_VIEW_DESC renderTextureUAVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC depthTextureUAVDesc = {.Format = d3d.depthTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC motionVectorTextureUAVDesc = {.Format = d3d.motionVectorTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3DDescriptor renderTextureUAVDescriptor = d3dAppendUAVDescriptor(&renderTextureUAVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor depthTextureUAVDescriptor = d3dAppendUAVDescriptor(&depthTextureUAVDesc, d3d.depthTexture->GetResource());
        D3DDescriptor motionVectorTextureUAVDescriptor = d3dAppendUAVDescriptor(&motionVectorTextureUAVDesc, d3d.motionVectorTexture->GetResource());
        D3DDescriptor skyboxTextureDescriptor = d3dAppendSRVDescriptor(nullptr, skybox->hdriTexture->GetResource());
        d3d.graphicsCmdList->SetComputeRootDescriptorTable(4, renderTextureUAVDescriptor.gpuHandle);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

        std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
        std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);
    }
    {
        uint swapChainBackBufferIndex = d3d.swapChain->GetCurrentBackBufferIndex();
        D3D12_RESOURCE_BARRIER transitions[] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.swapChainImages[swapChainBackBufferIndex], .StateBefore = D3D12_RESOURCE_STATE_PRESENT, .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET}},
        };
        d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);
        d3d.graphicsCmdList->OMSetRenderTargets(1, &d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], false, nullptr);
        D3D12_VIEWPORT viewport = {0, 0, (float)renderW, (float)renderH, 0, 1};
        RECT scissor = {0, 0, (long)renderW, (long)renderH};
        d3d.graphicsCmdList->RSSetViewports(1, &viewport);
        d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
        d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        {
            PIXSetMarker(d3d.graphicsCmdList, 0, "composite");
            d3d.graphicsCmdList->SetPipelineState(d3d.compositePSO);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.compositeRootSig);
            uint32 compositeFlags = 0;
            if (hdr) compositeFlags |= CompositeFlagHDR;
            d3d.graphicsCmdList->SetGraphicsRoot32BitConstant(0, compositeFlags, 0);
            D3D12_SHADER_RESOURCE_VIEW_DESC renderTextureSRVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = 1}};
            D3DDescriptor renderTextureSRVDescriptor = d3dAppendSRVDescriptor(&renderTextureSRVDesc, d3d.renderTexture->GetResource());
            d3d.graphicsCmdList->SetGraphicsRootDescriptorTable(1, renderTextureSRVDescriptor.gpuHandle);
            d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
        }
        {
            PIXSetMarker(d3d.graphicsCmdList, 0, "imgui");
            d3d.graphicsCmdList->SetPipelineState(d3d.imguiPSO);
            float blendFactor[] = {0, 0, 0, 0};
            d3d.graphicsCmdList->OMSetBlendFactor(blendFactor);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.imguiRootSig);
            int constants[3] = {renderW, renderH, hdr};
            d3d.graphicsCmdList->SetGraphicsRoot32BitConstants(0, countof(constants), constants, 0);
            D3DDescriptor imguiImageDescriptor = d3dAppendSRVDescriptor(nullptr, d3d.imguiTexture->GetResource());
            d3d.graphicsCmdList->SetGraphicsRootDescriptorTable(1, imguiImageDescriptor.gpuHandle);
            D3D12_VERTEX_BUFFER_VIEW vertBufferView = {d3d.imguiVertexBuffer.buffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiVertexBuffer.capacity, sizeof(ImDrawVert)};
            D3D12_INDEX_BUFFER_VIEW indexBufferView = {d3d.imguiIndexBuffer.buffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiIndexBuffer.capacity, DXGI_FORMAT_R32_UINT};
            d3d.graphicsCmdList->IASetVertexBuffers(0, 1, &vertBufferView);
            d3d.graphicsCmdList->IASetIndexBuffer(&indexBufferView);
            d3d.imguiVertexBuffer.size = 0;
            d3d.imguiIndexBuffer.size = 0;
            const ImDrawData* drawData = ImGui::GetDrawData();
            for (int i = 0; i < drawData->CmdListsCount; i++) {
                const ImDrawList* dlist = drawData->CmdLists[i];
                uint verticesSize = dlist->VtxBuffer.Size * sizeof(ImDrawVert);
                uint indicesSize = dlist->IdxBuffer.Size * sizeof(ImDrawIdx);
                memcpy(d3d.imguiVertexBuffer.ptr + d3d.imguiVertexBuffer.size, dlist->VtxBuffer.Data, verticesSize);
                memcpy(d3d.imguiIndexBuffer.ptr + d3d.imguiIndexBuffer.size, dlist->IdxBuffer.Data, indicesSize);
                uint vertexIndex = (uint)d3d.imguiVertexBuffer.size / sizeof(ImDrawVert);
                uint indiceIndex = (uint)d3d.imguiIndexBuffer.size / sizeof(ImDrawIdx);
                for (int i = 0; i < dlist->CmdBuffer.Size; i++) {
                    const ImDrawCmd& dcmd = dlist->CmdBuffer[i];
                    D3D12_RECT scissor = {(long)dcmd.ClipRect.x, (long)dcmd.ClipRect.y, (long)dcmd.ClipRect.z, (long)dcmd.ClipRect.w};
                    d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
                    d3d.graphicsCmdList->DrawIndexedInstanced(dcmd.ElemCount, 1, indiceIndex, vertexIndex, 0);
                    indiceIndex += dcmd.ElemCount;
                }
                d3d.imguiVertexBuffer.size += align(verticesSize, sizeof(ImDrawVert));
                d3d.imguiIndexBuffer.size += align(indicesSize, sizeof(ImDrawIdx));
                assert(d3d.imguiVertexBuffer.size < d3d.imguiVertexBuffer.capacity);
                assert(d3d.imguiIndexBuffer.size < d3d.imguiIndexBuffer.capacity);
            }
        }
        for (auto& transition : transitions) std::swap(transition.Transition.StateBefore, transition.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(countof(transitions), transitions);
    }
    {
        ZoneScopedN("ExecuteCommandLists");
        d3dCmdListExecute();
    }
    {
        ZoneScopedN("Present");
        assert(SUCCEEDED(d3d.swapChain->Present(0, 0)));
    }
    {
        d3dSignalFence(&d3d.renderFence);
    }
    {
        std::swap(d3d.graphicsCmdAllocator, d3d.graphicsCmdAllocatorPrevFrame);
        std::swap(d3d.graphicsCmdList, d3d.graphicsCmdListPrevFrame);
        std::swap(d3d.renderFence, d3d.renderFencePrevFrame);
        std::swap(d3d.cbvSrvUavDescriptorHeap, d3d.cbvSrvUavDescriptorHeapPrevFrame);
        std::swap(d3d.constantsBuffer, d3d.constantsBufferPrevFrame);
        std::swap(d3d.renderTexture, d3d.renderTexturePrevFrame);
        std::swap(d3d.imguiVertexBuffer, d3d.imguiVertexBufferPrevFrame);
        std::swap(d3d.imguiIndexBuffer, d3d.imguiIndexBufferPrevFrame);
    }
}

ImGuiKey virtualKeytoImGuiKey(WPARAM wparam) {
    switch (wparam) {
        case VK_TAB: return ImGuiKey_Tab;
        case VK_LEFT: return ImGuiKey_LeftArrow;
        case VK_RIGHT: return ImGuiKey_RightArrow;
        case VK_UP: return ImGuiKey_UpArrow;
        case VK_DOWN: return ImGuiKey_DownArrow;
        case VK_PRIOR: return ImGuiKey_PageUp;
        case VK_NEXT: return ImGuiKey_PageDown;
        case VK_HOME: return ImGuiKey_Home;
        case VK_END: return ImGuiKey_End;
        case VK_INSERT: return ImGuiKey_Insert;
        case VK_DELETE: return ImGuiKey_Delete;
        case VK_BACK: return ImGuiKey_Backspace;
        case VK_SPACE: return ImGuiKey_Space;
        case VK_RETURN: return ImGuiKey_Enter;
        case VK_ESCAPE: return ImGuiKey_Escape;
        case VK_OEM_7: return ImGuiKey_Apostrophe;
        case VK_OEM_COMMA: return ImGuiKey_Comma;
        case VK_OEM_MINUS: return ImGuiKey_Minus;
        case VK_OEM_PERIOD: return ImGuiKey_Period;
        case VK_OEM_2: return ImGuiKey_Slash;
        case VK_OEM_1: return ImGuiKey_Semicolon;
        case VK_OEM_PLUS: return ImGuiKey_Equal;
        case VK_OEM_4: return ImGuiKey_LeftBracket;
        case VK_OEM_5: return ImGuiKey_Backslash;
        case VK_OEM_6: return ImGuiKey_RightBracket;
        case VK_OEM_3: return ImGuiKey_GraveAccent;
        case VK_CAPITAL: return ImGuiKey_CapsLock;
        case VK_SCROLL: return ImGuiKey_ScrollLock;
        case VK_NUMLOCK: return ImGuiKey_NumLock;
        case VK_SNAPSHOT: return ImGuiKey_PrintScreen;
        case VK_PAUSE: return ImGuiKey_Pause;
        case VK_NUMPAD0: return ImGuiKey_Keypad0;
        case VK_NUMPAD1: return ImGuiKey_Keypad1;
        case VK_NUMPAD2: return ImGuiKey_Keypad2;
        case VK_NUMPAD3: return ImGuiKey_Keypad3;
        case VK_NUMPAD4: return ImGuiKey_Keypad4;
        case VK_NUMPAD5: return ImGuiKey_Keypad5;
        case VK_NUMPAD6: return ImGuiKey_Keypad6;
        case VK_NUMPAD7: return ImGuiKey_Keypad7;
        case VK_NUMPAD8: return ImGuiKey_Keypad8;
        case VK_NUMPAD9: return ImGuiKey_Keypad9;
        case VK_DECIMAL: return ImGuiKey_KeypadDecimal;
        case VK_DIVIDE: return ImGuiKey_KeypadDivide;
        case VK_MULTIPLY: return ImGuiKey_KeypadMultiply;
        case VK_SUBTRACT: return ImGuiKey_KeypadSubtract;
        case VK_ADD: return ImGuiKey_KeypadAdd;
        case (VK_RETURN + 256): return ImGuiKey_KeypadEnter;
        case VK_SHIFT: return ImGuiKey_LeftShift;
        case VK_LSHIFT: return ImGuiKey_LeftShift;
        case VK_CONTROL: return ImGuiKey_LeftCtrl;
        case VK_LCONTROL: return ImGuiKey_LeftCtrl;
        case VK_MENU: return ImGuiKey_LeftAlt;
        case VK_LMENU: return ImGuiKey_LeftAlt;
        case VK_LWIN: return ImGuiKey_LeftSuper;
        case VK_RSHIFT: return ImGuiKey_RightShift;
        case VK_RCONTROL: return ImGuiKey_RightCtrl;
        case VK_RMENU: return ImGuiKey_RightAlt;
        case VK_RWIN: return ImGuiKey_RightSuper;
        case VK_APPS: return ImGuiKey_Menu;
        case '0': return ImGuiKey_0;
        case '1': return ImGuiKey_1;
        case '2': return ImGuiKey_2;
        case '3': return ImGuiKey_3;
        case '4': return ImGuiKey_4;
        case '5': return ImGuiKey_5;
        case '6': return ImGuiKey_6;
        case '7': return ImGuiKey_7;
        case '8': return ImGuiKey_8;
        case '9': return ImGuiKey_9;
        case 'A': return ImGuiKey_A;
        case 'B': return ImGuiKey_B;
        case 'C': return ImGuiKey_C;
        case 'D': return ImGuiKey_D;
        case 'E': return ImGuiKey_E;
        case 'F': return ImGuiKey_F;
        case 'G': return ImGuiKey_G;
        case 'H': return ImGuiKey_H;
        case 'I': return ImGuiKey_I;
        case 'J': return ImGuiKey_J;
        case 'K': return ImGuiKey_K;
        case 'L': return ImGuiKey_L;
        case 'M': return ImGuiKey_M;
        case 'N': return ImGuiKey_N;
        case 'O': return ImGuiKey_O;
        case 'P': return ImGuiKey_P;
        case 'Q': return ImGuiKey_Q;
        case 'R': return ImGuiKey_R;
        case 'S': return ImGuiKey_S;
        case 'T': return ImGuiKey_T;
        case 'U': return ImGuiKey_U;
        case 'V': return ImGuiKey_V;
        case 'W': return ImGuiKey_W;
        case 'X': return ImGuiKey_X;
        case 'Y': return ImGuiKey_Y;
        case 'Z': return ImGuiKey_Z;
        case VK_F1: return ImGuiKey_F1;
        case VK_F2: return ImGuiKey_F2;
        case VK_F3: return ImGuiKey_F3;
        case VK_F4: return ImGuiKey_F4;
        case VK_F5: return ImGuiKey_F5;
        case VK_F6: return ImGuiKey_F6;
        case VK_F7: return ImGuiKey_F7;
        case VK_F8: return ImGuiKey_F8;
        case VK_F9: return ImGuiKey_F9;
        case VK_F10: return ImGuiKey_F10;
        case VK_F11: return ImGuiKey_F11;
        case VK_F12: return ImGuiKey_F12;
        default: return ImGuiKey_None;
    }
}

LRESULT windowEventHandler(HWND hwnd, UINT eventType, WPARAM wParam, LPARAM lParam) {
    LRESULT result = 0;
    switch (eventType) {
        default: {
            result = DefWindowProcA(hwnd, eventType, wParam, lParam);
        } break;
        case WM_ACTIVATEAPP: {
#ifdef EDITOR
            editor.camera.moving = false;
            windowHideCursor(false);
#endif
        } break;
        case WM_SHOWWINDOW:
        case WM_SIZE:
        case WM_MOVE: {
            uint prevRenderW = renderW;
            uint prevRenderH = renderH;
            windowUpdateSizes();
            if (renderW == 0 || renderH == 0) {
                renderW = prevRenderW;
                renderH = prevRenderH;
            }
            if (d3d.swapChain && renderW > 0 && renderH > 0 && (prevRenderW != renderW || prevRenderH != renderH)) {
                d3dWaitFence(&d3d.renderFence);
                d3dWaitFence(&d3d.renderFencePrevFrame);
                for (ID3D12Resource* image : d3d.swapChainImages) {
                    image->Release();
                }
                assert(SUCCEEDED(d3d.swapChain->ResizeBuffers(countof(d3d.swapChainImages), renderW, renderH, d3d.swapChainFormat, 0)));
                for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
                    ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
                    assert(SUCCEEDED(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image))));
                    (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
                    d3d.device->CreateRenderTargetView(*image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
                }
                {
                    D3D12_RESOURCE_DESC renderTextureDesc = d3d.renderTexture->GetResource()->GetDesc();
                    D3D12_RESOURCE_DESC renderTexturePrevFrameDesc = d3d.renderTexturePrevFrame->GetResource()->GetDesc();
                    D3D12_RESOURCE_DESC pathTracerAccumulationTextureDesc = d3d.pathTracerAccumulationTexture->GetResource()->GetDesc();
                    D3D12_RESOURCE_DESC depthTextureDesc = d3d.depthTexture->GetResource()->GetDesc();
                    D3D12_RESOURCE_DESC motionVectorTextureDesc = d3d.motionVectorTexture->GetResource()->GetDesc();

                    d3d.renderTexture->Release();
                    d3d.renderTexturePrevFrame->Release();
                    d3d.pathTracerAccumulationTexture->Release();
                    d3d.depthTexture->Release();
                    d3d.motionVectorTexture->Release();

                    renderTextureDesc.Width = renderW, renderTextureDesc.Height = renderH;
                    renderTexturePrevFrameDesc.Width = renderW, renderTexturePrevFrameDesc.Height = renderH;
                    pathTracerAccumulationTextureDesc.Width = renderW, pathTracerAccumulationTextureDesc.Height = renderH;
                    depthTextureDesc.Width = renderW, depthTextureDesc.Height = renderH;
                    motionVectorTextureDesc.Width = renderW, motionVectorTextureDesc.Height = renderH;

                    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr)));
                    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTexturePrevFrameDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexturePrevFrame, {}, nullptr)));
                    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &pathTracerAccumulationTextureDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &d3d.pathTracerAccumulationTexture, {}, nullptr)));
                    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &depthTextureDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &d3d.depthTexture, {}, nullptr)));
                    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &motionVectorTextureDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &d3d.motionVectorTexture, {}, nullptr)));

                    d3d.renderTexture->GetResource()->SetName(L"renderTexture");
                    d3d.renderTexturePrevFrame->GetResource()->SetName(L"renderTexturePrevFrame");
                    d3d.pathTracerAccumulationTexture->GetResource()->SetName(L"pathTracerAccumulationTexture");
                    d3d.depthTexture->GetResource()->SetName(L"depthTexture");
                    d3d.motionVectorTexture->GetResource()->SetName(L"motionVectorTexture");
                }
            }
        } break;
        case WM_CLOSE:
        case WM_QUIT: {
            quit = true;
        } break;
        case WM_KEYDOWN:
        case WM_KEYUP: {
            ImGui::GetIO().AddKeyEvent(virtualKeytoImGuiKey(wParam), eventType == WM_KEYDOWN);
        } break;
        case WM_SYSKEYDOWN:
        case WM_SYSKEYUP: {
            ImGui::GetIO().AddKeyEvent(virtualKeytoImGuiKey(wParam), eventType == WM_SYSKEYDOWN);
        } break;
        case WM_CHAR: {
            ImGui::GetIO().AddInputCharacter(LOWORD(wParam));
        } break;
        case WM_INPUT_DEVICE_CHANGE: {
            if (wParam == GIDC_ARRIVAL) {
                RID_DEVICE_INFO info;
                uint infoSize = sizeof(info);
                GetRawInputDeviceInfoA((HANDLE)lParam, RIDI_DEVICEINFO, &info, &infoSize);
                if (info.dwType == RIM_TYPEHID && info.hid.dwVendorId == 0x054c && info.hid.dwProductId == 0x0ce6) {
                    controllerDualSenseHID = (HANDLE)lParam;
                }
            }
        } break;
        case WM_INPUT: {
            static char rawInputBuffer[1024];
            uint rawInputBufferSize = sizeof(rawInputBuffer);
            if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, rawInputBuffer, &rawInputBufferSize, sizeof(RAWINPUTHEADER)) < sizeof(rawInputBuffer)) {
                RAWINPUT* rawInput = (RAWINPUT*)rawInputBuffer;
                if (rawInput->header.dwType == RIM_TYPEMOUSE) {
                    mouseDeltaRaw.x += rawInput->data.mouse.lLastX;
                    mouseDeltaRaw.y += rawInput->data.mouse.lLastY;
                }
                else if (rawInput->header.dwType == RIM_TYPEHID && rawInput->header.hDevice == controllerDualSenseHID) {
                    for (uint packetIndex = 0; packetIndex < rawInput->data.hid.dwCount; packetIndex++) {
                        // controller.getStateDualSense(&rawInput->data.hid.bRawData[rawInput->data.hid.dwSizeHid * packetIndex], rawInput->data.hid.dwSizeHid);
                    }
                }
            }
        } break;
        case WM_MOUSEMOVE: {
            ImGui::GetIO().AddMousePosEvent((float)(GET_X_LPARAM(lParam)), (float)(GET_Y_LPARAM(lParam)));
        } break;
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP: {
            ImGui::GetIO().AddMouseButtonEvent(0, eventType == WM_LBUTTONDOWN);
        } break;
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP: {
            ImGui::GetIO().AddMouseButtonEvent(1, eventType == WM_RBUTTONDOWN);
        } break;
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP: {
            ImGui::GetIO().AddMouseButtonEvent(2, eventType == WM_MBUTTONDOWN);
        } break;
        case WM_MOUSEWHEEL: {
            float wheelDelta = (float)(GET_WHEEL_DELTA_WPARAM(wParam)) / (float)WHEEL_DELTA;
            mouseWheel += wheelDelta;
            ImGui::GetIO().AddMouseWheelEvent(0, wheelDelta);
        } break;
    }
    return result;
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    assert(SetCurrentDirectoryW(exeDir.c_str()));
    settingsInit();
    windowInit();
    windowShow();
#ifndef EDITOR
    windowHideCursor(true);
#endif
    imguiInit();
    gameInputInit();
    d3dInit();
    d3dApplySettings();
    dlssInit();
    physxInit();
    worldInit();
    gameReadSave();
    loadSimpleAssets();
    assert(QueryPerformanceFrequency(&perfFrequency));
    while (!quit) {
        QueryPerformanceCounter(&perfCounterStart);
        d3dCompileShaders();
        liveReloadFuncs();
        mouseDeltaRaw = {0, 0};
        mouseWheel = 0;
        controllerGetState();
        MSG windowMsg;
        while (PeekMessageA(&windowMsg, (HWND)window.hwnd, 0, 0, PM_REMOVE)) {
            TranslateMessage(&windowMsg);
            DispatchMessageA(&windowMsg);
        }
        update();
        d3dRender();
        FrameMark;
        QueryPerformanceCounter(&perfCounterEnd);
        frameTime = (float)((double)(perfCounterEnd.QuadPart - perfCounterStart.QuadPart) / (double)perfFrequency.QuadPart);
    }
    editorSave();
    gameSave();
    settingsSave();
    return EXIT_SUCCESS;
}
