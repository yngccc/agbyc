#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <atlbase.h>
#include <atlconv.h>
#include <cderr.h>
#include <commdlg.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <shellscalingapi.h>
#include <userenv.h>
#include <windows.h>
#include <windowsx.h>
#include <xinput.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <stack>
#include <streambuf>
#include <string>
#include <vector>

#define _XM_SSE4_INTRINSICS_
#include <directXMath.h>
#include <directXTex.h>
using namespace DirectX;

#define RYML_SINGLE_HDR_DEFINE_NOW
#include <rapidyaml/rapidyaml-0.5.0.hpp>

#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>

// #define STB_IMAGE_IMPLEMENTATION
// #include <stb/stb_image.h>
// #define STB_DS_IMPLEMENTATION
// #include <stb/stb_ds.h>

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_OBSOLETE_KEYIO
#define IMGUI_USE_STB_SPRINTF
#define IMGUI_STB_SPRINTF_FILENAME <stb/stb_sprintf.h>
#include <imgui/imgui.cpp>
#include <imgui/imgui_draw.cpp>
#include <imgui/imgui_tables.cpp>
#include <imgui/imgui_widgets.cpp>
#include <imgui/imguizmo.cpp>
#undef snprintf
#undef vsnprintf

#include <d3d12ma/d3d12MemAlloc.cpp>

#define TRACY_ENABLE
#include <tracy/tracy/tracy.hpp>
#include <tracy/tracyclient.cpp>

typedef int8_t int8;
typedef int16_t int16;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint64_t uint64;
typedef uint32_t uint;

static const float e = 2.71828182845904523536f;
static const float pi = 3.14159265358979323846f;
static const float sqrt2 = 1.41421356237309504880f;

#define kilobytes(n) (1024 * (n))
#define megabytes(n) (1024 * 1024 * (n))
#define gigabytes(n) (1024 * 1024 * 1024 * (n))
#define radian(d) (d * (pi / 180.0f))
#define degree(r) (r * (180.0f / pi))

#undef assert
#define assert(expr) \
    if (!(expr)) abort();
#ifdef _DEBUG
#define assertDebug(expr) \
    if (!(expr)) abort();
#else
#define assertDebug(expr)
#endif

template <typename T, uint N>
constexpr uint countof(const T (&)[N]) { return N; }

template <typename T>
uint64 vectorSizeof(const std::vector<T>& v) { return v.size() * sizeof(T); }

template <typename T, typename T2>
T align(T x, T2 n) {
    T remainder = x % (T)n;
    return remainder == 0 ? x : x + ((T)n - remainder);
}

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
    uint x = 0, y = 0, z = 0, w = 0;
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct float2 {
    float x = 0, y = 0;

    float2() = default;
    float2(float x, float y) : x(x), y(y) {}
    bool operator==(float2 v) const { return x == v.x && y == v.y; }
    bool operator!=(float2 v) const { return x != v.x || y != v.y; }
    float2 operator*(float v) const { return float2(x * v, y * v); }
    float2 operator/(float v) const { return float2(x / v, y / v); }
    void operator<<(ryml::ConstNodeRef node) { node[0] >> x, node[1] >> y; }
    void operator>>(ryml::NodeRef node) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << x, node.append_child() << y; };
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
    void operator<<(ryml::ConstNodeRef node) { node[0] >> x, node[1] >> y, node[2] >> z; }
    void operator>>(ryml::NodeRef node) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << x, node.append_child() << y, node.append_child() << z; };
    float3 operator+(float3 v) const { return float3(x + v.x, y + v.y, z + v.z); }
    void operator+=(float3 v) { x += v.x, y += v.y, z += v.z; }
    float3 operator-() const { return float3(-x, -y, -z); }
    float3 operator-(float3 v) const { return float3(x - v.x, y - v.y, z - v.z); }
    void operator-=(float3 v) { x -= v.x, y -= v.y, z -= v.z; }
    float3 operator*(float scale) const { return float3(x * scale, y * scale, z * scale); }
    void operator*=(float scale) { x *= scale, y *= scale, z *= scale; }
    float3 operator/(float scale) const { return float3(x / scale, y / scale, z / scale); }
    void operator/=(float scale) { x /= scale, y /= scale, z /= scale; }
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, 0); }
    std::string toString() const { return std::format("[{}, {}, {}]", x, y, z); };
    float dot(float3 v) const { return x * v.x + y * v.y + z * v.z; }
    float3 cross(float3 v) const { return float3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    float length() const { return sqrtf(x * x + y * y + z * z); };
    float3 normalize() const {
        float l = length();
        return (l > 0) ? float3(x / l, y / l, z / l) : float3(x, y, z);
    }
};

struct float4 {
    float x = 0, y = 0, z = 0, w = 1;

    float4() = default;
    float4(const float* v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    float4(const XMVECTOR& v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)), w(XMVectorGetW(v)) {}
    void operator=(const XMVECTOR& v) { x = XMVectorGetX(v), y = XMVectorGetY(v), z = XMVectorGetZ(v), w = XMVectorGetW(v); }
    void operator=(const float3& v) { x = v.x, y = v.y, z = v.z, w = 0; }
    void operator<<(ryml::ConstNodeRef node) { node[0] >> x, node[1] >> y, node[2] >> z, node[3] >> w; }
    void operator>>(ryml::NodeRef node) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << x, node.append_child() << y, node.append_child() << z, node.append_child() << w; }
    float3 xyz() const { return float3(x, y, z); };
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, w); }
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); };
};

struct Transform {
    float3 s = {1, 1, 1};
    float4 r = {0, 0, 0, 1};
    float3 t = {0, 0, 0};

    void operator<<(ryml::ConstNodeRef node) { s << node["scale"], r << node["rotate"], t << node["translate"]; }
    void operator>>(ryml::NodeRef node) { s >> node["scale"], r >> node["rotate"], t >> node["translate"]; }
    XMMATRIX toMat() const { return XMMatrixAffineTransformation(s.toXMVector(), XMVectorSet(0, 0, 0, 0), r.toXMVector(), t.toXMVector()); }
};

float3 lerp(const float3& a, const float3& b, float t) { return a + ((b - a) * t); };

float4 slerp(const float4& a, const float4& b, float t) { return float4(XMQuaternionSlerp(a.toXMVector(), b.toXMVector(), t)); };

std::string toString(const XMVECTOR& vec) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n", XMVectorGetX(vec), XMVectorGetY(vec), XMVectorGetZ(vec), XMVectorGetW(vec));
}

std::string toString(const XMMATRIX& mat) {
    return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n",
                       XMVectorGetX(mat.r[0]), XMVectorGetX(mat.r[1]), XMVectorGetX(mat.r[2]), XMVectorGetX(mat.r[3]),
                       XMVectorGetY(mat.r[0]), XMVectorGetY(mat.r[1]), XMVectorGetY(mat.r[2]), XMVectorGetY(mat.r[3]),
                       XMVectorGetZ(mat.r[0]), XMVectorGetZ(mat.r[1]), XMVectorGetZ(mat.r[2]), XMVectorGetZ(mat.r[3]),
                       XMVectorGetW(mat.r[0]), XMVectorGetW(mat.r[1]), XMVectorGetW(mat.r[2]), XMVectorGetW(mat.r[3]));
}

std::filesystem::path exeDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, countof(buf));
    assert(n < countof(buf));
    std::filesystem::path path(buf);
    return path.parent_path();
}();

std::filesystem::path assetsDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, countof(buf));
    assert(n < countof(buf));
    std::filesystem::path path(buf);
    return path.parent_path().parent_path().parent_path() / "assets";
}();

bool fileExists(const std::filesystem::path& path) {
    DWORD dwAttrib = GetFileAttributesW(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

std::string fileReadStr(const std::filesystem::path& path) {
    std::ifstream t(path);
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return str;
}

std::vector<uint8> fileRead(const std::filesystem::path& path) {
    HANDLE hwnd = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    assert(hwnd != INVALID_HANDLE_VALUE);
    DWORD size = GetFileSize(hwnd, nullptr);
    assert(size != INVALID_FILE_SIZE);
    std::vector<uint8> data(size);
    DWORD byteRead;
    assert(ReadFile(hwnd, data.data(), size, &byteRead, nullptr) && byteRead == size);
    CloseHandle(hwnd);
    return data;
}

bool fileWriteStr(const std::filesystem::path& path, const std::string& str) {
    HANDLE hwnd = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hwnd != INVALID_HANDLE_VALUE) {
        DWORD bytesWritten = 0;
        if (WriteFile(hwnd, str.c_str(), (DWORD)str.length(), &bytesWritten, nullptr)) {
            CloseHandle(hwnd);
            return true;
        } else {
            CloseHandle(hwnd);
            return false;
        }
    }
    return false;
}

bool commandLineContain(int argc, char** argv, const char* str) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], str)) { return true; }
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

enum WindowMode {
    WindowModeWindowed,
    WindowModeBorderless,
    WindowModeFullscreen
};

struct Settings {
    WindowMode windowMode = WindowModeWindowed;
    uint windowX = 0, windowY = 0;
    uint windowW = 1920, windowH = 1080;
    uint renderW = 1920, renderH = 1080;
    DXGI_RATIONAL refreshRate = {60, 1};
    bool hdr = false;

    void load() {
        if (fileExists(exeDir / "settings.yaml")) {
            std::string yamlStr = fileReadStr(exeDir / "settings.yaml");
            ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
            ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
            yamlRoot["hdr"] >> hdr;
            yamlRoot["windowX"] >> windowX;
            yamlRoot["windowY"] >> windowY;
            yamlRoot["windowW"] >> windowW;
            yamlRoot["windowH"] >> windowH;
        }
    }

    void save() {
        ryml::Tree yamlTree;
        ryml::NodeRef yamlRoot = yamlTree.rootref();
        yamlRoot |= ryml::MAP;
        yamlRoot["hdr"] << hdr;
        yamlRoot["windowX"] << windowX;
        yamlRoot["windowY"] << windowY;
        yamlRoot["windowW"] << windowW;
        yamlRoot["windowH"] << windowH;
        std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
        assert(fileWriteStr(exeDir / "settings.yaml", yamlStr));
    }
};

static Settings settings = {};

struct Window {
    HWND hwnd;

    void init() {
        assert(SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) == S_OK);

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

        DWORD windowStyle = WS_OVERLAPPEDWINDOW;
        hwnd = CreateWindowExA(0, windowClass.lpszClassName, nullptr, windowStyle, settings.windowX, settings.windowY, settings.windowW, settings.windowH, nullptr, nullptr, instanceHandle, nullptr);
        assert(hwnd);

        updateSizes();

        RAWINPUTDEVICE rawInputDevice = {.usUsagePage = 0x01, .usUsage = 0x02};
        assert(RegisterRawInputDevices(&rawInputDevice, 1, sizeof(rawInputDevice)));
    }

    void show() {
        ShowWindow(hwnd, SW_SHOW);
    }

    void updateSizes() {
        RECT windowRect;
        RECT clientRect;
        assert(GetWindowRect(hwnd, &windowRect));
        assert(GetClientRect(hwnd, &clientRect));
        settings.windowX = windowRect.left;
        settings.windowY = windowRect.top;
        settings.windowW = windowRect.right - windowRect.left;
        settings.windowH = windowRect.bottom - windowRect.top;
        settings.renderW = clientRect.right - clientRect.left;
        settings.renderH = clientRect.bottom - clientRect.top;
    }
};

static Window window = {};

struct DisplayMode {
    uint resolutionWidth;
    uint resolutionHeight;
    std::vector<DXGI_RATIONAL> refreshRates;

    void addRefreshRate(DXGI_RATIONAL rate) {
        for (DXGI_RATIONAL& r : refreshRates) {
            if (r.Numerator == rate.Numerator && r.Denominator == rate.Denominator) { return; }
        }
        refreshRates.push_back(rate);
    }
};

void d3dMessageCallback(D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context) {
    if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR) {
        __debugbreak();
    }
}

#include "sharedStructs.h"

struct D3DDescriptor {
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

struct D3D {
    IDXGIOutput6* dxgiOutput;
    IDXGIAdapter4* dxgiAdapter;
    std::vector<DisplayMode> displayModes;
    ID3D12Device5* device;

    // bool gpuUploadHeapSupported;

    ID3D12CommandQueue* graphicsQueue;
    ID3D12Fence* graphicsQueueFence;
    HANDLE graphicsQueueFenceEvent;
    uint64 graphicsQueueFenceCounter;
    ID3D12CommandAllocator* graphicsCmdAllocator;
    ID3D12GraphicsCommandList4* graphicsCmdList;

    ID3D12CommandQueue* transferQueue;
    ID3D12Fence* transferQueueFence;
    HANDLE transferQueueFenceEvent;
    uint64 transferQueueFenceCounter;
    ID3D12CommandAllocator* transferCmdAllocator;
    ID3D12GraphicsCommandList4* transferCmdList;

    IDXGISwapChain4* swapChain;
    DXGI_FORMAT swapChainFormat;
    ID3D12Resource* swapChainImages[2];
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainImageRTVDescriptors[2];

    ID3D12DescriptorHeap* rtvDescriptorHeap;
    uint rtvDescriptorSize;
    uint rtvDescriptorCount;
    ID3D12DescriptorHeap* cbvSrvUavDescriptorHeap;
    uint cbvSrvUavDescriptorSize;
    uint cbvSrvUavDescriptorCapacity;
    uint cbvSrvUavDescriptorCount;

    D3D12MA::Allocator* allocator;

    D3D12MA::Allocation* stagingBuffer;
    uint8* stagingBufferPtr;
    uint stagingBufferOffset = 0;

    D3D12MA::Allocation* constantBuffer;
    uint8* constantBufferPtr;
    uint constantBufferOffset = 0;

    D3D12MA::Allocation* readBackUAVBuffer;
    D3D12MA::Allocation* readBackBuffer;
    ReadBackBuffer* readBackBufferPtr;

    D3D12MA::Allocation* renderTexture;
    DXGI_FORMAT renderTextureFormat;

    D3D12MA::Allocation* imguiImage;
    D3D12MA::Allocation* imguiVertexBuffer;
    uint8* imguiVertexBufferPtr;
    D3D12MA::Allocation* imguiIndexBuffer;
    uint8* imguiIndexBufferPtr;

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
    uint8* collisionQueriesBufferPtr;
    D3D12MA::Allocation* collisionQueryResultsUavBuffer;
    D3D12MA::Allocation* collisionQueryResultsBuffer;
    uint8* collisionQueryResultsBufferPtr;

    ID3D12PipelineState* vertexSkinningPSO;
    ID3D12RootSignature* vertexSkinningRootSig;

    ID3D12StateObject* renderScenePSO;
    ID3D12StateObjectProperties* renderSceneProps;
    ID3D12RootSignature* renderSceneRootSig;
    void* renderSceneRayGenID;
    void* renderScenePrimaryRayMissID;
    void* renderScenePrimaryRayHitGroupID;
    void* renderSceneSecondaryRayMissID;
    void* renderSceneSecondaryRayHitGroupID;

    ID3D12StateObject* collisionDetection;
    ID3D12StateObjectProperties* collisionDetectionProps;
    ID3D12RootSignature* collisionDetectionRootSig;
    void* collisionDetectionRayGenID;
    void* collisionDetectionMissID;
    void* collisionDetectionHitGroupID;

    ID3D12PipelineState* postProcessPSO;
    ID3D12RootSignature* postProcessRootSig;

    ID3D12PipelineState* imguiPSO;
    ID3D12RootSignature* imguiRootSig;

    void init(bool debug) {
        uint factoryFlags = 0;
        if (debug) {
            factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
            ID3D12Debug1* debug;
            assert(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)) == S_OK);
            debug->EnableDebugLayer();
            // debug->SetEnableGPUBasedValidation(true);
            // debug->SetEnableSynchronizedCommandQueueValidation(true);
        }

        IDXGIFactory7* dxgiFactory = nullptr;
        DXGI_ADAPTER_DESC dxgiAdapterDesc = {};
        DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};

        assert(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&dxgiFactory)) == S_OK);
        assert(dxgiFactory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&dxgiAdapter)) == S_OK);
        assert(D3D12CreateDevice(dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)) == S_OK);
        if (debug) {
            ID3D12InfoQueue1* infoQueue;
            assert(device->QueryInterface(IID_PPV_ARGS(&infoQueue)) == S_OK);
            DWORD callbackCookie;
            assert(infoQueue->RegisterMessageCallback(d3dMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &callbackCookie) == S_OK);
        }
        assert(dxgiAdapter->GetDesc(&dxgiAdapterDesc) == S_OK);
        assert(dxgiAdapter->EnumOutputs(0, (IDXGIOutput**)&dxgiOutput) == S_OK);
        assert(dxgiOutput->GetDesc1(&dxgiOutputDesc) == S_OK);
        settings.hdr = (dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS resourceBindingTier = {};
            D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 rayTracing = {};
            // D3D12_FEATURE_DATA_D3D12_OPTIONS16 gpuUploadHeap = {};
            assert(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &resourceBindingTier, sizeof(resourceBindingTier)) == S_OK);
            assert(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)) == S_OK);
            assert(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &rayTracing, sizeof(rayTracing)) == S_OK);
            // assert(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &gpuUploadHeap, sizeof(gpuUploadHeap)) == S_OK);
            assert(resourceBindingTier.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
            assert(shaderModel.HighestShaderModel == D3D_SHADER_MODEL_6_6);
            assert(rayTracing.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
            // gpuUploadHeapSupported = gpuUploadHeap.GPUUploadHeapSupported;
        }
        {
            D3D12_COMMAND_QUEUE_DESC graphicsQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
            assert(device->CreateCommandQueue(&graphicsQueueDesc, IID_PPV_ARGS(&graphicsQueue)) == S_OK);
            D3D12_COMMAND_QUEUE_DESC transferQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
            assert(device->CreateCommandQueue(&transferQueueDesc, IID_PPV_ARGS(&transferQueue)) == S_OK);

            assert(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&graphicsCmdAllocator)) == S_OK);
            assert(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&transferCmdAllocator)) == S_OK);

            assert(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, graphicsCmdAllocator, nullptr, IID_PPV_ARGS(&graphicsCmdList)) == S_OK);
            assert(graphicsCmdList->Close() == S_OK);
            assert(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, transferCmdAllocator, nullptr, IID_PPV_ARGS(&transferCmdList)) == S_OK);
            assert(transferCmdList->Close() == S_OK);

            assert(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&graphicsQueueFence)) == S_OK);
            graphicsQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
            assert(graphicsQueueFenceEvent);
            assert(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&transferQueueFence)) == S_OK);
            transferQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
            assert(transferQueueFenceEvent);
        }
        {
            swapChainFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
            uint dxgiModeCount = 0;
            dxgiOutput->GetDisplayModeList(swapChainFormat, 0, &dxgiModeCount, nullptr);
            std::vector<DXGI_MODE_DESC> dxgiModes(dxgiModeCount);
            dxgiOutput->GetDisplayModeList(swapChainFormat, 0, &dxgiModeCount, dxgiModes.data());
            for (DXGI_MODE_DESC& dxgiMode : dxgiModes) {
                bool hasResolution = false;
                for (DisplayMode& mode : displayModes) {
                    if (mode.resolutionWidth == dxgiMode.Width && mode.resolutionHeight == dxgiMode.Height) {
                        hasResolution = true;
                        mode.addRefreshRate(dxgiMode.RefreshRate);
                        break;
                    }
                }
                if (!hasResolution) displayModes.push_back(DisplayMode{dxgiMode.Width, dxgiMode.Height, {dxgiMode.RefreshRate}});
            }
            DXGI_SWAP_CHAIN_DESC1 desc = {
                .Width = (uint)settings.renderW,
                .Height = (uint)settings.renderH,
                .Format = swapChainFormat,
                .SampleDesc = {.Count = 1},
                .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER,
                .BufferCount = countof(swapChainImages),
                .Scaling = DXGI_SCALING_NONE,
                .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
                .AlphaMode = DXGI_ALPHA_MODE_IGNORE,
                .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH,
            };
            assert(dxgiFactory->CreateSwapChainForHwnd(graphicsQueue, window.hwnd, &desc, nullptr, nullptr, (IDXGISwapChain1**)&swapChain) == S_OK);

            DXGI_COLOR_SPACE_TYPE colorSpace = settings.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
            assert(swapChain->SetColorSpace1(colorSpace) == S_OK);
            for (uint imageIndex = 0; imageIndex < countof(swapChainImages); imageIndex++) {
                ID3D12Resource** image = &swapChainImages[imageIndex];
                assert(swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image)) == S_OK);
                (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
            }

            dxgiFactory->MakeWindowAssociation(window.hwnd, DXGI_MWA_NO_WINDOW_CHANGES); // disable alt-enter
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 16, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
            assert(device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&rtvDescriptorHeap)) == S_OK);
            rtvDescriptorCount = 0;
            rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            for (uint imageIndex = 0; imageIndex < countof(swapChainImages); imageIndex++) {
                ID3D12Resource** image = &swapChainImages[imageIndex];
                uint offset = rtvDescriptorSize * rtvDescriptorCount;
                swapChainImageRTVDescriptors[imageIndex] = {rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
                device->CreateRenderTargetView(*image, nullptr, swapChainImageRTVDescriptors[imageIndex]);
                rtvDescriptorCount += 1;
            }

            cbvSrvUavDescriptorCapacity = 1024;
            cbvSrvUavDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = cbvSrvUavDescriptorCapacity, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
            assert(device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&cbvSrvUavDescriptorHeap)) == S_OK);
        }
        {
            D3D12MA::ALLOCATOR_DESC allocatorDesc = {.Flags = D3D12MA::ALLOCATOR_FLAG_NONE, .pDevice = device, .pAdapter = dxgiAdapter};
            assert(D3D12MA::CreateAllocator(&allocatorDesc, &allocator) == S_OK);
        }
        {
            struct BufferDesc {
                D3D12MA::Allocation** buffer;
                uint8** bufferPtr;
                uint size;
                D3D12_HEAP_TYPE heapType;
                D3D12_RESOURCE_FLAGS flags;
                D3D12_RESOURCE_STATES initState;
                const wchar_t* name;
            } descs[] = {
                {&stagingBuffer, &stagingBufferPtr, megabytes(256), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"stagingBuffer"},
                {&constantBuffer, &constantBufferPtr, megabytes(4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer"},
                {&tlasInstancesBuildInfosBuffer, &tlasInstancesBuildInfosBufferPtr, megabytes(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfosBuffer"},
                {&tlasInstancesInfosBuffer, &tlasInstancesInfosBufferPtr, megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesInfosBuffer"},
                {&blasGeometriesInfosBuffer, &blasGeometriesInfosBufferPtr, megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"blasGeometriesInfosBuffer"},
                {&tlasBuffer, nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlasBuffer"},
                {&tlasScratchBuffer, nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratchBuffer"},
                {&imguiVertexBuffer, &imguiVertexBufferPtr, megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer"},
                {&imguiIndexBuffer, &imguiIndexBufferPtr, megabytes(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer"},
                {&readBackUAVBuffer, nullptr, megabytes(2), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"readBackUavBuffer"},
                {&readBackBuffer, (uint8**)&readBackBufferPtr, megabytes(2), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"readBackBuffer"},
                {&collisionQueriesBuffer, &collisionQueriesBufferPtr, megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueriesBuffer"},
                {&collisionQueryResultsUavBuffer, nullptr, megabytes(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUavBuffer"},
                {&collisionQueryResultsBuffer, &collisionQueryResultsBufferPtr, megabytes(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResultsBuffer"},
            };
            for (BufferDesc& desc : descs) {
                D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = desc.size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = desc.flags};
                D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = desc.heapType};
                assert(allocator->CreateResource(&allocationDesc, &bufferDesc, desc.initState, nullptr, desc.buffer, {}, nullptr) == S_OK);
                (*desc.buffer)->GetResource()->SetName(desc.name);
                if (desc.bufferPtr) {
                    assert((*desc.buffer)->GetResource()->Map(0, nullptr, (void**)desc.bufferPtr) == S_OK);
                }
            }
        }
        {
            D3D12_RESOURCE_DESC renderTextureDesc = {
                .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                .Width = settings.renderW,
                .Height = settings.renderH,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
                .SampleDesc = {.Count = 1},
                .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
                .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };

            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            assert(allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &renderTexture, {}, nullptr) == S_OK);
            renderTexture->GetResource()->SetName(L"renderTexture");
            renderTextureFormat = renderTextureDesc.Format;
        }
        {
            assert(ImGui::CreateContext());
            ImGui::StyleColorsDark();
            // ImGui::StyleColorsLight();
            // ImGui::StyleColorsClassic();
            ImGuiIO& io = ImGui::GetIO();
            io.IniFilename = "imgui.ini";
            io.FontGlobalScale = (float)settings.renderH / 1000.0f;
            assert(io.Fonts->AddFontDefault());

            transferQueueStartRecording();
            stagingBufferOffset = 0;
            {
                uint8* imguiTextureData;
                int imguiTextureWidth, imguiTextureHeight;
                ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&imguiTextureData, &imguiTextureWidth, &imguiTextureHeight);
                D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint64)imguiTextureWidth, .Height = (uint)imguiTextureHeight, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
                D3D12_SUBRESOURCE_DATA data = {.pData = imguiTextureData, .RowPitch = imguiTextureWidth * 4, .SlicePitch = imguiTextureWidth * imguiTextureHeight * 4};
                imguiImage = create2DImage(desc, &data);
            }
            {
                uint8_4 defaultMaterialBaseColorImageData[4] = {{255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}};
                D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
                D3D12_SUBRESOURCE_DATA data = {.pData = defaultMaterialBaseColorImageData, .RowPitch = 8, .SlicePitch = 16};
                defaultMaterialBaseColorImage = create2DImage(desc, &data);
                defaultMaterialBaseColorImageSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
            }
            D3D12_RESOURCE_BARRIER barriers[2] = {
                {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = imguiImage->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}},
                {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = defaultMaterialBaseColorImage->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}},
            };
            transferCmdList->ResourceBarrier(2, barriers);
            transferQueueSubmitRecording();
            transferQueueWait();
        }

        compilePipelines();
    }

    void compilePipelines() {
        {
            std::vector<uint8> rtByteCode = fileRead(exeDir / "renderScene.cso");
            assert(device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&renderSceneRootSig)) == S_OK);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"primaryRayMiss"}, {L"primaryRayHitGroup"}, {L"primaryRayClosestHit"}, {L"secondaryRayMiss"}, {L"secondaryRayHitGroup"}, {L"secondaryRayClosestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&renderScenePSO)) == S_OK);
            assert(renderScenePSO->QueryInterface(IID_PPV_ARGS(&renderSceneProps)) == S_OK);
            assert(renderSceneRayGenID = renderSceneProps->GetShaderIdentifier(L"rayGen"));
            assert(renderScenePrimaryRayMissID = renderSceneProps->GetShaderIdentifier(L"primaryRayMiss"));
            assert(renderScenePrimaryRayHitGroupID = renderSceneProps->GetShaderIdentifier(L"primaryRayHitGroup"));
            assert(renderSceneSecondaryRayMissID = renderSceneProps->GetShaderIdentifier(L"secondaryRayMiss"));
            assert(renderSceneSecondaryRayHitGroupID = renderSceneProps->GetShaderIdentifier(L"secondaryRayHitGroup"));
        }
        {
            std::vector<uint8> rtByteCode = fileRead(exeDir / "collisionDetection.cso");
            assert(device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&collisionDetectionRootSig)) == S_OK);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"miss"}, {L"hitGroup"}, {L"closestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&collisionDetection)) == S_OK);
            assert(collisionDetection->QueryInterface(IID_PPV_ARGS(&collisionDetectionProps)) == S_OK);
            assert(collisionDetectionRayGenID = collisionDetectionProps->GetShaderIdentifier(L"rayGen"));
            assert(collisionDetectionMissID = collisionDetectionProps->GetShaderIdentifier(L"miss"));
            assert(collisionDetectionHitGroupID = collisionDetectionProps->GetShaderIdentifier(L"hitGroup"));
        }
        {
            std::vector<uint8> vsByteCode = fileRead(exeDir / "postProcessVS.cso");
            std::vector<uint8> psByteCode = fileRead(exeDir / "postProcessPS.cso");
            assert(device->CreateRootSignature(0, psByteCode.data(), psByteCode.size(), IID_PPV_ARGS(&postProcessRootSig)) == S_OK);
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
                .VS = {vsByteCode.data(), vsByteCode.size()},
                .PS = {psByteCode.data(), psByteCode.size()},
                .BlendState = {.RenderTarget = {{.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL}}},
                .SampleMask = 0xffffffff,
                .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_BACK},
                .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                .NumRenderTargets = 1,
                .RTVFormats = {swapChainFormat},
                .SampleDesc = {.Count = 1},
            };
            assert(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&postProcessPSO)) == S_OK);
        }
        {
            std::vector<uint8> vsByteCode = fileRead(exeDir / "ImGuiVS.cso");
            std::vector<uint8> psByteCode = fileRead(exeDir / "ImGuiPS.cso");
            assert(device->CreateRootSignature(0, vsByteCode.data(), vsByteCode.size(), IID_PPV_ARGS(&imguiRootSig)) == S_OK);
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
                .RTVFormats = {swapChainFormat},
                .SampleDesc = {.Count = 1},
            };
            assert(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&imguiPSO)) == S_OK);
        }
        {
            std::vector<uint8> csByteCode = fileRead(exeDir / "vertexSkinning.cso");
            assert(device->CreateRootSignature(0, csByteCode.data(), csByteCode.size(), IID_PPV_ARGS(&vertexSkinningRootSig)) == S_OK);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {.pRootSignature = vertexSkinningRootSig, .CS = {.pShaderBytecode = csByteCode.data(), .BytecodeLength = csByteCode.size()}};
            assert(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&vertexSkinningPSO)) == S_OK);
        }
    }

    void resizeSwapChain(uint width, uint height) {
        graphicsQueueWait();
        for (ID3D12Resource* image : swapChainImages) { image->Release(); }
        assert(swapChain->ResizeBuffers(countof(swapChainImages), width, height, swapChainFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH) == S_OK);
        for (uint imageIndex = 0; imageIndex < countof(swapChainImages); imageIndex++) {
            ID3D12Resource** image = &swapChainImages[imageIndex];
            assert(swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image)) == S_OK);
            (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
            device->CreateRenderTargetView(*image, nullptr, swapChainImageRTVDescriptors[imageIndex]);
        }
        renderTexture->Release();
        D3D12_RESOURCE_DESC renderTextureDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = width, .Height = height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = renderTextureFormat, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        assert(allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &renderTexture, {}, nullptr) == S_OK);
        renderTexture->GetResource()->SetName(L"renderTexture");
    }

    void applySettings() {
        DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};
        assert(dxgiOutput->GetDesc1(&dxgiOutputDesc) == S_OK);
        if (settings.hdr && dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
            assert(swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) == S_OK);
        } else {
            assert(swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) == S_OK);
        }
        if (settings.windowMode == WindowModeWindowed) {
            assert(swapChain->SetFullscreenState(false, nullptr) == S_OK);
            DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
            MONITORINFO mi = {.cbSize = sizeof(mi)};
            assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
            assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW) != 0);
            assert(SetWindowPos(window.hwnd, NULL, settings.windowX, settings.windowY, settings.windowW, settings.windowH, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
        } else if (settings.windowMode == WindowModeBorderless) {
            assert(swapChain->SetFullscreenState(false, nullptr) == S_OK);
            DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
            MONITORINFO mi = {.cbSize = sizeof(mi)};
            assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
            assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW) != 0);
            assert(SetWindowPos(window.hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
        } else if (settings.windowMode == WindowModeFullscreen) {
            DXGI_MODE_DESC dxgiMode = {.Width = settings.windowW, .Height = settings.windowH, .RefreshRate = settings.refreshRate, .Format = swapChainFormat};
            assert(swapChain->ResizeTarget(&dxgiMode) == S_OK);
            assert(swapChain->SetFullscreenState(true, nullptr) == S_OK);
        }
    }

    D3DDescriptor appendCBVDescriptor(D3D12_CONSTANT_BUFFER_VIEW_DESC* constantBufferViewDesc) {
        assert(cbvSrvUavDescriptorCount < cbvSrvUavDescriptorCapacity);
        uint offset = cbvSrvUavDescriptorSize * cbvSrvUavDescriptorCount;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
        device->CreateConstantBufferView(constantBufferViewDesc, cpuHandle);
        cbvSrvUavDescriptorCount++;
        return {cpuHandle, gpuHandle};
    }

    D3DDescriptor appendSRVDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC* resourceViewDesc, ID3D12Resource* resource) {
        assert(cbvSrvUavDescriptorCount < cbvSrvUavDescriptorCapacity);
        uint offset = cbvSrvUavDescriptorSize * cbvSrvUavDescriptorCount;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
        device->CreateShaderResourceView(resource, resourceViewDesc, cpuHandle);
        cbvSrvUavDescriptorCount++;
        return {cpuHandle, gpuHandle};
    }

    D3DDescriptor appendUAVDescriptor(D3D12_UNORDERED_ACCESS_VIEW_DESC* unorderedAccessViewDesc, ID3D12Resource* resource) {
        assert(cbvSrvUavDescriptorCount < cbvSrvUavDescriptorCapacity);
        uint offset = cbvSrvUavDescriptorSize * cbvSrvUavDescriptorCount;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
        device->CreateUnorderedAccessView(resource, nullptr, unorderedAccessViewDesc, cpuHandle);
        cbvSrvUavDescriptorCount++;
        return {cpuHandle, gpuHandle};
    }

    void graphicsQueueStartRecording() {
        assert(graphicsCmdAllocator->Reset() == S_OK);
        assert(graphicsCmdList->Reset(graphicsCmdAllocator, nullptr) == S_OK);
    }

    void graphicsQueueSubmitRecording() {
        assert(graphicsCmdList->Close() == S_OK);
        graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&graphicsCmdList);
    }

    void graphicsQueueWait() {
        graphicsQueueFenceCounter += 1;
        graphicsQueue->Signal(graphicsQueueFence, graphicsQueueFenceCounter);
        if (graphicsQueueFence->GetCompletedValue() < graphicsQueueFenceCounter) {
            assert(graphicsQueueFence->SetEventOnCompletion(graphicsQueueFenceCounter, graphicsQueueFenceEvent) == S_OK);
            assert(WaitForSingleObjectEx(graphicsQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
        }
    }

    void transferQueueStartRecording() {
        assert(transferCmdAllocator->Reset() == S_OK);
        assert(transferCmdList->Reset(transferCmdAllocator, nullptr) == S_OK);
    }

    void transferQueueSubmitRecording() {
        assert(transferCmdList->Close() == S_OK);
        transferQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&transferCmdList);
    }

    void transferQueueWait() {
        transferQueueFenceCounter += 1;
        transferQueue->Signal(transferQueueFence, transferQueueFenceCounter);
        if (transferQueueFence->GetCompletedValue() < transferQueueFenceCounter) {
            assert(transferQueueFence->SetEventOnCompletion(transferQueueFenceCounter, transferQueueFenceEvent) == S_OK);
            assert(WaitForSingleObjectEx(transferQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
        }
    }

    D3D12MA::Allocation* create2DImage(const D3D12_RESOURCE_DESC& resourceDesc, D3D12_SUBRESOURCE_DATA* imageMips) {
        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        D3D12MA::Allocation* image;
        assert(allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr) == S_OK);
        stagingBufferOffset = align(stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
        uint rowCounts[16];
        uint64 rowSizes[16];
        uint64 requiredSize;
        device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
        assert(stagingBufferOffset + requiredSize < stagingBuffer->GetSize());
        for (uint mipIndex = 0; mipIndex < resourceDesc.MipLevels; mipIndex++) {
            mipFootprints[mipIndex].Offset += stagingBufferOffset;
        }
        assert(UpdateSubresources(transferCmdList, image->GetResource(), stagingBuffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, imageMips) == requiredSize);
        stagingBufferOffset += (uint)requiredSize;
        return image;
    }

    D3D12MA::Allocation* create2DImageDDS(const std::filesystem::path& ddsFilePath) {
        ScratchImage scratchImage;
        assert(LoadFromDDSFile(ddsFilePath.c_str(), DDS_FLAGS_NONE, nullptr, scratchImage) == S_OK);
        assert(scratchImage.GetImageCount() == scratchImage.GetMetadata().mipLevels);
        const TexMetadata& scratchImageInfo = scratchImage.GetMetadata();
        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint)scratchImageInfo.width, .Height = (uint)scratchImageInfo.height, .DepthOrArraySize = (uint16)scratchImageInfo.arraySize, .MipLevels = (uint16)scratchImageInfo.mipLevels, .Format = scratchImageInfo.format, .SampleDesc = {.Count = 1}};
        D3D12MA::Allocation* image;
        assert(allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr) == S_OK);
        stagingBufferOffset = align(stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
        uint rowCounts[16];
        uint64 rowSizes[16];
        uint64 requiredSize;
        D3D12_SUBRESOURCE_DATA srcData[16];
        device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
        assert(stagingBufferOffset + requiredSize < stagingBuffer->GetSize());
        for (uint mipIndex = 0; mipIndex < scratchImageInfo.mipLevels; mipIndex++) {
            mipFootprints[mipIndex].Offset += stagingBufferOffset;
            const Image& image = scratchImage.GetImages()[mipIndex];
            srcData[mipIndex] = {.pData = image.pixels, .RowPitch = (int64)image.rowPitch, .SlicePitch = (int64)image.slicePitch};
        }
        assert(UpdateSubresources(transferCmdList, image->GetResource(), stagingBuffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, srcData) == requiredSize);
        stagingBufferOffset += (uint)requiredSize;
        return image;
    }
};

static D3D d3d = {};

struct CameraEditor {
    float3 position = {0, 0, -10};
    float3 lookAt = {0, 0, 0};
    float2 pitchYaw = {0, 0};
    float fovVertical = 50;
    float sensitivity = 1;
    float rotationSensitivity = 1;
    float controllerSensitivity = 1;

    void updateLookAt() {
        pitchYaw.x = std::clamp(pitchYaw.x, -pi * 0.4f, pi * 0.4f);
        pitchYaw.y = std::remainderf(pitchYaw.y, pi * 2.0f);
        XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(pitchYaw.x, pitchYaw.y, 0);
        float3 dir = XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion);
        lookAt = position + dir;
    }
    void rotate(float pitchDelta, float yawDelta) {
        pitchYaw.x += pitchDelta;
        pitchYaw.y += yawDelta;
        updateLookAt();
    }
    void translate(float3 translate) {
        float3 dz = (lookAt - position).normalize();
        float3 dx = dz.cross({0, 1, 0});
        float3 dy = dz.cross({1, 0, 0});
        position += dx * translate.x;
        lookAt += dx * translate.x;
        position += dy * translate.y;
        lookAt += dy * translate.y;
        position += dz * translate.z;
        lookAt += dz * translate.z;
    }
    void focus(float3 position, float distance) {
    }
};

struct CameraThirdPerson {
    float3 lookAtOffset;
    float3 lookAt;
    float3 rotation;
    float distance;
    float3 position;
    float3 dir;
    float sensitivity;
    float controllerSensitivity;

    // void update() {
    //     rotation.x = std::clamp(rotation.x, -pi * 0.4f, pi * 0.4f);
    //     rotation.y = std::remainderf(rotation.y, pi * 2.0f);
    //     XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(rotation.x, rotation.y, 0);
    //     float3 d = float3(XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion)).normalize();
    //     position = lookAt + d * distance;
    //     dir = -d;
    // }
};

struct ModelImage {
    D3D12MA::Allocation* gpuData = nullptr;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
};

struct ModelTextureSampler {
};

struct ModelTexture {
    ModelImage* image = nullptr;
    ModelTextureSampler sampler;
};

struct ModelMaterial {
    std::string name;
    float4 baseColorFactor = {1, 1, 1, 1};
    ModelTexture* baseColorTexture = nullptr;
};

struct ModelPrimitive {
    uint verticesBufferOffset;
    uint verticesCount;
    uint indicesBufferOffset;
    uint indicesCount;
    ModelMaterial* material = nullptr;
};

struct ModelMesh {
    std::string name;
    std::vector<ModelPrimitive> primitives;
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    D3D12MA::Allocation* verticesBuffer = nullptr;
    D3D12MA::Allocation* indicesBuffer = nullptr;
    D3D12MA::Allocation* blas = nullptr;
    D3D12MA::Allocation* blasScratch = nullptr;
};

struct ModelNode {
    std::string name;
    ModelNode* parent = nullptr;
    std::vector<ModelNode*> children;
    XMMATRIX globalTransform;
    XMMATRIX localTransform;
    ModelMesh* mesh = nullptr;
};

struct ModelJoint {
    ModelNode* node = nullptr;
    XMMATRIX inverseBindMat;
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
    ModelNode* node = nullptr;
    ModelAnimationSampler* sampler = nullptr;
    ModelAnimationChannelType type;
};

struct ModelAnimation {
    std::string name;
    std::vector<ModelAnimationChannel> channels;
    std::vector<ModelAnimationSampler> samplers;
    float timeLength = 0.0f;
};

struct Model {
    std::filesystem::path filePath;
    cgltf_data* gltfData = nullptr;
    std::vector<ModelMesh> meshes;
    std::vector<ModelNode> nodes;
    std::vector<ModelNode*> rootNodes;
    std::vector<ModelJoint> joints;
    std::vector<ModelAnimation> animations;
    std::vector<ModelMaterial> materials;
    std::vector<ModelTexture> textures;
    std::vector<ModelImage> images;
};

struct ModelInstanceAnimationState {
    uint index = 0;
    double time = 0;
    D3D12MA::Allocation* skinJointsBuffer = nullptr;
    uint8* skinJointsBufferPtr = nullptr;
    std::vector<D3D12MA::Allocation*> meshVerticesBuffers;
    std::vector<D3D12MA::Allocation*> meshBlases;
    std::vector<D3D12MA::Allocation*> meshBlasScratches;

    void releaseResources() {
        if (skinJointsBuffer) skinJointsBuffer->Release();
        for (D3D12MA::Allocation* buffer : meshVerticesBuffers) buffer->Release();
        for (D3D12MA::Allocation* buffer : meshBlases) buffer->Release();
        for (D3D12MA::Allocation* buffer : meshBlasScratches) buffer->Release();
    }
};

struct ModelInstance {
    uint index = UINT_MAX;
    ModelInstanceAnimationState animationState;
};

struct Player {
    ModelInstance model;
    Transform transform;
    float3 velocity;
    float3 acceleration;
    CameraThirdPerson camera;

    void releaseResource() { model.animationState.releaseResources(); }
};

struct StaticObject {
    std::string name;
    ModelInstance model;
    Transform transform;
    bool toBeDeleted;

    void releaseResource() { model.animationState.releaseResources(); }
};

struct DynamicObject {
    std::string name;
    ModelInstance model;
    Transform transform;
    bool toBeDeleted;

    void releaseResource() { model.animationState.releaseResources(); }
};

struct Skybox {
    std::filesystem::path hdriTextureFilePath;
    D3D12MA::Allocation* hdriTexture;
};

enum EditorUndoType {
    WorldEditorUndoTypeObjectDeletion
};

struct EditorUndoObjectDeletion {
    WorldObjectType objectType;
    void* object;
};

struct EditorUndo {
    EditorUndoType type;
    union {
        EditorUndoObjectDeletion* objectDeletion;
    };
};

struct Editor {
    bool enable = true;
    bool active = true;
    CameraEditor camera;
    bool cameraMoving = false;
    WorldObjectType selectedObjectType = WorldObjectTypeNone;
    uint selectedObjectIndex = 0;
    std::stack<EditorUndo> undos;

    Player player;
    std::vector<StaticObject> staticObjects;
    std::vector<DynamicObject> dynamicObjects;
};

Editor editor;

struct World {
    std::filesystem::path filePath;
    std::vector<Model> models;
    Player player;
    std::vector<StaticObject> staticObjects;
    std::vector<DynamicObject> dynamicObjects;
    Skybox skybox;
    std::vector<Light> lights;

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> tlasInstancesBuildInfos;
    std::vector<TLASInstanceInfo> tlasInstancesInfos;
    std::vector<BLASGeometryInfo> blasGeometriesInfos;

    void load(const std::filesystem::path& path) {
        if (std::filesystem::exists(path)) {
            std::string yamlStr = fileReadStr(path);
            ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
            ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
            filePath = path;

            if (editor.enable && yamlRoot.has_child("editorCamera")) {
                ryml::ConstNodeRef cameraYaml = yamlRoot["editorCamera"];
                editor.camera.position << cameraYaml["position"];
                editor.camera.pitchYaw << cameraYaml["pitchYaw"];
                editor.camera.updateLookAt();
                cameraYaml["fovVertical"] >> editor.camera.fovVertical;
                cameraYaml["sensitivity"] >> editor.camera.sensitivity;
                cameraYaml["rotationSensitivity"] >> editor.camera.rotationSensitivity;
                cameraYaml["controllerSensitivity"] >> editor.camera.controllerSensitivity;
            } else {
                editor.camera = {};
            }
            {
                ryml::ConstNodeRef skyboxYaml = yamlRoot["skybox"];
                std::string file;
                skyboxYaml["file"] >> file;
                skybox.hdriTextureFilePath = file;
                d3d.transferQueueStartRecording();
                d3d.stagingBufferOffset = 0;
                skybox.hdriTexture = d3d.create2DImageDDS(assetsDir / skybox.hdriTextureFilePath);
                D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = skybox.hdriTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
                d3d.transferCmdList->ResourceBarrier(1, &barrier);
                d3d.transferQueueSubmitRecording();
                d3d.transferQueueWait();
            }
            {
                ryml::ConstNodeRef playerYaml = yamlRoot["player"];
                std::string file;
                playerYaml["file"] >> file;
                player.model = loadModel(file);
                player.transform << playerYaml;
                player.velocity << playerYaml["velocity"];
                player.acceleration << playerYaml["acceleration"];
                player.camera.lookAtOffset << playerYaml["cameraLookAtOffset"];
                player.camera.rotation << playerYaml["cameraRotation"];
                playerYaml["cameraDistance"] >> player.camera.distance;
                player.camera.lookAt = player.transform.t + player.camera.lookAtOffset;
                if (editor.enable) editor.player = player;
            }
            ryml::ConstNodeRef staticObjectsYaml = yamlRoot["staticObjects"];
            for (ryml::ConstNodeRef const& staticObjectYaml : staticObjectsYaml) {
                StaticObject& obj = staticObjects.emplace_back();
                staticObjectYaml["name"] >> obj.name;
                std::string file;
                staticObjectYaml["file"] >> file;
                obj.model = loadModel(file);
                obj.transform << staticObjectYaml;
                if (editor.enable) editor.staticObjects.push_back(obj);
            }
            ryml::ConstNodeRef dynamicObjectsYaml = yamlRoot["dynamicObjects"];
            for (ryml::ConstNodeRef const& dynamicObjectYaml : dynamicObjectsYaml) {
                DynamicObject& obj = dynamicObjects.emplace_back();
                dynamicObjectYaml["name"] >> obj.name;
                std::string file;
                dynamicObjectYaml["file"] >> file;
                obj.model = loadModel(file);
                obj.transform << dynamicObjectYaml;
                if (editor.enable) editor.dynamicObjects.push_back(obj);
            }
        } else {
            assert(false && "implementation");
        }
    }

    void save() {
        if (!editor.enable) return;

        ryml::Tree yamlTree;
        ryml::NodeRef yamlRoot = yamlTree.rootref();
        yamlRoot |= ryml::MAP;

        ryml::NodeRef cameraYaml = yamlRoot["editorCamera"];
        cameraYaml |= ryml::MAP;
        editor.camera.position >> cameraYaml["position"];
        editor.camera.pitchYaw >> cameraYaml["pitchYaw"];
        cameraYaml["fovVertical"] << editor.camera.fovVertical;
        cameraYaml["sensitivity"] << editor.camera.sensitivity;
        cameraYaml["rotationSensitivity"] << editor.camera.rotationSensitivity;
        cameraYaml["controllerSensitivity"] << editor.camera.controllerSensitivity;

        ryml::NodeRef skyboxYaml = yamlRoot["skybox"];
        skyboxYaml |= ryml::MAP;
        skyboxYaml["file"] << skybox.hdriTextureFilePath.string();

        ryml::NodeRef playerYaml = yamlRoot["player"];
        playerYaml |= ryml::MAP;
        playerYaml["file"] << models[editor.player.model.index].filePath.string();
        editor.player.transform >> playerYaml;
        editor.player.velocity >> playerYaml["velocity"];
        editor.player.acceleration >> playerYaml["acceleration"];
        editor.player.camera.lookAtOffset >> playerYaml["cameraLookAtOffset"];
        editor.player.camera.rotation >> playerYaml["cameraRotation"];
        playerYaml["cameraDistance"] << editor.player.camera.distance;

        ryml::NodeRef staticObjectsYaml = yamlRoot["staticObjects"];
        staticObjectsYaml |= ryml::SEQ;
        for (StaticObject& staticObject : editor.staticObjects) {
            ryml::NodeRef staticObjectYaml = staticObjectsYaml.append_child();
            staticObjectYaml |= ryml::MAP;
            staticObjectYaml["name"] << staticObject.name;
            staticObjectYaml["file"] << models[staticObject.model.index].filePath.string();
            staticObject.transform >> staticObjectYaml;
        }

        ryml::NodeRef dynamicObjectsYaml = yamlRoot["dynamicObjects"];
        dynamicObjectsYaml |= ryml::SEQ;
        for (DynamicObject& dynamicObject : editor.dynamicObjects) {
            ryml::NodeRef dynamicObjectYaml = dynamicObjectsYaml.append_child();
            dynamicObjectYaml |= ryml::MAP;
            dynamicObjectYaml["name"] << dynamicObject.name;
            dynamicObjectYaml["file"] << models[dynamicObject.model.index].filePath.string();
            dynamicObject.transform >> dynamicObjectYaml;
        }

        std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
        assert(fileWriteStr(filePath, yamlStr));
    }

    ModelInstance loadModel(const std::filesystem::path& filePath) {
        if (filePath.extension() != ".gltf") return {.index = UINT_MAX};

        auto modelIter = std::find_if(models.begin(), models.end(), [&](auto& model) { return model.filePath == filePath; });
        if (modelIter == models.end()) {
            Model& model = models.emplace_back();
            modelIter = models.end() - 1;

            const std::filesystem::path gltfFilePath = assetsDir / filePath;
            const std::filesystem::path gltfFileFolderPath = gltfFilePath.parent_path();
            cgltf_options gltfOptions = {};
            cgltf_data* gltfData = nullptr;
            cgltf_result gltfParseFileResult = cgltf_parse_file(&gltfOptions, gltfFilePath.string().c_str(), &gltfData);
            assert(gltfParseFileResult == cgltf_result_success);
            cgltf_result gltfLoadBuffersResult = cgltf_load_buffers(&gltfOptions, gltfData, gltfFilePath.string().c_str());
            assert(gltfLoadBuffersResult == cgltf_result_success);
            model.filePath = filePath;
            model.gltfData = gltfData;

            d3d.transferQueueStartRecording();
            d3d.stagingBufferOffset = 0;

            model.images.reserve(gltfData->images_count);
            for (uint imageIndex = 0; imageIndex < gltfData->images_count; imageIndex++) {
                cgltf_image& gltfImage = gltfData->images[imageIndex];
                ModelImage& image = model.images.emplace_back();
                std::filesystem::path imageFilePath = gltfFileFolderPath / gltfImage.uri;
                std::filesystem::path imageDDSFilePath = imageFilePath;
                imageDDSFilePath.replace_extension(".dds");
                assert(std::filesystem::exists(imageDDSFilePath));
                image.gpuData = d3d.create2DImageDDS(imageDDSFilePath);
                D3D12_RESOURCE_DESC imageDesc = image.gpuData->GetResource()->GetDesc();
                image.srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
                D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = image.gpuData->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
                d3d.transferCmdList->ResourceBarrier(1, &barrier);
            }
            model.textures.reserve(gltfData->textures_count);
            for (uint textureIndex = 0; textureIndex < gltfData->textures_count; textureIndex++) {
                cgltf_texture& gltfTexture = gltfData->textures[textureIndex];
                ModelTexture& texture = model.textures.emplace_back();
                assertDebug(gltfTexture.image && gltfTexture.sampler);
                texture.image = &model.images[gltfTexture.image - &gltfData->images[0]];
            }
            model.materials.reserve(gltfData->materials_count);
            for (uint materialIndex = 0; materialIndex < gltfData->materials_count; materialIndex++) {
                cgltf_material& gltfMaterial = gltfData->materials[materialIndex];
                ModelMaterial& material = model.materials.emplace_back();
                if (gltfMaterial.name) material.name = gltfMaterial.name;
                assertDebug(gltfMaterial.has_pbr_metallic_roughness);
                material.baseColorFactor = float4(gltfMaterial.pbr_metallic_roughness.base_color_factor);
                if (gltfMaterial.pbr_metallic_roughness.base_color_texture.texture) {
                    assertDebug(gltfMaterial.pbr_metallic_roughness.base_color_texture.texcoord == 0);
                    assertDebug(!gltfMaterial.pbr_metallic_roughness.base_color_texture.has_transform);
                    material.baseColorTexture = &model.textures[gltfMaterial.pbr_metallic_roughness.base_color_texture.texture - &gltfData->textures[0]];
                }
            }
            model.meshes.reserve(gltfData->meshes_count);
            for (uint meshIndex = 0; meshIndex < gltfData->meshes_count; meshIndex++) {
                cgltf_mesh& gltfMesh = gltfData->meshes[meshIndex];
                ModelMesh& mesh = model.meshes.emplace_back();
                if (gltfMesh.name) mesh.name = gltfMesh.name;
                mesh.primitives.reserve(gltfMesh.primitives_count);
                for (cgltf_primitive& gltfPrimitive : std::span(gltfMesh.primitives, gltfMesh.primitives_count)) {
                    cgltf_accessor* indices = gltfPrimitive.indices;
                    cgltf_accessor* positions = nullptr;
                    cgltf_accessor* normals = nullptr;
                    cgltf_accessor* uvs = nullptr;
                    cgltf_accessor* jointIndices = nullptr;
                    cgltf_accessor* jointWeights = nullptr;
                    for (cgltf_attribute& attribute : std::span(gltfPrimitive.attributes, gltfPrimitive.attributes_count)) {
                        if (attribute.type == cgltf_attribute_type_position) {
                            positions = attribute.data;
                        } else if (attribute.type == cgltf_attribute_type_normal) {
                            normals = attribute.data;
                        } else if (attribute.type == cgltf_attribute_type_texcoord) {
                            uvs = attribute.data;
                        } else if (attribute.type == cgltf_attribute_type_joints) {
                            jointIndices = attribute.data;
                        } else if (attribute.type == cgltf_attribute_type_weights) {
                            jointWeights = attribute.data;
                        }
                    }
                    assert(gltfPrimitive.type == cgltf_primitive_type_triangles);
                    assert(indices && positions && normals);
                    assert(indices->count % 3 == 0 && indices->type == cgltf_type_scalar && (indices->component_type == cgltf_component_type_r_16u || indices->component_type == cgltf_component_type_r_32u));
                    assert(positions->type == cgltf_type_vec3 && positions->component_type == cgltf_component_type_r_32f);
                    assert(normals->count == positions->count && normals->type == cgltf_type_vec3 && normals->component_type == cgltf_component_type_r_32f);
                    if (uvs) assert(uvs->count == positions->count && uvs->component_type == cgltf_component_type_r_32f && uvs->type == cgltf_type_vec2);
                    if (jointIndices) assert(jointIndices->count == positions->count && (jointIndices->component_type == cgltf_component_type_r_16u || jointIndices->component_type == cgltf_component_type_r_8u) && jointIndices->type == cgltf_type_vec4 && (jointIndices->stride == 8 || jointIndices->stride == 4));
                    if (jointWeights) assert(jointWeights->count == positions->count && jointWeights->component_type == cgltf_component_type_r_32f && jointWeights->type == cgltf_type_vec4 && jointWeights->stride == 16);
                    float3* positionsBuffer = (float3*)((uint8*)(positions->buffer_view->buffer->data) + positions->offset + positions->buffer_view->offset);
                    float3* normalsBuffer = (float3*)((uint8*)(normals->buffer_view->buffer->data) + normals->offset + normals->buffer_view->offset);
                    void* indicesBuffer = (uint8*)(indices->buffer_view->buffer->data) + indices->offset + indices->buffer_view->offset;
                    float2* uvsBuffer = uvs ? (float2*)((uint8*)(uvs->buffer_view->buffer->data) + uvs->offset + uvs->buffer_view->offset) : nullptr;
                    void* jointIndicesBuffer = jointIndices ? (uint8*)(jointIndices->buffer_view->buffer->data) + jointIndices->offset + jointIndices->buffer_view->offset : nullptr;
                    float4* jointWeightsBuffer = jointWeights ? (float4*)((uint8*)(jointWeights->buffer_view->buffer->data) + jointWeights->offset + jointWeights->buffer_view->offset) : nullptr;

                    ModelPrimitive& primitive = mesh.primitives.emplace_back();
                    primitive.verticesBufferOffset = (uint)mesh.vertices.size();
                    primitive.verticesCount = (uint)positions->count;
                    primitive.indicesBufferOffset = (uint)mesh.indices.size();
                    primitive.indicesCount = (uint)indices->count;
                    for (uint vertexIndex = 0; vertexIndex < positions->count; vertexIndex++) {
                        Vertex vertex = {.position = positionsBuffer[vertexIndex], .normal = normalsBuffer[vertexIndex]};
                        if (uvsBuffer) vertex.uv = uvsBuffer[vertexIndex];
                        if (jointIndicesBuffer) {
                            if (jointIndices->component_type == cgltf_component_type_r_16u) vertex.joints = ((uint16_4*)jointIndicesBuffer)[vertexIndex];
                            else vertex.joints = ((uint8_4*)jointIndicesBuffer)[vertexIndex];
                        }
                        if (jointWeightsBuffer) vertex.jointWeights = jointWeightsBuffer[vertexIndex];
                        mesh.vertices.push_back(vertex);
                    }
                    if (indices->component_type == cgltf_component_type_r_16u) mesh.indices.append_range(std::span((uint16*)indicesBuffer, indices->count));
                    else if (indices->component_type == cgltf_component_type_r_32u) mesh.indices.append_range(std::span((uint*)indicesBuffer, indices->count));
                    if (gltfPrimitive.material) primitive.material = &model.materials[gltfPrimitive.material - gltfData->materials];
                }

                D3D12MA::ALLOCATION_DESC verticeIndicesBuffersAllocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC verticeIndicesBuffersDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
                verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.vertices);
                assert(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.verticesBuffer, {}, nullptr) == S_OK);
                verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.indices);
                assert(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.indicesBuffer, {}, nullptr) == S_OK);

                memcpy(d3d.stagingBufferPtr + d3d.stagingBufferOffset, mesh.vertices.data(), vectorSizeof(mesh.vertices));
                d3d.transferCmdList->CopyBufferRegion(mesh.verticesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), d3d.stagingBufferOffset, vectorSizeof(mesh.vertices));
                d3d.stagingBufferOffset += (uint)vectorSizeof(mesh.vertices);
                memcpy(d3d.stagingBufferPtr + d3d.stagingBufferOffset, mesh.indices.data(), vectorSizeof(mesh.indices));
                d3d.transferCmdList->CopyBufferRegion(mesh.indicesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), d3d.stagingBufferOffset, vectorSizeof(mesh.indices));
                d3d.stagingBufferOffset += (uint)vectorSizeof(mesh.indices);
                assert(d3d.stagingBufferOffset < d3d.stagingBuffer->GetSize());
                D3D12_RESOURCE_BARRIER bufferBarriers[2] = {
                    {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = mesh.verticesBuffer->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}},
                    {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = mesh.indicesBuffer->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}},
                };
                d3d.transferCmdList->ResourceBarrier(countof(bufferBarriers), bufferBarriers);

                std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
                for (ModelPrimitive& primitive : mesh.primitives) {
                    D3D12_RAYTRACING_GEOMETRY_DESC& geometryDesc = geometryDescs.emplace_back();
                    geometryDesc = {
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
                    };
                }
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE, .NumDescs = (uint)geometryDescs.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .pGeometryDescs = geometryDescs.data()};
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
                d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

                D3D12MA::ALLOCATION_DESC blasAllocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC blasDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
                blasDesc.Width = prebuildInfo.ResultDataMaxSizeInBytes;
                assert(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &mesh.blas, {}, nullptr) == S_OK);
                blasDesc.Width = prebuildInfo.ScratchDataSizeInBytes;
                assert(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &mesh.blasScratch, {}, nullptr) == S_OK);
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = mesh.blas->GetResource()->GetGPUVirtualAddress(), .Inputs = inputs, .ScratchAccelerationStructureData = mesh.blasScratch->GetResource()->GetGPUVirtualAddress()};
                d3d.transferCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            }
            model.nodes.resize(gltfData->nodes_count);
            for (uint nodeIndex = 0; nodeIndex < gltfData->nodes_count; nodeIndex++) {
                cgltf_node& gltfNode = gltfData->nodes[nodeIndex];
                ModelNode& node = model.nodes[nodeIndex];
                if (gltfNode.name) node.name = gltfNode.name;
                if (gltfNode.parent) {
                    uint parentNodeIndex = (uint)(gltfNode.parent - gltfData->nodes);
                    assertDebug(parentNodeIndex >= 0 && parentNodeIndex < gltfData->nodes_count);
                    node.parent = &model.nodes[parentNodeIndex];
                } else {
                    node.parent = nullptr;
                }
                for (cgltf_node* child : std::span(gltfNode.children, gltfNode.children_count)) {
                    uint childNodeIndex = (uint)(child - gltfData->nodes);
                    assertDebug(childNodeIndex >= 0 && childNodeIndex < gltfData->nodes_count);
                    node.children.push_back(&model.nodes[childNodeIndex]);
                }
                float nodeTransform[16];
                cgltf_node_transform_world(&gltfNode, nodeTransform);
                node.globalTransform = XMMATRIX(nodeTransform);
                if (gltfNode.has_matrix) {
                    node.localTransform = XMMATRIX(gltfNode.matrix);
                } else {
                    Transform localTransform = {};
                    if (gltfNode.has_scale) localTransform.s = float3(gltfNode.scale);
                    if (gltfNode.has_rotation) localTransform.r = float4(gltfNode.rotation);
                    if (gltfNode.has_translation) localTransform.t = float3(gltfNode.translation);
                    node.localTransform = localTransform.toMat();
                }
                if (gltfNode.mesh) {
                    uint meshIndex = (uint)(gltfNode.mesh - gltfData->meshes);
                    assertDebug(meshIndex >= 0 && meshIndex < gltfData->meshes_count);
                    node.mesh = &model.meshes[meshIndex];
                } else {
                    node.mesh = nullptr;
                }
            }
            assertDebug(gltfData->scenes_count == 1);
            for (cgltf_node* gltfNode : std::span(gltfData->scenes[0].nodes, gltfData->scenes[0].nodes_count)) {
                uint nodeIndex = (uint)(gltfNode - gltfData->nodes);
                assertDebug(nodeIndex >= 0 && nodeIndex < gltfData->nodes_count);
                model.rootNodes.push_back(&model.nodes[nodeIndex]);
            }
            assertDebug(gltfData->skins_count <= 1);
            if (gltfData->skins_count == 1) {
                cgltf_skin& gltfSkin = gltfData->skins[0];
                assertDebug(gltfSkin.inverse_bind_matrices->type == cgltf_type_mat4);
                assertDebug(gltfSkin.inverse_bind_matrices->count == gltfSkin.joints_count);
                assertDebug(gltfSkin.inverse_bind_matrices->stride == 16 * sizeof(float));
                assertDebug(gltfSkin.inverse_bind_matrices->buffer_view->size == 64 * gltfSkin.joints_count);
                model.joints.reserve(gltfSkin.joints_count);
                for (uint jointIndex = 0; jointIndex < gltfSkin.joints_count; jointIndex++) {
                    cgltf_node* jointNode = gltfSkin.joints[jointIndex];
                    uint nodeIndex = (uint)(jointNode - gltfData->nodes);
                    assertDebug(nodeIndex >= 0 && nodeIndex < gltfData->nodes_count);
                    float* matsData = (float*)((uint8*)(gltfSkin.inverse_bind_matrices->buffer_view->buffer->data) + gltfSkin.inverse_bind_matrices->offset + gltfSkin.inverse_bind_matrices->buffer_view->offset);
                    matsData += jointIndex * 16;
                    model.joints.push_back({.node = &model.nodes[nodeIndex], .inverseBindMat = XMMATRIX(matsData)});
                }
            }
            model.animations.reserve(gltfData->animations_count);
            for (cgltf_animation& gltfAnimation : std::span(gltfData->animations, gltfData->animations_count)) {
                ModelAnimation& animation = model.animations.emplace_back();
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
                    } else if (gltfSampler.interpolation == cgltf_interpolation_type_step) {
                        sampler.interpolation = AnimationSamplerInterpolationStep;
                    } else if (gltfSampler.interpolation == cgltf_interpolation_type_cubic_spline) {
                        sampler.interpolation = AnimationSamplerInterpolationCubicSpline;
                        assertDebug(false);
                    } else {
                        assertDebug(false);
                    }
                    assertDebug(gltfSampler.input->component_type == cgltf_component_type_r_32f && gltfSampler.input->type == cgltf_type_scalar);
                    assertDebug(gltfSampler.output->component_type == cgltf_component_type_r_32f);
                    assertDebug((gltfSampler.output->type == cgltf_type_vec3 && gltfSampler.output->stride == sizeof(float3)) || (gltfSampler.output->type == cgltf_type_vec4 && gltfSampler.output->stride == sizeof(float4)));
                    assertDebug(gltfSampler.input->count >= 2 && gltfSampler.input->count == gltfSampler.output->count);
                    float* inputs = (float*)((uint8*)(gltfSampler.input->buffer_view->buffer->data) + gltfSampler.input->offset + gltfSampler.input->buffer_view->offset);
                    void* outputs = (uint8*)(gltfSampler.output->buffer_view->buffer->data) + gltfSampler.output->offset + gltfSampler.output->buffer_view->offset;
                    assertDebug(inputs[0] == 0.0f);
                    assertDebug(inputs[gltfSampler.input->count - 1] == animation.timeLength);
                    sampler.keyFrames.reserve(gltfSampler.input->count);
                    for (uint frameIndex = 0; frameIndex < gltfSampler.input->count; frameIndex++) {
                        ModelAnimationSamplerKeyFrame& keyFrame = sampler.keyFrames.emplace_back();
                        keyFrame.time = inputs[frameIndex];
                        if (gltfSampler.output->type == cgltf_type_vec3) {
                            keyFrame.xyzw = ((float3*)outputs)[frameIndex];
                        } else {
                            keyFrame.xyzw = ((float4*)outputs)[frameIndex];
                        }
                    }
                }
                animation.channels.reserve(gltfAnimation.channels_count);
                for (cgltf_animation_channel& gltfChannel : std::span(gltfAnimation.channels, gltfAnimation.channels_count)) {
                    ModelAnimationChannel& channel = animation.channels.emplace_back();
                    uint nodeIndex = (uint)(gltfChannel.target_node - gltfData->nodes);
                    uint samplerIndex = (uint)(gltfChannel.sampler - gltfAnimation.samplers);
                    assertDebug(nodeIndex >= 0 && nodeIndex < gltfData->nodes_count);
                    assertDebug(samplerIndex >= 0 && samplerIndex < gltfAnimation.samplers_count);
                    channel.node = &model.nodes[nodeIndex];
                    channel.sampler = &animation.samplers[samplerIndex];
                    if (gltfChannel.target_path == cgltf_animation_path_type_translation) {
                        assertDebug(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                        channel.type = AnimationChannelTypeTranslate;
                    } else if (gltfChannel.target_path == cgltf_animation_path_type_rotation) {
                        assertDebug(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec4);
                        channel.type = AnimationChannelTypeRotate;
                    } else if (gltfChannel.target_path == cgltf_animation_path_type_scale) {
                        assertDebug(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                        channel.type = AnimationChannelTypeScale;
                    } else {
                        assertDebug(false);
                    }
                }
            }

            d3d.transferQueueSubmitRecording();
            d3d.transferQueueWait();
        }

        ModelInstance modelInstance = {};
        modelInstance.index = (uint)(modelIter - models.begin());

        if (modelIter->joints.size() > 0) {
            D3D12MA::ALLOCATION_DESC jointBufferAllocDesc = {.HeapType = D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC jointBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = sizeof(struct Joint) * modelIter->joints.size(), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            assert(d3d.allocator->CreateResource(&jointBufferAllocDesc, &jointBufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &modelInstance.animationState.skinJointsBuffer, {}, nullptr) == S_OK);
            assert(modelInstance.animationState.skinJointsBuffer->GetResource()->Map(0, nullptr, (void**)&modelInstance.animationState.skinJointsBufferPtr) == S_OK);
            for (ModelMesh& mesh : modelIter->meshes) {
                D3D12MA::ALLOCATION_DESC verticesBufferAllocDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC verticesBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = mesh.verticesBuffer->GetSize(), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
                D3D12MA::Allocation* verticesBuffer;
                assert(d3d.allocator->CreateResource(&verticesBufferAllocDesc, &verticesBufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &verticesBuffer, {}, nullptr) == S_OK);
                modelInstance.animationState.meshVerticesBuffers.push_back(verticesBuffer);

                D3D12MA::Allocation* blasBuffer;
                D3D12MA::Allocation* blasScratchBuffer;
                D3D12MA::ALLOCATION_DESC blasAllocDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC blasDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
                blasDesc.Width = mesh.blas->GetSize();
                assert(d3d.allocator->CreateResource(&blasAllocDesc, &blasDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &blasBuffer, {}, nullptr) == S_OK);
                blasDesc.Width = mesh.blasScratch->GetSize();
                assert(d3d.allocator->CreateResource(&blasAllocDesc, &blasDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &blasScratchBuffer, {}, nullptr) == S_OK);
                modelInstance.animationState.meshBlases.push_back(blasBuffer);
                modelInstance.animationState.meshBlasScratches.push_back(blasScratchBuffer);
            }
        }

        return modelInstance;
    }

    void resetObjectsToEditor() {
        if (!editor.enable) return;
        player = editor.player;
        staticObjects = editor.staticObjects;
        dynamicObjects = editor.dynamicObjects;
    }
};

static World world = {};

ImGuiKey toImGuiKey(WPARAM wparam) {
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

static bool quit = false;
static LARGE_INTEGER perfFrequency = {};
static LARGE_INTEGER perfCounters[2] = {};
static double frameTime = 0;
static uint mouseSelectX = UINT_MAX;
static uint mouseSelectY = UINT_MAX;
static int mouseDeltaRaw[2] = {};
static float mouseWheel = 0;

LRESULT windowEventHandler(HWND hwnd, UINT eventType, WPARAM wParam, LPARAM lParam) {
    LRESULT result = 0;
    switch (eventType) {
    default: {
        result = DefWindowProcA(hwnd, eventType, wParam, lParam);
    } break;
    case WM_SHOWWINDOW:
    case WM_SIZE:
    case WM_MOVE: {
        uint prevRenderW = settings.renderW;
        uint prevRenderH = settings.renderH;
        window.updateSizes();
        if (d3d.swapChain && settings.renderW > 0 && settings.renderH > 0 && (prevRenderW != settings.renderW || prevRenderH != settings.renderH)) {
            d3d.resizeSwapChain(settings.renderW, settings.renderH);
        }
    } break;
    case WM_CLOSE:
    case WM_QUIT: {
        quit = true;
    } break;
    case WM_KEYDOWN:
    case WM_KEYUP: {
        ImGui::GetIO().AddKeyEvent(toImGuiKey(wParam), eventType == WM_KEYDOWN);
    } break;
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP: {
        ImGui::GetIO().AddKeyEvent(toImGuiKey(wParam), eventType == WM_SYSKEYDOWN);
    } break;
    case WM_CHAR: {
        ImGui::GetIO().AddInputCharacter(LOWORD(wParam));
    } break;
    case WM_INPUT: {
        static char rawInputBuffer[256];
        uint size = sizeof(rawInputBuffer);
        if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, rawInputBuffer, &size, sizeof(RAWINPUTHEADER)) < sizeof(rawInputBuffer)) {
            RAWINPUT* raw = (RAWINPUT*)rawInputBuffer;
            if (raw->header.dwType == RIM_TYPEMOUSE) {
                mouseDeltaRaw[0] += raw->data.mouse.lLastX;
                mouseDeltaRaw[1] += raw->data.mouse.lLastY;
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

struct Controller {
    bool a, b, x, y;
    bool up, down, left, right;
    bool lb, rb;
    bool ls, rs;
    bool back, start;
    float lt, rt;
    float2 lStick;
    float2 rStick;
    float deadZone = 0.25f;

    void getInputs() {
        static XINPUT_STATE prevState = {};
        XINPUT_STATE state = {};
        DWORD result = XInputGetState(0, &state);
        if (result == ERROR_SUCCESS) {
            a = state.Gamepad.wButtons & XINPUT_GAMEPAD_A;
            b = state.Gamepad.wButtons & XINPUT_GAMEPAD_B;
            x = state.Gamepad.wButtons & XINPUT_GAMEPAD_X;
            y = state.Gamepad.wButtons & XINPUT_GAMEPAD_Y;
            up = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP;
            down = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN;
            left = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT;
            right = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT;
            lb = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER;
            rb = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER;
            ls = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB;
            rs = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB;
            back = state.Gamepad.wButtons & XINPUT_GAMEPAD_BACK;
            start = state.Gamepad.wButtons & XINPUT_GAMEPAD_START;
            lt = state.Gamepad.bLeftTrigger / 255.0f;
            rt = state.Gamepad.bRightTrigger / 255.0f;
            lStick.x = state.Gamepad.sThumbLX / 32767.0f;
            lStick.y = state.Gamepad.sThumbLY / 32767.0f;
            rStick.x = state.Gamepad.sThumbRX / 32767.0f;
            rStick.y = state.Gamepad.sThumbRY / 32767.0f;

            float lDistance = lStick.length();
            if (lDistance > 0) {
                float lDistanceNew = std::max(0.0f, lDistance - deadZone) / (1.0f - deadZone);
                lStick = lStick / lDistance * lDistanceNew;
            }
            float rDistance = rStick.length();
            if (rDistance > 0) {
                float rDistanceNew = std::max(0.0f, rDistance - deadZone) / (1.0f - deadZone);
                rStick = rStick / rDistance * rDistanceNew;
            }
        } else {
            *this = {};
        }
        prevState = state;
    }

    bool stickMoved() {
        return lStick != float2(0, 0) || rStick != float2(0, 0);
    }
};

static Controller controller = {};

void hideCursor(bool hide) {
    if (hide) {
        POINT cursorP;
        GetCursorPos(&cursorP);
        RECT rect = {cursorP.x, cursorP.y, cursorP.x, cursorP.y};
        ClipCursor(&rect);
        ShowCursor(false);
    } else {
        ClipCursor(nullptr);
        ShowCursor(true);
    }
}

void editorUpdate() {
    if (!editor.enable) return;

    if (ImGui::IsKeyPressed(ImGuiKey_P, false) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
        editor.active = !editor.active;
        hideCursor(!editor.active);
        world.resetObjectsToEditor();
    }
    if (!editor.active) return;

    if (d3d.graphicsQueueFenceCounter > 0) {
        uint mouseSelectInstanceIndex = d3d.readBackBufferPtr->mouseSelectInstanceIndex;
        if (mouseSelectInstanceIndex < world.tlasInstancesInfos.size()) {
            TLASInstanceInfo& info = world.tlasInstancesInfos[mouseSelectInstanceIndex];
            editor.selectedObjectType = info.objectType;
            editor.selectedObjectIndex = info.objectIndex;
        } else {
            if (mouseSelectX != UINT_MAX && mouseSelectY != UINT_MAX) {
                editor.selectedObjectType = WorldObjectTypeNone;
            }
        }
        // world.player.transform.translate = readBackBuffer->playerPosition;
        // world.player.velocity = readBackBuffer->playerVelocity;
        // world.player.acceleration = readBackBuffer->playerAcceleration;
    }
    {
        auto advanceAnimation = [](ModelInstance& modelInstance, float time) {
            Model& model = world.models[modelInstance.index];
            if (modelInstance.animationState.index < model.animations.size()) {
                ModelAnimation& animation = model.animations[modelInstance.animationState.index];
                modelInstance.animationState.time += time;
                if (modelInstance.animationState.time > animation.timeLength) {
                    modelInstance.animationState.time -= animation.timeLength;
                }
            }
        };
        advanceAnimation(editor.player.model, (float)frameTime);
        for (StaticObject& obj : editor.staticObjects) advanceAnimation(obj.model, (float)frameTime);
        for (DynamicObject& obj : editor.dynamicObjects) advanceAnimation(obj.model, (float)frameTime);
    }

    static ImVec2 mousePosPrev = ImGui::GetMousePos();
    ImVec2 mousePos = ImGui::GetMousePos();
    ImVec2 mouseDelta = mousePos - mousePosPrev;
    mousePosPrev = mousePos;

    static std::vector<std::string> logs = {};
    static bool addObjectPopup = false;
    ImVec2 mainMenuBarPos;
    ImVec2 mainMenuBarSize;
    if (ImGui::BeginMainMenuBar()) {
        mainMenuBarPos = ImGui::GetWindowPos();
        mainMenuBarSize = ImGui::GetWindowSize();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::BeginMenu("New")) {
                if (ImGui::MenuItem("World")) {
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Open")) {
                if (ImGui::MenuItem("World")) {
                    char filePath[256] = {};
                    OPENFILENAMEA openFileName = {.lStructSize = sizeof(openFileName), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                    if (GetOpenFileNameA(&openFileName)) {
                    } else {
                        DWORD error = CommDlgExtendedError();
                        if (error == FNERR_BUFFERTOOSMALL) {
                        } else if (error == FNERR_INVALIDFILENAME) {
                        } else if (error == FNERR_SUBCLASSFAILURE) {
                        }
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit")) { quit = true; }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Display")) {
            if (ImGui::MenuItem(settings.hdr ? "HDR (On)" : "HDR (Off)")) {
                settings.hdr = !settings.hdr;
                d3d.applySettings();
            } else if (ImGui::MenuItem("Windowed")) {
                settings.windowMode = WindowModeWindowed;
                d3d.applySettings();
            } else if (ImGui::MenuItem("Borderless Fullscreen")) {
                settings.windowMode = WindowModeBorderless;
                d3d.applySettings();
            }
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
            ImGui::Text("Exclusive Fullscreen");
            for (DisplayMode& mode : d3d.displayModes) {
                std::string text = std::format("{}x{}", mode.resolutionWidth, mode.resolutionHeight);
                if (ImGui::BeginMenu(text.c_str())) {
                    for (DXGI_RATIONAL& refreshRate : mode.refreshRates) {
                        text = std::format("{:.2f}hz", (float)refreshRate.Numerator / (float)refreshRate.Denominator);
                        if (ImGui::MenuItem(text.c_str())) {
                            settings.windowMode = WindowModeFullscreen;
                            settings.windowW = mode.resolutionWidth;
                            settings.windowH = mode.resolutionHeight;
                            settings.refreshRate = refreshRate;
                            d3d.applySettings();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Editor")) {
            if (ImGui::BeginMenu("Camera")) {
                ImGui::SliderFloat("Sensitivity", &editor.camera.sensitivity, 0.0f, 100.0f);
                ImGui::SliderFloat("RotationSensitivity", &editor.camera.rotationSensitivity, 0.1f, 10.0f);
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Add")) {
                addObjectPopup = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Game")) {
            if (ImGui::MenuItem("Play", "CTRL+P")) {
                editor.active = false;
                hideCursor(false);
                world.resetObjectsToEditor();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    ImVec2 objectWindowPos = ImVec2(settings.renderW * 0.85f, mainMenuBarSize.y);
    ImVec2 objectWindowSize = ImVec2(settings.renderW * 0.15f, settings.renderH * 0.3f);
    ImGui::SetNextWindowPos(objectWindowPos);
    ImGui::SetNextWindowSize(objectWindowSize);
    if (ImGui::Begin("Objects")) {
        if (ImGui::Selectable("Player", editor.selectedObjectType == WorldObjectTypePlayer)) {
            editor.selectedObjectType = WorldObjectTypePlayer;
            editor.selectedObjectIndex = 0;
        }
        if (editor.selectedObjectType == WorldObjectTypePlayer && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("player edit");
        }
        if (editor.selectedObjectType == WorldObjectTypePlayer && ImGui::BeginPopup("player edit")) {
            if (ImGui::Selectable("focus")) editor.camera.focus(editor.player.transform.t, 1);
            ImGui::EndPopup();
        }
        ImGui::SetNextItemOpen(true);
        if (ImGui::TreeNode("Static Objects")) {
            int objID = 0;
            for (uint objIndex = 0; objIndex < editor.staticObjects.size(); objIndex++) {
                StaticObject& object = editor.staticObjects[objIndex];
                ImGui::PushID(objID++);
                if (ImGui::Selectable(object.name.c_str(), editor.selectedObjectType == WorldObjectTypeStaticObject && editor.selectedObjectIndex == objIndex)) {
                    editor.selectedObjectType = WorldObjectTypeStaticObject;
                    editor.selectedObjectIndex = objIndex;
                }
                ImGui::PopID();
                if (editor.selectedObjectType == WorldObjectTypeStaticObject && editor.selectedObjectIndex == objIndex && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImGui::OpenPopup("static object edit");
                }
                if (editor.selectedObjectType == WorldObjectTypeStaticObject && editor.selectedObjectIndex == objIndex && ImGui::BeginPopup("static object edit")) {
                    if (ImGui::Selectable("focus")) {
                        editor.camera.focus(object.transform.t, 1);
                    }
                    if (ImGui::Selectable("delete")) {
                        object.toBeDeleted = true;
                        editor.selectedObjectType = WorldObjectTypeNone;
                    }
                    ImGui::EndPopup();
                }
            }
            ImGui::TreePop();
        }
        ImGui::SetNextItemOpen(true);
        if (ImGui::TreeNode("Dyanmic Objects")) {
            int objID = 0;
            for (uint objIndex = 0; objIndex < editor.dynamicObjects.size(); objIndex++) {
                DynamicObject& object = editor.dynamicObjects[objIndex];
                ImGui::PushID(objID++);
                if (ImGui::Selectable(object.name.c_str(), editor.selectedObjectType == WorldObjectTypeDynamicObject && editor.selectedObjectIndex == objIndex)) {
                    editor.selectedObjectType = WorldObjectTypeDynamicObject;
                    editor.selectedObjectIndex = objIndex;
                }
                ImGui::PopID();
                if (editor.selectedObjectType == WorldObjectTypeDynamicObject && editor.selectedObjectIndex == objIndex && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImGui::OpenPopup("dynamic object edit");
                }
                if (editor.selectedObjectType == WorldObjectTypeDynamicObject && editor.selectedObjectIndex == objIndex && ImGui::BeginPopup("dynamic object edit")) {
                    if (ImGui::Selectable("focus")) {
                        editor.camera.focus(object.transform.t, 1);
                    }
                    if (ImGui::Selectable("delete")) {
                        object.toBeDeleted = true;
                        editor.selectedObjectType = WorldObjectTypeNone;
                    }
                    ImGui::EndPopup();
                }
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
    auto TransformProperties = [](Transform* transform) {
        ImGui::SetNextItemOpen(true);
        if (ImGui::TreeNode("Transform")) {
            ImGui::InputFloat3("S", &transform->s.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##scale")) { transform->s = float3(1, 1, 1); }
            ImGui::InputFloat4("R", &transform->r.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##rotate")) { transform->r = float4(0, 0, 0, 1); }
            ImGui::InputFloat3("T", &transform->t.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##translate")) { transform->t = float3(0, 0, 0); }
            ImGui::TreePop();
        }
    };
    auto ModelProperties = [](ModelInstance& modelInstance) {
        ImGui::SetNextItemOpen(true);
        if (ImGui::TreeNode("Model")) {
            Model& model = world.models[modelInstance.index];
            ImGui::Text(std::format("File: {}", model.filePath.string()).c_str());
            ImGui::SetNextItemOpen(true);
            if (ImGui::TreeNode("Animations")) {
                for (uint animationIndex = 0; animationIndex < model.animations.size(); animationIndex++) {
                    ModelAnimation& animation = model.animations[animationIndex];
                    ImGui::Text(std::format("#{}: {}", animationIndex, animation.name).c_str());
                    ImGui::SameLine(ImGui::GetWindowWidth() * 0.8f);
                    ImGui::PushID(animationIndex);
                    if (ImGui::Button("play")) {
                        modelInstance.animationState.index = animationIndex;
                        modelInstance.animationState.time = 0;
                    }
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
    };
    ImVec2 propertiesWindowPos = objectWindowPos + ImVec2(0, objectWindowSize.y);
    ImVec2 propertiesWindowSize = objectWindowSize;
    ImGui::SetNextWindowPos(propertiesWindowPos);
    ImGui::SetNextWindowSize(propertiesWindowSize);
    if (ImGui::Begin("Properties")) {
        if (editor.selectedObjectType == WorldObjectTypePlayer) {
            ImGui::Text("Player");
            TransformProperties(&editor.player.transform);
            if (ImGui::TreeNode("Movement")) {
                ImGui::InputFloat3("Velocity", &editor.player.velocity.x);
                ImGui::InputFloat3("Acceleration", &editor.player.acceleration.x);
                ImGui::TreePop();
            }
            ModelProperties(editor.player.model);
        } else if (editor.selectedObjectType == WorldObjectTypeStaticObject) {
            StaticObject& object = editor.staticObjects[editor.selectedObjectIndex];
            ImGui::Text("Static Object #%d", editor.selectedObjectIndex);
            ImGui::Text("Name \"%s\"", object.name.c_str());
            TransformProperties(&object.transform);
            ModelProperties(object.model);
        } else if (editor.selectedObjectType == WorldObjectTypeDynamicObject) {
            DynamicObject& object = editor.dynamicObjects[editor.selectedObjectIndex];
            ImGui::Text("Dynamic Object #%d", editor.selectedObjectIndex);
            ImGui::Text("Name \"%s\"", object.name.c_str());
            TransformProperties(&object.transform);
            ModelProperties(object.model);
        }
    }
    ImGui::End();
    ImVec2 logWindowPos = propertiesWindowPos + ImVec2(0, propertiesWindowSize.y);
    ImVec2 logWindowSize = propertiesWindowSize;
    ImGui::SetNextWindowPos(logWindowPos);
    ImGui::SetNextWindowSize(logWindowSize);
    if (ImGui::Begin("Logs")) {
        for (std::string& log : logs) { ImGui::BulletText(log.c_str()); }
    }
    ImGui::End();
    if (addObjectPopup) {
        ImGui::OpenPopup("Add Object");
        addObjectPopup = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Add Object", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        static int objectType = 0;
        static char objectName[32] = {};
        static char filePath[256] = {};
        ImGui::Combo("Object Type", &objectType, "Static Object\0Dyanmic Object\0");
        if (objectType == 0 || objectType == 1) {
            ImGui::InputText("Name", objectName, sizeof(objectName));
            ImGui::InputText("File", filePath, sizeof(filePath));
            ImGui::SameLine();
            if (ImGui::Button("Browse")) {
                OPENFILENAMEA openfileName = {.lStructSize = sizeof(OPENFILENAMEA), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                GetOpenFileNameA(&openfileName);
            }
        }
        if (ImGui::Button("Add")) {
            if (objectName[0] == '\0') {
                logs.push_back("error: object name is empty");
            } else {
                std::filesystem::path path = std::filesystem::relative(filePath, assetsDir);
                ModelInstance model = world.loadModel(path);
                if (model.index >= world.models.size()) {
                    logs.push_back(std::format("error: failed to load model file: \"{}\"", path.string()));
                } else {
                    if (objectType == 0) {
                        StaticObject staticObject = {.name = objectName, .model = model};
                        editor.staticObjects.push_back(std::move(staticObject));
                    } else if (objectType == 1) {
                        DynamicObject dynamicObject = {.name = objectName, .model = model};
                        editor.dynamicObjects.push_back(std::move(dynamicObject));
                    }
                }
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
        mouseSelectX = (uint)mousePos.x;
        mouseSelectY = (uint)mousePos.y;
    } else {
        mouseSelectX = UINT_MAX;
        mouseSelectY = UINT_MAX;
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::GetIO().WantCaptureMouse) {
        editor.cameraMoving = true;
        hideCursor(true);
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        editor.cameraMoving = false;
        hideCursor(false);
    }
    if (editor.cameraMoving || controller.stickMoved()) {
        float pitch = (mouseDeltaRaw[1] / 500.0f * editor.camera.rotationSensitivity) - (controller.rStick.y * (float)frameTime * editor.camera.controllerSensitivity);
        float yaw = (mouseDeltaRaw[0] / 500.0f * editor.camera.rotationSensitivity) + (controller.rStick.x * (float)frameTime * editor.camera.controllerSensitivity);
        editor.camera.sensitivity = std::clamp(editor.camera.sensitivity + ImGui::GetIO().MouseWheel, 0.0f, 100.0f);
        float distance = (float)frameTime / 5.0f * editor.camera.sensitivity;
        float3 translate = {0, 0, 0};
        if (ImGui::IsKeyDown(ImGuiKey_W)) translate.z = distance;
        if (ImGui::IsKeyDown(ImGuiKey_S)) translate.z = -distance;
        if (ImGui::IsKeyDown(ImGuiKey_A)) translate.x = distance;
        if (ImGui::IsKeyDown(ImGuiKey_D)) translate.x = -distance;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) translate.y = distance;
        if (ImGui::IsKeyDown(ImGuiKey_E)) translate.y = -distance;
        // editor.camera.position += editor.camera.dir.cross(float3(0, 1, 0)) * distance * -controller.lStick.x;
        // editor.camera.position += editor.camera.dir * distance * controller.lStick.y;
        editor.camera.rotate(pitch, yaw);
        editor.camera.translate(translate);
    }

    static ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;
    static ImGuizmo::MODE gizmoMode = ImGuizmo::WORLD;
    if (ImGui::IsKeyPressed(ImGuiKey_Space) && ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
        ImGui::OpenPopup("gizmo selection");
    }
    if (ImGui::BeginPopup("gizmo selection")) {
        if (ImGui::Selectable("world")) {
            gizmoMode = ImGuizmo::WORLD;
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Selectable("local")) {
            gizmoMode = ImGuizmo::LOCAL;
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();
        if (ImGui::Selectable("scale")) {
            gizmoOperation = ImGuizmo::SCALE;
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Selectable("rotate")) {
            gizmoOperation = ImGuizmo::ROTATE;
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Selectable("translate")) {
            gizmoOperation = ImGuizmo::TRANSLATE;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    const XMMATRIX lookAtMat = XMMatrixLookAtLH(editor.camera.position.toXMVector(), editor.camera.lookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
    const XMMATRIX perspectiveMat = XMMatrixPerspectiveFovLH(radian(editor.camera.fovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f);
    auto transformGizmo = [&](Transform* transform) {
        XMMATRIX transformMat = transform->toMat();
        if (!ImGui::IsAnyItemActive()) {
            if (ImGui::IsKeyPressed(ImGuiKey_T)) gizmoOperation = ImGuizmo::TRANSLATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_R)) gizmoOperation = ImGuizmo::ROTATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_S)) gizmoOperation = ImGuizmo::SCALE;
        }
        if (ImGuizmo::Manipulate((const float*)&lookAtMat, (const float*)&perspectiveMat, gizmoOperation, gizmoMode, (float*)&transformMat)) {
            XMVECTOR scale, rotate, translate;
            if (XMMatrixDecompose(&scale, &rotate, &translate, transformMat)) {
                transform->s = scale, transform->r = rotate, transform->t = translate;
            }
        }
    };
    if (editor.selectedObjectType == WorldObjectTypePlayer) {
        transformGizmo(&editor.player.transform);
    } else if (editor.selectedObjectType == WorldObjectTypeStaticObject && editor.selectedObjectIndex < editor.staticObjects.size()) {
        transformGizmo(&editor.staticObjects[editor.selectedObjectIndex].transform);
    } else if (editor.selectedObjectType == WorldObjectTypeDynamicObject && editor.selectedObjectIndex < editor.dynamicObjects.size()) {
        transformGizmo(&editor.dynamicObjects[editor.selectedObjectIndex].transform);
    }
    // static const XMMATRIX gridMat = XMMatrixIdentity();
    // ImGuizmo::DrawGrid((const float*)&lookAtMat, (const float*)&perspectiveMat, (const float*)&gridMat, 10);

    {
        auto objIter = editor.staticObjects.begin();
        while (objIter != editor.staticObjects.end()) {
            if (objIter->toBeDeleted) {
                objIter->releaseResource();
                objIter = editor.staticObjects.erase(objIter);
            } else {
                objIter++;
            }
        }
    }
    {
        auto objIter = editor.dynamicObjects.begin();
        while (objIter != editor.dynamicObjects.end()) {
            if (objIter->toBeDeleted) {
                objIter->releaseResource();
                objIter = editor.dynamicObjects.erase(objIter);
            } else {
                objIter++;
            }
        }
    }
}

void gameUpdate() {
    if (editor.enable && editor.active) return;
    {
        Model& model = world.models[world.player.model.index];
        if (world.player.model.animationState.index < model.animations.size()) {
            ModelAnimation& animation = model.animations[world.player.model.animationState.index];
            world.player.model.animationState.time += frameTime;
            if (world.player.model.animationState.time > animation.timeLength) {
                world.player.model.animationState.time -= animation.timeLength;
            }
        }
    }
}

void update() {
    ZoneScoped;
    ImGui::GetIO().DeltaTime = (float)frameTime;
    ImGui::GetIO().DisplaySize = ImVec2((float)settings.renderW, (float)settings.renderH);
    ImGui::NewFrame();
    ImGuizmo::SetRect(0, 0, (float)settings.renderW, (float)settings.renderH);
    ImGuizmo::BeginFrame();

    editorUpdate();
    gameUpdate();

    ImGui::Render();
}

void updateAnimatedModel(ModelInstance& modelInstance) {
    ZoneScoped;
    Model& model = world.models[modelInstance.index];
    if (model.joints.empty() || model.animations.size() == 0) return;
    if (modelInstance.animationState.index >= model.animations.size()) return;
    ModelAnimation& animation = model.animations[modelInstance.animationState.index];

    std::vector<Transform> nodeLocalTransforms(model.nodes.size());
    std::vector<XMMATRIX> nodeLocalTransformMats(model.nodes.size());
    std::vector<XMMATRIX> nodeGlobalTransformMats(model.nodes.size());
    std::vector<XMMATRIX> jointTransformMats(model.joints.size());
    {
        ZoneScopedN("get nodeLocalTransforms");
        for (ModelAnimationChannel& channel : animation.channels) {
            float4 frame0 = channel.sampler->keyFrames[0].xyzw;
            float4 frame1 = channel.sampler->keyFrames[1].xyzw;
            float progress = 0;
            for (uint frameIndex = 1; frameIndex < channel.sampler->keyFrames.size(); frameIndex++) {
                ModelAnimationSamplerKeyFrame& keyFrame = channel.sampler->keyFrames[frameIndex];
                if (modelInstance.animationState.time <= keyFrame.time) {
                    ModelAnimationSamplerKeyFrame& keyFramePrevious = channel.sampler->keyFrames[frameIndex - 1];
                    frame0 = keyFramePrevious.xyzw;
                    frame1 = keyFrame.xyzw;
                    progress = ((float)modelInstance.animationState.time - keyFramePrevious.time) / (keyFrame.time - keyFramePrevious.time);
                    break;
                }
            }
            int64 nodeIndex = channel.node - &model.nodes[0];
            if (channel.type == AnimationChannelTypeTranslate) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    nodeLocalTransforms[nodeIndex].t = lerp(frame0.xyz(), frame1.xyz(), progress);
                } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    nodeLocalTransforms[nodeIndex].t = progress < 1.0f ? frame0.xyz() : frame1.xyz();
                }
            } else if (channel.type == AnimationChannelTypeRotate) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    nodeLocalTransforms[nodeIndex].r = slerp(frame0, frame1, progress);
                } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    nodeLocalTransforms[nodeIndex].r = progress < 1.0f ? frame0 : frame1;
                }
            } else if (channel.type == AnimationChannelTypeScale) {
                if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                    nodeLocalTransforms[nodeIndex].s = lerp(frame0.xyz(), frame1.xyz(), progress);
                } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                    nodeLocalTransforms[nodeIndex].s = progress < 1.0f ? frame0.xyz() : frame1.xyz();
                }
            }
        }
        for (uint nodeIndex = 0; nodeIndex < model.nodes.size(); nodeIndex++) {
            nodeLocalTransformMats[nodeIndex] = nodeLocalTransforms[nodeIndex].toMat();
        }
    }
    {
        ZoneScopedN("get jointTransformMats");
        for (ModelNode* rootNode : model.rootNodes) {
            assertDebug(rootNode->parent == nullptr);
            struct Node {
                ModelNode* node;
                XMMATRIX transformMat;
            };
            std::stack<Node> nodeStack;
            nodeStack.push({rootNode, nodeLocalTransformMats[rootNode - &model.nodes[0]]});
            while (!nodeStack.empty()) {
                Node parentNode = nodeStack.top();
                nodeStack.pop();
                nodeGlobalTransformMats[parentNode.node - &model.nodes[0]] = parentNode.transformMat;
                for (ModelNode* childNode : parentNode.node->children) {
                    int64 childNodeIndex = childNode - &model.nodes[0];
                    nodeStack.push({childNode, XMMatrixMultiply(nodeLocalTransformMats[childNodeIndex], parentNode.transformMat)});
                }
            }
        }
        for (uint jointIndex = 0; jointIndex < model.joints.size(); jointIndex++) {
            int64 nodeIndex = model.joints[jointIndex].node - &model.nodes[0];
            jointTransformMats[jointIndex] = XMMatrixTranspose(XMMatrixMultiply(model.joints[jointIndex].inverseBindMat, nodeGlobalTransformMats[nodeIndex]));
        }
    }
    memcpy(modelInstance.animationState.skinJointsBufferPtr, jointTransformMats.data(), vectorSizeof(jointTransformMats));

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (uint meshIndex = 0; meshIndex < model.meshes.size(); meshIndex++) {
        ModelMesh& mesh = model.meshes[meshIndex];
        D3D12MA::Allocation* verticesBuffer = modelInstance.animationState.meshVerticesBuffers[meshIndex];
        D3D12_RESOURCE_BARRIER verticesBufferBarriers[2] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = verticesBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = verticesBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}},
        };
        d3d.graphicsCmdList->ResourceBarrier(1, &verticesBufferBarriers[0]);
        d3d.graphicsCmdList->SetPipelineState(d3d.vertexSkinningPSO);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.vertexSkinningRootSig);
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(0, modelInstance.animationState.skinJointsBuffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, mesh.verticesBuffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRootUnorderedAccessView(2, verticesBuffer->GetResource()->GetGPUVirtualAddress());
        d3d.graphicsCmdList->SetComputeRoot32BitConstant(3, (uint)mesh.vertices.size(), 0);
        d3d.graphicsCmdList->Dispatch((uint)mesh.vertices.size() / 32 + 1, 1, 1);
        d3d.graphicsCmdList->ResourceBarrier(1, &verticesBufferBarriers[1]);

        std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
        geometryDescs.reserve(mesh.primitives.size());
        for (ModelPrimitive& primitive : mesh.primitives) {
            geometryDescs.push_back({
                .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                .Triangles = {
                    .IndexFormat = DXGI_FORMAT_R32_UINT,
                    .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
                    .IndexCount = (uint)primitive.indicesCount,
                    .VertexCount = (uint)primitive.verticesCount,
                    .IndexBuffer = mesh.indicesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.indicesBufferOffset * sizeof(uint),
                    .VertexBuffer = {.StartAddress = verticesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.verticesBufferOffset * sizeof(struct Vertex), .StrideInBytes = sizeof(struct Vertex)},
                },
            });
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {
            .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
            .NumDescs = (uint)geometryDescs.size(),
            .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
            .pGeometryDescs = geometryDescs.data(),
        };
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {.DestAccelerationStructureData = modelInstance.animationState.meshBlases[meshIndex]->GetResource()->GetGPUVirtualAddress(), .Inputs = blasInputs, .ScratchAccelerationStructureData = modelInstance.animationState.meshBlasScratches[meshIndex]->GetResource()->GetGPUVirtualAddress()};
        d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);
        barriers.push_back({.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = modelInstance.animationState.meshBlases[meshIndex]->GetResource()}});
    }
    d3d.graphicsCmdList->ResourceBarrier((uint)barriers.size(), barriers.data());
}

void addTlasInstance(ModelInstance& modelInstance, const XMMATRIX& objectTransform, WorldObjectType objectType, uint objectIndex) {
    ZoneScoped;
    Model& model = world.models[modelInstance.index];
    bool selected = false;
    if (editor.enable && editor.active) selected = editor.selectedObjectType == objectType && editor.selectedObjectIndex == objectIndex;
    TLASInstanceInfo tlasInstanceInfo = {.objectType = objectType, .objectIndex = objectIndex, .selected = selected, .skinJointsDescriptor = UINT32_MAX, .blasGeometriesOffset = (uint)world.blasGeometriesInfos.size()};
    bool animatedModel = !model.joints.empty();
    for (ModelNode& node : model.nodes) {
        if (!node.mesh) continue;
        int64 meshIndex = node.mesh - &model.meshes[0];
        D3D12MA::Allocation* meshBlas = animatedModel ? modelInstance.animationState.meshBlases[meshIndex] : node.mesh->blas;
        D3D12MA::Allocation* meshVerticesBuffer = animatedModel ? modelInstance.animationState.meshVerticesBuffers[meshIndex] : node.mesh->verticesBuffer;
        D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {.InstanceID = d3d.cbvSrvUavDescriptorCount, .InstanceMask = 0xff, .AccelerationStructure = meshBlas->GetResource()->GetGPUVirtualAddress()};
        XMMATRIX transform = node.globalTransform;
        transform = XMMatrixMultiply(transform, XMMatrixScaling(1, 1, -1)); // convert RH to LH
        transform = XMMatrixMultiply(transform, objectTransform);
        transform = XMMatrixTranspose(transform);
        memcpy(instanceDesc.Transform, &transform, sizeof(instanceDesc.Transform));
        world.tlasInstancesBuildInfos.push_back(instanceDesc);
        world.tlasInstancesInfos.push_back(tlasInstanceInfo);
        for (uint primitiveIndex = 0; primitiveIndex < node.mesh->primitives.size(); primitiveIndex++) {
            ModelPrimitive& primitive = node.mesh->primitives[primitiveIndex];
            D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.verticesBufferOffset, .NumElements = primitive.verticesCount, .StructureByteStride = sizeof(struct Vertex)}};
            D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.indicesBufferOffset, .NumElements = primitive.indicesCount, .StructureByteStride = sizeof(uint)}};
            d3d.appendSRVDescriptor(&vertexBufferDesc, meshVerticesBuffer->GetResource());
            d3d.appendSRVDescriptor(&indexBufferDesc, node.mesh->indicesBuffer->GetResource());
            if (primitive.material) {
                world.blasGeometriesInfos.push_back(BLASGeometryInfo{.baseColorFactor = primitive.material->baseColorFactor});
                if (primitive.material->baseColorTexture) {
                    d3d.appendSRVDescriptor(&primitive.material->baseColorTexture->image->srvDesc, primitive.material->baseColorTexture->image->gpuData->GetResource());
                } else {
                    d3d.appendSRVDescriptor(&d3d.defaultMaterialBaseColorImageSRVDesc, d3d.defaultMaterialBaseColorImage->GetResource());
                }
            } else {
                world.blasGeometriesInfos.push_back(BLASGeometryInfo{.baseColorFactor = {1, 1, 1, 1}});
                assert(false);
            }
        }
    }
}

D3D12_DISPATCH_RAYS_DESC fillRayTracingShaderTable(uint8* buffer, ID3D12Resource* bufferGPU, uint* bufferOffset, void* rayGenID, std::span<void*> missIDs, std::span<void*> hitGroupIDs) {
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    *bufferOffset = align(*bufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    memcpy(buffer + *bufferOffset, rayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    dispatchDesc.RayGenerationShaderRecord = {bufferGPU->GetGPUVirtualAddress() + *bufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
    *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.MissShaderTable = {bufferGPU->GetGPUVirtualAddress() + *bufferOffset, missIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* missID : missIDs) {
        memcpy(buffer + *bufferOffset, missID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    *bufferOffset = align(*bufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.HitGroupTable = {bufferGPU->GetGPUVirtualAddress() + *bufferOffset, hitGroupIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* hitGroupID : hitGroupIDs) {
        memcpy(buffer + *bufferOffset, hitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    return dispatchDesc;
}

void render() {
    ZoneScoped;
    d3d.graphicsQueueStartRecording();

    D3D12_RESOURCE_BARRIER renderTextureTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
    d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);

    d3d.cbvSrvUavDescriptorCount = 0;
    d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap);

    world.tlasInstancesBuildInfos.resize(0);
    world.tlasInstancesInfos.resize(0);
    world.blasGeometriesInfos.resize(0);

    d3d.stagingBufferOffset = 0;
    d3d.constantBufferOffset = 0;
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC renderTextureSRVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = 1}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC renderTextureUAVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3D12_CONSTANT_BUFFER_VIEW_DESC renderInfoCBVDesc = {.BufferLocation = d3d.constantBuffer->GetResource()->GetGPUVirtualAddress(), .SizeInBytes = align((uint)sizeof(struct RenderInfo), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT)};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasViewDesc = {.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .RaytracingAccelerationStructure = {.Location = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress()}};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasInstancesInfosDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.tlasInstancesInfosBuffer->GetSize() / sizeof(struct TLASInstanceInfo)), .StructureByteStride = sizeof(struct TLASInstanceInfo)}};
        D3D12_SHADER_RESOURCE_VIEW_DESC blasGeometriesInfosDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.blasGeometriesInfosBuffer->GetSize() / sizeof(struct BLASGeometryInfo)), .StructureByteStride = sizeof(struct BLASGeometryInfo)}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC readBackBufferDesc = {.ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct ReadBackBuffer)}};
        D3D12_SHADER_RESOURCE_VIEW_DESC collisionQueriesDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQuery)}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC collisionQueryResultsDesc = {.ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQueryResult)}};

        D3DDescriptor renderTextureSRVDescriptor = d3d.appendSRVDescriptor(&renderTextureSRVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderTextureUAVDescriptor = d3d.appendUAVDescriptor(&renderTextureUAVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderInfoDescriptor = d3d.appendCBVDescriptor(&renderInfoCBVDesc);
        D3DDescriptor tlasDescriptor = d3d.appendSRVDescriptor(&tlasViewDesc, nullptr);
        D3DDescriptor tlasInstancesInfosDescriptor = d3d.appendSRVDescriptor(&tlasInstancesInfosDesc, d3d.tlasInstancesInfosBuffer->GetResource());
        D3DDescriptor blasGeometriesInfosDescriptor = d3d.appendSRVDescriptor(&blasGeometriesInfosDesc, d3d.blasGeometriesInfosBuffer->GetResource());
        D3DDescriptor skyboxTextureDescriptor = d3d.appendSRVDescriptor(nullptr, world.skybox.hdriTexture->GetResource());
        D3DDescriptor readBackBufferDescriptor = d3d.appendUAVDescriptor(&readBackBufferDesc, d3d.readBackUAVBuffer->GetResource());
        D3DDescriptor imguiImageDescriptor = d3d.appendSRVDescriptor(nullptr, d3d.imguiImage->GetResource());
        D3DDescriptor collisionQueriesDescriptor = d3d.appendSRVDescriptor(&collisionQueriesDesc, d3d.collisionQueriesBuffer->GetResource());
        D3DDescriptor collisionQueryResultsDescriptor = d3d.appendUAVDescriptor(&collisionQueryResultsDesc, d3d.collisionQueryResultsUavBuffer->GetResource());
    }
    {
        float3 cameraPosition = world.player.camera.position;
        float3 cameraLookAt = world.player.camera.lookAt;
        float cameraFovVertical = 50;
        if (editor.enable && editor.active) {
            cameraPosition = editor.camera.position;
            cameraLookAt = editor.camera.lookAt;
            cameraFovVertical = editor.camera.fovVertical;
        }
        RenderInfo renderInfo = {
            .cameraViewMat = XMMatrixTranspose(XMMatrixInverse(nullptr, XMMatrixLookAtLH(cameraPosition.toXMVector(), cameraLookAt.toXMVector(), XMVectorSet(0, 1, 0, 0)))),
            .cameraProjMat = XMMatrixPerspectiveFovLH(radian(cameraFovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f),
            .resolution = {settings.renderW, settings.renderH},
            .mouseSelectPosition = {mouseSelectX, mouseSelectY},
            .hdr = settings.hdr,
            .frameTime = (float)frameTime,
        };
        assertDebug(d3d.constantBufferOffset == 0);
        memcpy(d3d.constantBufferPtr + d3d.constantBufferOffset, &renderInfo, sizeof(renderInfo));
        d3d.constantBufferOffset += sizeof(renderInfo);
    }
    {
        if (editor.enable && editor.active) {
            updateAnimatedModel(editor.player.model);
            for (StaticObject& obj : editor.staticObjects) updateAnimatedModel(obj.model);
            for (DynamicObject& obj : editor.dynamicObjects) updateAnimatedModel(obj.model);
            addTlasInstance(editor.player.model, editor.player.transform.toMat(), WorldObjectTypePlayer, 0);
            for (uint objIndex = 0; objIndex < editor.staticObjects.size(); objIndex++) {
                addTlasInstance(editor.staticObjects[objIndex].model, editor.staticObjects[objIndex].transform.toMat(), WorldObjectTypeStaticObject, objIndex);
            }
            for (uint objIndex = 0; objIndex < editor.dynamicObjects.size(); objIndex++) {
                addTlasInstance(editor.dynamicObjects[objIndex].model, editor.dynamicObjects[objIndex].transform.toMat(), WorldObjectTypeDynamicObject, objIndex);
            }
        } else {
            updateAnimatedModel(world.player.model);
            for (StaticObject& obj : world.staticObjects) updateAnimatedModel(obj.model);
            for (DynamicObject& obj : world.dynamicObjects) updateAnimatedModel(obj.model);
            addTlasInstance(world.player.model, world.player.transform.toMat(), WorldObjectTypePlayer, 0);
            for (uint objIndex = 0; objIndex < world.staticObjects.size(); objIndex++) {
                addTlasInstance(world.staticObjects[objIndex].model, world.staticObjects[objIndex].transform.toMat(), WorldObjectTypeStaticObject, objIndex);
            }
            for (uint objIndex = 0; objIndex < world.dynamicObjects.size(); objIndex++) {
                addTlasInstance(world.dynamicObjects[objIndex].model, world.dynamicObjects[objIndex].transform.toMat(), WorldObjectTypeDynamicObject, objIndex);
            }
        }
        {
            ZoneScopedN("buildTLAS");
            assert(vectorSizeof(world.tlasInstancesBuildInfos) < d3d.tlasInstancesBuildInfosBuffer->GetSize());
            assert(vectorSizeof(world.tlasInstancesInfos) < d3d.tlasInstancesInfosBuffer->GetSize());
            assert(vectorSizeof(world.blasGeometriesInfos) < d3d.blasGeometriesInfosBuffer->GetSize());
            memcpy(d3d.tlasInstancesBuildInfosBufferPtr, world.tlasInstancesBuildInfos.data(), vectorSizeof(world.tlasInstancesBuildInfos));
            memcpy(d3d.tlasInstancesInfosBufferPtr, world.tlasInstancesInfos.data(), vectorSizeof(world.tlasInstancesInfos));
            memcpy(d3d.blasGeometriesInfosBufferPtr, world.blasGeometriesInfos.data(), vectorSizeof(world.blasGeometriesInfos));

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, .NumDescs = (uint)world.tlasInstancesBuildInfos.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .InstanceDescs = d3d.tlasInstancesBuildInfosBuffer->GetResource()->GetGPUVirtualAddress()};
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
            d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);
            assert(prebuildInfo.ResultDataMaxSizeInBytes < d3d.tlasBuffer->GetSize());
            assert(prebuildInfo.ScratchDataSizeInBytes < d3d.tlasScratchBuffer->GetSize());

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress(), .Inputs = inputs, .ScratchAccelerationStructureData = d3d.tlasScratchBuffer->GetResource()->GetGPUVirtualAddress()};
            d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            D3D12_RESOURCE_BARRIER tlasBarrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = d3d.tlasBuffer->GetResource()}};
            d3d.graphicsCmdList->ResourceBarrier(1, &tlasBarrier);
        }
        {
            ZoneScopedN("renderScene");
            void* missIDs[2] = {d3d.renderScenePrimaryRayMissID, d3d.renderSceneSecondaryRayMissID};
            void* hitGroupIDs[2] = {d3d.renderScenePrimaryRayHitGroupID, d3d.renderSceneSecondaryRayHitGroupID};
            D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(d3d.constantBufferPtr, d3d.constantBuffer->GetResource(), &d3d.constantBufferOffset, d3d.renderSceneRayGenID, missIDs, hitGroupIDs);
            dispatchDesc.Width = settings.renderW, dispatchDesc.Height = settings.renderH, dispatchDesc.Depth = 1;
            assert(d3d.constantBufferOffset < d3d.constantBuffer->GetSize());

            d3d.graphicsCmdList->SetPipelineState1(d3d.renderScenePSO);
            d3d.graphicsCmdList->SetComputeRootSignature(d3d.renderSceneRootSig);
            d3d.graphicsCmdList->DispatchRays(&dispatchDesc);
        }
    }
    {
        D3D12_RESOURCE_BARRIER readBackBufferTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.readBackUAVBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        D3D12_RESOURCE_BARRIER collisionQueryResultsTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.collisionQueryResultsUavBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

        void* missIDs[1] = {d3d.collisionDetectionMissID};
        void* hitGroupIDs[1] = {d3d.collisionDetectionHitGroupID};
        D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(d3d.constantBufferPtr, d3d.constantBuffer->GetResource(), &d3d.constantBufferOffset, d3d.collisionDetectionRayGenID, missIDs, hitGroupIDs);
        dispatchDesc.Width = 1, dispatchDesc.Height = 1, dispatchDesc.Depth = 1;
        assert(d3d.constantBufferOffset < d3d.constantBuffer->GetSize());

        d3d.graphicsCmdList->SetPipelineState1(d3d.collisionDetection);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.collisionDetectionRootSig);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

        std::swap(readBackBufferTransition.Transition.StateBefore, readBackBufferTransition.Transition.StateAfter);
        std::swap(collisionQueryResultsTransition.Transition.StateBefore, collisionQueryResultsTransition.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

        d3d.graphicsCmdList->CopyBufferRegion(d3d.readBackBuffer->GetResource(), 0, d3d.readBackUAVBuffer->GetResource(), 0, sizeof(struct ReadBackBuffer));
        d3d.graphicsCmdList->CopyBufferRegion(d3d.collisionQueryResultsBuffer->GetResource(), 0, d3d.collisionQueryResultsUavBuffer->GetResource(), 0, d3d.collisionQueryResultsBuffer->GetSize());
    }
    {
        uint swapChainBackBufferIndex = d3d.swapChain->GetCurrentBackBufferIndex();
        D3D12_RESOURCE_BARRIER swapChainImageTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.swapChainImages[swapChainBackBufferIndex], .StateBefore = D3D12_RESOURCE_STATE_PRESENT, .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET}};
        d3d.graphicsCmdList->ResourceBarrier(1, &swapChainImageTransition);
        d3d.graphicsCmdList->OMSetRenderTargets(1, &d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], false, nullptr);
        // float swapChainClearColor[4] = {0, 0, 0, 0};
        // d3d.graphicsCmdList->ClearRenderTargetView(d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], swapChainClearColor, 0, nullptr);
        D3D12_VIEWPORT viewport = {0, 0, (float)settings.renderW, (float)settings.renderH, 0, 1};
        RECT scissor = {0, 0, (long)settings.renderW, (long)settings.renderH};
        d3d.graphicsCmdList->RSSetViewports(1, &viewport);
        d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
        {
            d3d.graphicsCmdList->SetPipelineState(d3d.postProcessPSO);
            d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.postProcessRootSig);
            std::swap(renderTextureTransition.Transition.StateBefore, renderTextureTransition.Transition.StateAfter);
            d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);
            d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
        }
        {
            d3d.graphicsCmdList->SetPipelineState(d3d.imguiPSO);
            float blendFactor[] = {0, 0, 0, 0};
            d3d.graphicsCmdList->OMSetBlendFactor(blendFactor);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.imguiRootSig);
            D3D12_VERTEX_BUFFER_VIEW vertBufferView = {d3d.imguiVertexBuffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiVertexBuffer->GetSize(), sizeof(ImDrawVert)};
            D3D12_INDEX_BUFFER_VIEW indexBufferView = {d3d.imguiIndexBuffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiIndexBuffer->GetSize(), DXGI_FORMAT_R16_UINT};
            assert(d3d.imguiVertexBuffer->GetResource()->Map(0, nullptr, (void**)&d3d.imguiVertexBufferPtr) == S_OK);
            assert(d3d.imguiIndexBuffer->GetResource()->Map(0, nullptr, (void**)&d3d.imguiIndexBufferPtr) == S_OK);
            d3d.graphicsCmdList->IASetVertexBuffers(0, 1, &vertBufferView);
            d3d.graphicsCmdList->IASetIndexBuffer(&indexBufferView);
            d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            uint vertBufferOffset = 0;
            uint indexBufferOffset = 0;
            const ImDrawData* drawData = ImGui::GetDrawData();
            for (int i = 0; i < drawData->CmdListsCount; i++) {
                const ImDrawList* dlist = drawData->CmdLists[i];
                uint verticesSize = dlist->VtxBuffer.Size * sizeof(ImDrawVert);
                uint indicesSize = dlist->IdxBuffer.Size * sizeof(ImDrawIdx);
                memcpy(d3d.imguiVertexBufferPtr + vertBufferOffset, dlist->VtxBuffer.Data, verticesSize);
                memcpy(d3d.imguiIndexBufferPtr + indexBufferOffset, dlist->IdxBuffer.Data, indicesSize);
                uint vertexIndex = vertBufferOffset / sizeof(ImDrawVert);
                uint indiceIndex = indexBufferOffset / sizeof(ImDrawIdx);
                for (int i = 0; i < dlist->CmdBuffer.Size; i++) {
                    const ImDrawCmd& dcmd = dlist->CmdBuffer[i];
                    D3D12_RECT scissor = {(long)dcmd.ClipRect.x, (long)dcmd.ClipRect.y, (long)dcmd.ClipRect.z, (long)dcmd.ClipRect.w};
                    d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
                    d3d.graphicsCmdList->DrawIndexedInstanced(dcmd.ElemCount, 1, indiceIndex, vertexIndex, 0);
                    indiceIndex += dcmd.ElemCount;
                }
                vertBufferOffset = vertBufferOffset + align(verticesSize, sizeof(ImDrawVert));
                indexBufferOffset = indexBufferOffset + align(indicesSize, sizeof(ImDrawIdx));
                assert(vertBufferOffset < d3d.imguiVertexBuffer->GetSize());
                assert(indexBufferOffset < d3d.imguiIndexBuffer->GetSize());
            }
        }
        std::swap(swapChainImageTransition.Transition.StateBefore, swapChainImageTransition.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(1, &swapChainImageTransition);
    }
    d3d.graphicsQueueSubmitRecording();
    assert(d3d.swapChain->Present(0, 0) == S_OK);
    d3d.graphicsQueueWait();
}

int main(int argc, char** argv) {
    assert(QueryPerformanceFrequency(&perfFrequency));
    if (commandLineContain(argc, argv, "showConsole")) { showConsole(); }
    settings.load();
    window.init();
    window.show();
    d3d.init(commandLineContain(argc, argv, "d3ddebug"));
    d3d.applySettings();
    world.load(assetsDir / "worlds/world.yaml");
    while (!quit) {
        QueryPerformanceCounter(&perfCounters[0]);
        ZoneScoped;
        mouseDeltaRaw[0] = 0, mouseDeltaRaw[1] = 0, mouseWheel = 0;
        MSG windowMsg;
        while (PeekMessageA(&windowMsg, (HWND)window.hwnd, 0, 0, PM_REMOVE)) {
            TranslateMessage(&windowMsg);
            DispatchMessageA(&windowMsg);
        }
        controller.getInputs();
        update();
        render();
        FrameMark;
        QueryPerformanceCounter(&perfCounters[1]);
        frameTime = (double)(perfCounters[1].QuadPart - perfCounters[0].QuadPart) / (double)perfFrequency.QuadPart;
    }
    world.save();
    settings.save();
    return EXIT_SUCCESS;
}
