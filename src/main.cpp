#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <atlbase.h>
#include <atlconv.h>
#include <cderr.h>
#include <commdlg.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <shellscalingapi.h>
#include <userenv.h>
#include <windows.h>
#include <windowsx.h>
#include <xinput.h>
#undef near
#undef far

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

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
// #define STB_DS_IMPLEMENTATION
// #include <stb/stb_ds.h>

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_OBSOLETE_KEYIO
#define IMGUI_USE_STB_SPRINTF
#include <imgui/imgui.cpp>
#include <imgui/imgui_draw.cpp>
#include <imgui/imgui_tables.cpp>
#include <imgui/imgui_widgets.cpp>
#include <imgui/imguizmo.cpp>
#undef snprintf
#undef vsnprintf

#include <d3d12ma/d3d12MemAlloc.cpp>

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

template <typename T>
void vectorDeleteElements(std::vector<T>& v) {
    auto iter = v.begin();
    while (iter != v.end()) {
        if (iter->toBeDeleted) {
            iter = v.erase(iter);
        } else {
            iter++;
        }
    }
}

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
    float3 scale = {1, 1, 1};
    float4 rotate = {0, 0, 0, 1};
    float3 translate = {0, 0, 0};

    void operator<<(ryml::ConstNodeRef node) { scale << node["scale"], rotate << node["rotate"], translate << node["translate"]; }
    void operator>>(ryml::NodeRef node) { scale >> node["scale"], rotate >> node["rotate"], translate >> node["translate"]; }
    XMMATRIX toMat() const { return XMMatrixAffineTransformation(scale.toXMVector(), XMVectorSet(0, 0, 0, 0), rotate.toXMVector(), translate.toXMVector()); }
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

std::vector<unsigned char> fileRead(const std::filesystem::path& path) {
    HANDLE hwnd = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hwnd != INVALID_HANDLE_VALUE) {
        DWORD size = GetFileSize(hwnd, nullptr);
        if (size != INVALID_FILE_SIZE) {
            std::vector<unsigned char> data(size);
            DWORD byteRead;
            if (ReadFile(hwnd, data.data(), size, &byteRead, nullptr) && byteRead == size) {
                CloseHandle(hwnd);
                return data;
            }
        }
        CloseHandle(hwnd);
    }
    return {};
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
};

static Settings settings = {};

void settingsLoad() {
    if (fileExists(exeDir / "settings.yaml")) {
        std::string yamlStr = fileReadStr(exeDir / "settings.yaml");
        ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
        ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
        yamlRoot["hdr"] >> settings.hdr;
        yamlRoot["windowX"] >> settings.windowX;
        yamlRoot["windowY"] >> settings.windowY;
        yamlRoot["windowW"] >> settings.windowW;
        yamlRoot["windowH"] >> settings.windowH;
    }
}

void settingsSave() {
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;
    yamlRoot["hdr"] << settings.hdr;
    yamlRoot["windowX"] << settings.windowX;
    yamlRoot["windowY"] << settings.windowY;
    yamlRoot["windowW"] << settings.windowW;
    yamlRoot["windowH"] << settings.windowH;
    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    assert(fileWriteStr(exeDir / "settings.yaml", yamlStr));
}

struct Window {
    HWND hwnd;
};

static Window window = {};

void windowUpdateSizes() {
    RECT windowRect;
    RECT clientRect;
    assert(GetWindowRect(window.hwnd, &windowRect));
    assert(GetClientRect(window.hwnd, &clientRect));
    settings.windowX = windowRect.left;
    settings.windowY = windowRect.top;
    settings.windowW = windowRect.right - windowRect.left;
    settings.windowH = windowRect.bottom - windowRect.top;
    settings.renderW = clientRect.right - clientRect.left;
    settings.renderH = clientRect.bottom - clientRect.top;
}

void windowInit() {
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
    window.hwnd = CreateWindowExA(0, windowClass.lpszClassName, nullptr, windowStyle, settings.windowX, settings.windowY, settings.windowW, settings.windowH, nullptr, nullptr, instanceHandle, nullptr);
    assert(window.hwnd);

    windowUpdateSizes();

    RAWINPUTDEVICE rawInputDevice = {.usUsagePage = 0x01, .usUsage = 0x02, /*.dwFlags = RIDEV_INPUTSINK, .hwndTarget = window.handle*/};
    assert(RegisterRawInputDevices(&rawInputDevice, 1, sizeof(rawInputDevice)));
}

struct DisplayMode {
    struct Resolution {
        uint width, height;
    };
    Resolution resolution;
    std::vector<DXGI_RATIONAL> refreshRates;
    bool operator==(Resolution res) { return res.width == resolution.width && res.height == resolution.height; }
    void addRefreshRate(DXGI_RATIONAL rate) {
        for (DXGI_RATIONAL& r : refreshRates) {
            if (r.Numerator == rate.Numerator && r.Denominator == rate.Denominator) { return; }
        }
        refreshRates.push_back(rate);
    }
};

struct D3D {
    IDXGIOutput6* dxgiOutput;
    IDXGIAdapter4* dxgiAdapter;
    std::vector<DisplayMode> displayModes;
    ID3D12Device5* device;

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
    char* stagingBufferPtr;
    D3D12_GPU_VIRTUAL_ADDRESS stagingBufferGPUAddress;

    D3D12MA::Allocation* constantBuffer;
    char* constantBufferPtr;

    D3D12MA::Allocation* readBackUavBuffer;
    D3D12MA::Allocation* readBackBuffer;
    char* readBackBufferPtr;

    D3D12MA::Allocation* renderTexture;
    DXGI_FORMAT renderTextureFormat;

    D3D12MA::Allocation* imguiTexture;
    D3D12MA::Allocation* imguiVertexBuffer;
    char* imguiVertexBufferPtr;
    D3D12MA::Allocation* imguiIndexBuffer;
    char* imguiIndexBufferPtr;

    D3D12MA::Allocation* tlasInstancesBuildInfosBuffer;
    char* tlasInstancesBuildInfosBufferPtr;
    D3D12MA::Allocation* tlasInstancesInfosBuffer;
    char* tlasInstancesInfosBufferPtr;
    D3D12MA::Allocation* tlasBuffer;
    D3D12MA::Allocation* tlasScratchBuffer;

    D3D12MA::Allocation* collisionQueriesBuffer;
    char* collisionQueriesBufferPtr;
    D3D12MA::Allocation* collisionQueryResultsUavBuffer;
    D3D12MA::Allocation* collisionQueryResultsBuffer;
    char* collisionQueryResultsBufferPtr;

    ID3D12StateObject* renderSceneSO;
    ID3D12StateObjectProperties* renderSceneSOProps;
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

    ID3D12PipelineState* postProcess;
    ID3D12RootSignature* postProcessRootSig;

    ID3D12PipelineState* imgui;
    ID3D12RootSignature* imguiRootSig;
};

static D3D d3d = {};

struct D3DDescriptor {
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

D3DDescriptor d3dAppendCbvDescriptor(D3D12_CONSTANT_BUFFER_VIEW_DESC* constantBufferViewDesc) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateConstantBufferView(constantBufferViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendSrvDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC* resourceViewDesc, ID3D12Resource* resource) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateShaderResourceView(resource, resourceViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendUavDescriptor(D3D12_UNORDERED_ACCESS_VIEW_DESC* unorderedAccessViewDesc, ID3D12Resource* resource) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateUnorderedAccessView(resource, nullptr, unorderedAccessViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

void d3dGraphicsQueueStartRecording() {
    assert(d3d.graphicsCmdAllocator->Reset() == S_OK);
    assert(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocator, nullptr) == S_OK);
}

void d3dGraphicsQueueSubmitRecording() {
    assert(d3d.graphicsCmdList->Close() == S_OK);
    d3d.graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.graphicsCmdList);
}

void d3dGraphicsQueueWait() {
    d3d.graphicsQueueFenceCounter += 1;
    d3d.graphicsQueue->Signal(d3d.graphicsQueueFence, d3d.graphicsQueueFenceCounter);
    if (d3d.graphicsQueueFence->GetCompletedValue() < d3d.graphicsQueueFenceCounter) {
        assert(d3d.graphicsQueueFence->SetEventOnCompletion(d3d.graphicsQueueFenceCounter, d3d.graphicsQueueFenceEvent) == S_OK);
        assert(WaitForSingleObjectEx(d3d.graphicsQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
    }
}

void d3dTransferQueueStartRecording() {
    assert(d3d.transferCmdAllocator->Reset() == S_OK);
    assert(d3d.transferCmdList->Reset(d3d.transferCmdAllocator, nullptr) == S_OK);
}

void d3dTransferQueueSubmitRecording() {
    assert(d3d.transferCmdList->Close() == S_OK);
    d3d.transferQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.transferCmdList);
}

void d3dTransferQueueWait() {
    d3d.transferQueueFenceCounter += 1;
    d3d.transferQueue->Signal(d3d.transferQueueFence, d3d.transferQueueFenceCounter);
    if (d3d.transferQueueFence->GetCompletedValue() < d3d.transferQueueFenceCounter) {
        assert(d3d.transferQueueFence->SetEventOnCompletion(d3d.transferQueueFenceCounter, d3d.transferQueueFenceEvent) == S_OK);
        assert(WaitForSingleObjectEx(d3d.transferQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
    }
}

void d3dMessageCallback(D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context) {
    if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR) {
        __debugbreak();
    }
}

void d3dInit(bool debug) {
    uint factoryFlags = 0;
    if (debug) {
        factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
        ID3D12Debug1* debug;
        assert(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)) == S_OK);
        debug->EnableDebugLayer();
        // d3d.debug->SetEnableGPUBasedValidation(true);
        // d3d.debug->SetEnableSynchronizedCommandQueueValidation(true);
    }

    IDXGIFactory7* dxgiFactory = nullptr;
    DXGI_ADAPTER_DESC dxgiAdapterDesc = {};
    DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};

    assert(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&dxgiFactory)) == S_OK);
    assert(dxgiFactory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&d3d.dxgiAdapter)) == S_OK);
    assert(D3D12CreateDevice(d3d.dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d.device)) == S_OK);
    if (debug) {
        ID3D12InfoQueue1* infoQueue;
        assert(d3d.device->QueryInterface(IID_PPV_ARGS(&infoQueue)) == S_OK);
        DWORD callbackCookie;
        assert(infoQueue->RegisterMessageCallback(d3dMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &callbackCookie) == S_OK);
    }
    assert(d3d.dxgiAdapter->GetDesc(&dxgiAdapterDesc) == S_OK);
    assert(d3d.dxgiAdapter->EnumOutputs(0, (IDXGIOutput**)&d3d.dxgiOutput) == S_OK);
    assert(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc) == S_OK);
    settings.hdr = (dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS resourceBindingTier = {};
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 rayTracing = {};
        // D3D12_FEATURE_DATA_D3D12_OPTIONS16 gpuUploadHeap = {};
        assert(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &resourceBindingTier, sizeof(resourceBindingTier)) == S_OK);
        assert(d3d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)) == S_OK);
        assert(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &rayTracing, sizeof(rayTracing)) == S_OK);
        // assert(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &gpuUploadHeap, sizeof(gpuUploadHeap)));
        assert(resourceBindingTier.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        assert(shaderModel.HighestShaderModel == D3D_SHADER_MODEL_6_6);
        assert(rayTracing.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
        // assert(gpuUploadHeap.GPUUploadHeapSupported);
    }
    {
        D3D12_COMMAND_QUEUE_DESC graphicsQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
        assert(d3d.device->CreateCommandQueue(&graphicsQueueDesc, IID_PPV_ARGS(&d3d.graphicsQueue)) == S_OK);
        D3D12_COMMAND_QUEUE_DESC transferQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
        assert(d3d.device->CreateCommandQueue(&transferQueueDesc, IID_PPV_ARGS(&d3d.transferQueue)) == S_OK);

        assert(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.graphicsCmdAllocator)) == S_OK);
        assert(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.transferCmdAllocator)) == S_OK);

        assert(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.graphicsCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.graphicsCmdList)) == S_OK);
        assert(d3d.graphicsCmdList->Close() == S_OK);
        assert(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.transferCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.transferCmdList)) == S_OK);
        assert(d3d.transferCmdList->Close() == S_OK);

        assert(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.graphicsQueueFence)) == S_OK);
        d3d.graphicsQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
        assert(d3d.graphicsQueueFenceEvent);
        assert(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.transferQueueFence)) == S_OK);
        d3d.transferQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
        assert(d3d.transferQueueFenceEvent);
    }
    {
        d3d.swapChainFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
        uint dxgiModeCount = 0;
        d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, nullptr);
        std::vector<DXGI_MODE_DESC> dxgiModes(dxgiModeCount);
        d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, dxgiModes.data());
        for (DXGI_MODE_DESC& dxgiMode : dxgiModes) {
            DisplayMode::Resolution res = {dxgiMode.Width, dxgiMode.Height};
            auto modeIter = std::find(d3d.displayModes.begin(), d3d.displayModes.end(), res);
            if (modeIter == d3d.displayModes.end()) {
                d3d.displayModes.push_back({{dxgiMode.Width, dxgiMode.Height}, {dxgiMode.RefreshRate}});
            } else {
                modeIter->addRefreshRate(dxgiMode.RefreshRate);
            }
        }
        DXGI_SWAP_CHAIN_DESC1 desc = {
            .Width = (uint)settings.renderW,
            .Height = (uint)settings.renderH,
            .Format = d3d.swapChainFormat,
            .SampleDesc = {.Count = 1},
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER,
            .BufferCount = countof(d3d.swapChainImages),
            .Scaling = DXGI_SCALING_NONE,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
            .AlphaMode = DXGI_ALPHA_MODE_IGNORE,
            .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH,
        };
        assert(dxgiFactory->CreateSwapChainForHwnd(d3d.graphicsQueue, window.hwnd, &desc, nullptr, nullptr, (IDXGISwapChain1**)&d3d.swapChain) == S_OK);

        DXGI_COLOR_SPACE_TYPE colorSpace = settings.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        assert(d3d.swapChain->SetColorSpace1(colorSpace) == S_OK);
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
            assert(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image)) == S_OK);
            (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
        }

        dxgiFactory->MakeWindowAssociation(window.hwnd, DXGI_MWA_NO_WINDOW_CHANGES); // disable alt-enter
    }
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 16, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        assert(d3d.device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&d3d.rtvDescriptorHeap)) == S_OK);
        d3d.rtvDescriptorCount = 0;
        d3d.rtvDescriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
            uint offset = d3d.rtvDescriptorSize * d3d.rtvDescriptorCount;
            d3d.swapChainImageRTVDescriptors[imageIndex] = {d3d.rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
            d3d.device->CreateRenderTargetView(*image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
            d3d.rtvDescriptorCount += 1;
        }

        d3d.cbvSrvUavDescriptorCapacity = 1024;
        d3d.cbvSrvUavDescriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = d3d.cbvSrvUavDescriptorCapacity, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        assert(d3d.device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&d3d.cbvSrvUavDescriptorHeap)) == S_OK);
    }
    {
        D3D12MA::ALLOCATOR_DESC allocatorDesc = {.Flags = D3D12MA::ALLOCATOR_FLAG_NONE, .pDevice = d3d.device, .pAdapter = d3d.dxgiAdapter};
        assert(D3D12MA::CreateAllocator(&allocatorDesc, &d3d.allocator) == S_OK);
    }
    {
        struct BufferDesc {
            D3D12MA::Allocation** buffer;
            char** bufferPtr;
            uint size;
            D3D12_HEAP_TYPE heapType;
            D3D12_RESOURCE_FLAGS flags;
            D3D12_RESOURCE_STATES initState;
            const wchar_t* name;
        } descs[] = {
            {&d3d.stagingBuffer, &d3d.stagingBufferPtr, megabytes(256), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"stagingBuffer"},
            {&d3d.constantBuffer, &d3d.constantBufferPtr, megabytes(4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer"},
            {&d3d.tlasInstancesBuildInfosBuffer, &d3d.tlasInstancesBuildInfosBufferPtr, megabytes(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfosBuffer"},
            {&d3d.tlasInstancesInfosBuffer, &d3d.tlasInstancesInfosBufferPtr, megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesExtraInfosBuffer"},
            {&d3d.tlasBuffer, nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlasBuffer"},
            {&d3d.tlasScratchBuffer, nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratchBuffer"},
            {&d3d.imguiVertexBuffer, &d3d.imguiVertexBufferPtr, megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer"},
            {&d3d.imguiIndexBuffer, &d3d.imguiIndexBufferPtr, megabytes(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer"},
            {&d3d.readBackUavBuffer, nullptr, megabytes(2), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"readBackUavBuffer"},
            {&d3d.readBackBuffer, &d3d.readBackBufferPtr, megabytes(2), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"readBackBuffer"},
            {&d3d.collisionQueriesBuffer, &d3d.collisionQueriesBufferPtr, megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueriesBuffer"},
            {&d3d.collisionQueryResultsUavBuffer, nullptr, megabytes(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUavBuffer"},
            {&d3d.collisionQueryResultsBuffer, &d3d.collisionQueryResultsBufferPtr, megabytes(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResultsBuffer"},
        };
        for (BufferDesc& desc : descs) {
            D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = desc.size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = desc.flags};
            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = desc.heapType};
            assert(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, desc.initState, nullptr, desc.buffer, {}, nullptr) == S_OK);
            (*desc.buffer)->GetResource()->SetName(desc.name);
            if (desc.bufferPtr) {
                assert((*desc.buffer)->GetResource()->Map(0, nullptr, (void**)desc.bufferPtr) == S_OK);
            }
        }
        d3d.stagingBufferGPUAddress = d3d.stagingBuffer->GetResource()->GetGPUVirtualAddress();
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
        assert(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr) == S_OK);
        d3d.renderTexture->GetResource()->SetName(L"renderTexture");
        d3d.renderTextureFormat = renderTextureDesc.Format;
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

        uint8* imguiTextureData;
        int imguiTextureWidth, imguiTextureHeight;
        ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&imguiTextureData, &imguiTextureWidth, &imguiTextureHeight);
        struct TextureDesc {
            D3D12MA::Allocation** texture;
            D3D12_RESOURCE_DESC desc;
            uint8* data;
            const wchar_t* name;
        } descs[] = {
            {.texture = &d3d.imguiTexture,
             .desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint64)imguiTextureWidth, .Height = (uint)imguiTextureHeight, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}},
             .data = imguiTextureData,
             .name = L"imguiTexture"},
        };
        for (TextureDesc& desc : descs) {
            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            assert(d3d.allocator->CreateResource(&allocationDesc, &desc.desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, desc.texture, {}, nullptr) == S_OK);
            (*desc.texture)->GetResource()->SetName(desc.name);
        }
        uint64 stagingBufferOffset = 0;
        d3dTransferQueueStartRecording();
        for (TextureDesc& desc : descs) {
            const uint maxMipmapCount = 16;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipmapFootPrints[maxMipmapCount] = {};
            uint mipmapRowCounts[maxMipmapCount] = {};
            uint64 mipmapRowSizes[maxMipmapCount] = {};
            uint64 textureSize = 0;
            d3d.device->GetCopyableFootprints(&desc.desc, 0, desc.desc.MipLevels, 0, mipmapFootPrints, mipmapRowCounts, mipmapRowSizes, &textureSize);
            assert(textureSize < d3d.stagingBuffer->GetSize());
            if ((stagingBufferOffset + textureSize) >= d3d.stagingBuffer->GetSize()) {
                d3dTransferQueueSubmitRecording();
                d3dTransferQueueWait();
                d3dTransferQueueStartRecording();
                stagingBufferOffset = 0;
            }
            uint64 dataOffset = 0;
            for (uint mip = 0; mip < desc.desc.MipLevels; mip++) {
                uint64 mipmapStagingBufferOffset = stagingBufferOffset + mipmapFootPrints[mip].Offset;
                for (uint row = 0; row < mipmapRowCounts[mip]; row++) {
                    memcpy(d3d.stagingBufferPtr + mipmapStagingBufferOffset, desc.data + dataOffset, mipmapRowSizes[mip]);
                    mipmapStagingBufferOffset += mipmapFootPrints[mip].Footprint.RowPitch;
                    dataOffset += mipmapRowSizes[mip];
                }
                D3D12_TEXTURE_COPY_LOCATION dstCopyLocation = {.pResource = (*desc.texture)->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, .SubresourceIndex = mip};
                D3D12_TEXTURE_COPY_LOCATION srcCopyLocation = {.pResource = d3d.stagingBuffer->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = {.Offset = stagingBufferOffset + mipmapFootPrints[mip].Offset, .Footprint = mipmapFootPrints[mip].Footprint}};
                d3d.transferCmdList->CopyTextureRegion(&dstCopyLocation, 0, 0, 0, &srcCopyLocation, nullptr);
                D3D12_RESOURCE_BARRIER textureBarrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = (*desc.texture)->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}};
                d3d.transferCmdList->ResourceBarrier(1, &textureBarrier);
            }
            stagingBufferOffset = align(stagingBufferOffset + textureSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        }
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }
}

void d3dCompilePipelines() {
    {
        std::vector<unsigned char> rtByteCode = fileRead(exeDir / "renderScene.cso");
        assert(rtByteCode.size() > 0);
        assert(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.renderSceneRootSig)) == S_OK);
        D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"primaryRayMiss"}, {L"primaryRayHitGroup"}, {L"primaryRayClosestHit"}, {L"secondaryRayMiss"}, {L"secondaryRayHitGroup"}, {L"secondaryRayClosestHit"}};
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
        D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
        D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
        assert(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.renderSceneSO)) == S_OK);
        assert(d3d.renderSceneSO->QueryInterface(IID_PPV_ARGS(&d3d.renderSceneSOProps)) == S_OK);
        assert(d3d.renderSceneRayGenID = d3d.renderSceneSOProps->GetShaderIdentifier(L"rayGen"));
        assert(d3d.renderScenePrimaryRayMissID = d3d.renderSceneSOProps->GetShaderIdentifier(L"primaryRayMiss"));
        assert(d3d.renderScenePrimaryRayHitGroupID = d3d.renderSceneSOProps->GetShaderIdentifier(L"primaryRayHitGroup"));
        assert(d3d.renderSceneSecondaryRayMissID = d3d.renderSceneSOProps->GetShaderIdentifier(L"secondaryRayMiss"));
        assert(d3d.renderSceneSecondaryRayHitGroupID = d3d.renderSceneSOProps->GetShaderIdentifier(L"secondaryRayHitGroup"));
    }
    {
        std::vector<unsigned char> rtByteCode = fileRead(exeDir / "collisionDetection.cso");
        assert(rtByteCode.size() > 0);
        assert(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.collisionDetectionRootSig)) == S_OK);
        D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"miss"}, {L"hitGroup"}, {L"closestHit"}};
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
        D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
        D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
        assert(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.collisionDetection)) == S_OK);
        assert(d3d.collisionDetection->QueryInterface(IID_PPV_ARGS(&d3d.collisionDetectionProps)) == S_OK);
        assert(d3d.collisionDetectionRayGenID = d3d.collisionDetectionProps->GetShaderIdentifier(L"rayGen"));
        assert(d3d.collisionDetectionMissID = d3d.collisionDetectionProps->GetShaderIdentifier(L"miss"));
        assert(d3d.collisionDetectionHitGroupID = d3d.collisionDetectionProps->GetShaderIdentifier(L"hitGroup"));
    }
    {
        std::vector<unsigned char> vsByteCode = fileRead(exeDir / "postProcessVS.cso");
        std::vector<unsigned char> psByteCode = fileRead(exeDir / "postProcessPS.cso");
        assert(vsByteCode.size() > 0);
        assert(psByteCode.size() > 0);
        assert(d3d.device->CreateRootSignature(0, psByteCode.data(), psByteCode.size(), IID_PPV_ARGS(&d3d.postProcessRootSig)) == S_OK);
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
        assert(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.postProcess)) == S_OK);
    }
    {
        std::vector<unsigned char> vsByteCode = fileRead(exeDir / "ImGuiVS.cso");
        std::vector<unsigned char> psByteCode = fileRead(exeDir / "ImGuiPS.cso");
        assert(vsByteCode.size() > 0);
        assert(psByteCode.size() > 0);
        assert(d3d.device->CreateRootSignature(0, vsByteCode.data(), vsByteCode.size(), IID_PPV_ARGS(&d3d.imguiRootSig)) == S_OK);
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
        assert(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.imgui)) == S_OK);
    }
}

void d3dResizeSwapChain(uint width, uint height) {
    d3dGraphicsQueueWait();
    for (ID3D12Resource* image : d3d.swapChainImages) { image->Release(); }
    assert(d3d.swapChain->ResizeBuffers(countof(d3d.swapChainImages), width, height, d3d.swapChainFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH) == S_OK);
    for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
        ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
        assert(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image)) == S_OK);
        (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
        d3d.device->CreateRenderTargetView(*image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
    }
    d3d.renderTexture->Release();
    D3D12_RESOURCE_DESC renderTextureDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = width, .Height = height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = d3d.renderTextureFormat, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    assert(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr) == S_OK);
    d3d.renderTexture->GetResource()->SetName(L"renderTexture");
}

static const uint64 settingsHDR = 1;
static const uint64 settingWindowMode = 1 << 1;
static const uint64 settingAll = ~0;

void applySettings(uint64 settingBits) {
    if (settingBits & settingsHDR) {
        DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};
        assert(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc) == S_OK);
        if (settings.hdr && dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
            assert(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) == S_OK);
        } else {
            assert(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) == S_OK);
        }
    }
    if (settingBits & settingWindowMode) {
        if (settings.windowMode == WindowModeWindowed) {
            assert(d3d.swapChain->SetFullscreenState(false, nullptr) == S_OK);
            DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
            MONITORINFO mi = {.cbSize = sizeof(mi)};
            assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
            assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW) != 0);
            assert(SetWindowPos(window.hwnd, NULL, settings.windowX, settings.windowY, settings.windowW, settings.windowH, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
        } else if (settings.windowMode == WindowModeBorderless) {
            assert(d3d.swapChain->SetFullscreenState(false, nullptr) == S_OK);
            DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
            MONITORINFO mi = {.cbSize = sizeof(mi)};
            assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
            assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW) != 0);
            assert(SetWindowPos(window.hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
        } else if (settings.windowMode == WindowModeFullscreen) {
            DXGI_MODE_DESC dxgiMode = {.Width = settings.windowW, .Height = settings.windowH, .RefreshRate = settings.refreshRate, .Format = d3d.swapChainFormat};
            assert(d3d.swapChain->ResizeTarget(&dxgiMode) == S_OK);
            assert(d3d.swapChain->SetFullscreenState(true, nullptr) == S_OK);
        }
    }
}

#include "sceneStructs.h"

struct CameraEditor {
    float3 position;
    float3 lookAt;
    float2 pitchYaw;
    float fovVertical;
    float sensitivity;
    float rotationSensitivity;
    float controllerSensitivity;

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
};

struct ModelTexture {
    ModelImage* image = nullptr;
};

struct ModelMaterial {
    std::string name;
    float4 baseColor = {0, 0, 0, 0};
    ModelTexture* baseColorTexture = nullptr;
    ;
};

struct ModelPrimitive {
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    D3D12MA::Allocation* verticesBuffer = nullptr;
    D3D12MA::Allocation* indicesBuffer = nullptr;
    ModelMaterial* material = nullptr;
};

struct ModelMesh {
    std::string name;
    std::vector<ModelPrimitive> primitives;
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
    ModelAnimationChannelType type;
    ModelAnimationSampler* sampler = nullptr;
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
    std::vector<std::vector<D3D12MA::Allocation*>> meshPrimitivesVerticesBuffers;
    std::vector<D3D12MA::Allocation*> meshBlases;
    std::vector<D3D12MA::Allocation*> meshBlasScratches;
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
};

struct StaticObject {
    std::string name;
    ModelInstance model;
    Transform transform;
    bool toBeDeleted;
};

struct DynamicObject {
    std::string name;
    ModelInstance model;
    Transform transform;
    bool toBeDeleted;
};

struct Skybox {
    std::filesystem::path hdriTextureFilePath;
    D3D12MA::Allocation* hdriTexture;
};

enum SceneEditorUndoType {
    SceneEditorUndoTypeObjectDeletion
};

struct SceneEditorUndoObjectDeletion {
    SceneObjectType objectType;
    void* object;
};

struct SceneEditorUndo {
    SceneEditorUndoType type;
    union {
        SceneEditorUndoObjectDeletion* objectDeletion;
    };
};

struct SceneEditor {
    CameraEditor camera;
    bool cameraMoving;
    SceneObjectType selectedObjectType;
    uint selectedObjectIndex;
    std::stack<SceneEditorUndo> undos;
};

struct Scene {
    std::filesystem::path filePath;
    Player player;
    std::vector<Model> models;
    std::vector<StaticObject> staticObjects;
    std::vector<DynamicObject> dynamicObjects;
    Skybox skybox;
    std::vector<Light> lights;

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> tlasInstancesBuildInfos;
    std::vector<TLASInstanceInfo> tlasInstancesInfos;

    SceneEditor editor;
};

static Scene scene = {};

ModelInstance sceneLoadModelGLTF(const std::filesystem::path& filePath) {
    auto modelIter = std::find_if(scene.models.begin(), scene.models.end(), [&](auto& model) { return model.filePath == filePath; });
    if (modelIter == scene.models.end()) {
        Model& model = scene.models.emplace_back();
        modelIter = scene.models.end() - 1;

        const std::filesystem::path gltfFilePath = assetsDir / filePath;
        const std::filesystem::path gltfFileFolderPath = gltfFilePath.parent_path();
        cgltf_options gltfOptions = {};
        cgltf_data* gltfData = nullptr;
        cgltf_result gltfParseFileResult = cgltf_parse_file(&gltfOptions, gltfFilePath.string().c_str(), &gltfData);
        cgltf_result gltfLoadBuffersResult = cgltf_load_buffers(&gltfOptions, gltfData, gltfFilePath.string().c_str());
        assert(gltfParseFileResult == cgltf_result_success);
        assert(gltfLoadBuffersResult == cgltf_result_success);
        model.filePath = filePath;
        model.gltfData = gltfData;

        d3dTransferQueueStartRecording();
        uint stagingBufferOffset = 0;

        model.images.reserve(gltfData->images_count);
        for (uint imageIndex = 0; imageIndex < gltfData->images_count; imageIndex++) {
            cgltf_image& gltfImage = gltfData->images[imageIndex];
            ModelImage& image = model.images.emplace_back();
            std::filesystem::path imageFilePath = gltfFileFolderPath / gltfImage.uri;
            std::filesystem::path imageDDSFilePath = imageFilePath;
            imageDDSFilePath.replace_extension(".dds");
            if (std::filesystem::exists(imageDDSFilePath)) {
                ScratchImage scratchImage;
                assert(LoadFromDDSFile(imageDDSFilePath.c_str(), DDS_FLAGS_NONE, nullptr, scratchImage) == S_OK);
                D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC resourceDesc = {
                    .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    .Width = (uint)scratchImage.GetMetadata().width,
                    .Height = (uint)scratchImage.GetMetadata().height,
                    .DepthOrArraySize = (uint16)scratchImage.GetMetadata().arraySize,
                    .MipLevels = (uint16)scratchImage.GetMetadata().mipLevels,
                    .Format = scratchImage.GetMetadata().format,
                    .SampleDesc = {.Count = 1},
                };
                assert(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image.gpuData, {}, nullptr) == S_OK);
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
                uint64 totalSize;
                d3d.device->GetCopyableFootprints(&resourceDesc, 0, (uint)scratchImage.GetImageCount(), 0, mipFootprints, nullptr, nullptr, &totalSize);
                stagingBufferOffset = align(stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
                assert(stagingBufferOffset + totalSize < d3d.stagingBuffer->GetSize());
                for (uint mipIndex = 0; mipIndex < scratchImage.GetImageCount(); mipIndex++) {
                    const Image& mip = scratchImage.GetImages()[mipIndex];
                    memcpy(d3d.stagingBufferPtr + stagingBufferOffset + mipFootprints[mipIndex].Offset, mip.pixels, mip.slicePitch);
                    D3D12_TEXTURE_COPY_LOCATION copyDst = {.pResource = image.gpuData->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, .SubresourceIndex = mipIndex};
                    D3D12_TEXTURE_COPY_LOCATION copySrc = {.pResource = d3d.stagingBuffer->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = mipFootprints[mipIndex]};
                    d3d.transferCmdList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);
                }
                stagingBufferOffset += totalSize;
                D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = image.gpuData->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
                d3d.transferCmdList->ResourceBarrier(1, &barrier);
            } else {
                assert(false && "Implement");
                int width, height, channel;
                unsigned char* imageData = stbi_load(imageFilePath.string().c_str(), &width, &height, &channel, 4);
                assert(imageData);
                stbi_image_free(imageData);
            }
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
            material.baseColor = float4(gltfMaterial.pbr_metallic_roughness.base_color_factor);
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
                assertDebug(gltfPrimitive.type == cgltf_primitive_type_triangles);
                assertDebug(indices && positions && normals);
                assertDebug(indices->count % 3 == 0 && indices->type == cgltf_type_scalar && (indices->component_type == cgltf_component_type_r_16u || indices->component_type == cgltf_component_type_r_32u));
                assertDebug(positions->type == cgltf_type_vec3 && positions->component_type == cgltf_component_type_r_32f);
                assertDebug(normals->count == positions->count && normals->type == cgltf_type_vec3 && normals->component_type == cgltf_component_type_r_32f);
                if (uvs) assertDebug(uvs->count == positions->count && uvs->component_type == cgltf_component_type_r_32f && uvs->type == cgltf_type_vec2);
                if (jointIndices) assertDebug(jointIndices->count == positions->count && (jointIndices->component_type == cgltf_component_type_r_16u || jointIndices->component_type == cgltf_component_type_r_8u) && jointIndices->type == cgltf_type_vec4 && (jointIndices->stride == 8 || jointIndices->stride == 4));
                if (jointWeights) assertDebug(jointWeights->count == positions->count && jointWeights->component_type == cgltf_component_type_r_32f && jointWeights->type == cgltf_type_vec4 && jointWeights->stride == 16);
                float3* positionsBuffer = (float3*)((uint8*)(positions->buffer_view->buffer->data) + positions->offset + positions->buffer_view->offset);
                float3* normalsBuffer = (float3*)((uint8*)(normals->buffer_view->buffer->data) + normals->offset + normals->buffer_view->offset);
                void* indicesBuffer = (uint8*)(indices->buffer_view->buffer->data) + indices->offset + indices->buffer_view->offset;
                float2* uvsBuffer = uvs ? (float2*)((uint8*)(uvs->buffer_view->buffer->data) + uvs->offset + uvs->buffer_view->offset) : nullptr;
                void* jointIndicesBuffer = jointIndices ? (uint8*)(jointIndices->buffer_view->buffer->data) + jointIndices->offset + jointIndices->buffer_view->offset : nullptr;
                float4* jointWeightsBuffer = jointWeights ? (float4*)((uint8*)(jointWeights->buffer_view->buffer->data) + jointWeights->offset + jointWeights->buffer_view->offset) : nullptr;

                ModelPrimitive& primitive = mesh.primitives.emplace_back();
                primitive.vertices.reserve(positions->count);
                primitive.indices.reserve(indices->count);
                for (uint vertexIndex = 0; vertexIndex < positions->count; vertexIndex++) {
                    Vertex vertex = {.position = positionsBuffer[vertexIndex], .normal = normalsBuffer[vertexIndex]};
                    if (uvsBuffer) vertex.uv = uvsBuffer[vertexIndex];
                    if (jointIndicesBuffer) {
                        if (jointIndices->component_type == cgltf_component_type_r_16u) {
                            vertex.joints = ((uint16_4*)jointIndicesBuffer)[vertexIndex];
                        } else {
                            vertex.joints = ((uint8_4*)jointIndicesBuffer)[vertexIndex];
                        }
                    }
                    if (jointWeightsBuffer) vertex.jointWeights = jointWeightsBuffer[vertexIndex];
                    primitive.vertices.push_back(vertex);
                }
                if (indices->component_type == cgltf_component_type_r_16u) {
                    primitive.indices.append_range(std::span((uint16*)indicesBuffer, indices->count));
                } else if (indices->component_type == cgltf_component_type_r_32u) {
                    primitive.indices.append_range(std::span((uint*)indicesBuffer, indices->count));
                }
                if (gltfPrimitive.material) {
                    primitive.material = &model.materials[gltfPrimitive.material - gltfData->materials];
                }

                D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
                resourceDesc.Width = vectorSizeof(primitive.vertices);
                assert(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &primitive.verticesBuffer, {}, nullptr) == S_OK);
                resourceDesc.Width = vectorSizeof(primitive.indices);
                assert(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &primitive.indicesBuffer, {}, nullptr) == S_OK);

                memcpy(d3d.stagingBufferPtr + stagingBufferOffset, primitive.vertices.data(), vectorSizeof(primitive.vertices));
                d3d.transferCmdList->CopyBufferRegion(primitive.verticesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), stagingBufferOffset, vectorSizeof(primitive.vertices));
                stagingBufferOffset += vectorSizeof(primitive.vertices);
                memcpy(d3d.stagingBufferPtr + stagingBufferOffset, primitive.indices.data(), vectorSizeof(primitive.indices));
                d3d.transferCmdList->CopyBufferRegion(primitive.indicesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), stagingBufferOffset, vectorSizeof(primitive.indices));
                stagingBufferOffset += vectorSizeof(primitive.indices);
                assert(stagingBufferOffset < d3d.stagingBuffer->GetSize());
            }
            std::vector<D3D12_RESOURCE_BARRIER> bufferBarriers;
            bufferBarriers.reserve(mesh.primitives.size() * 2);
            for (ModelPrimitive& primitive : mesh.primitives) {
                D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
                barrier.Transition.pResource = primitive.verticesBuffer->GetResource();
                bufferBarriers.push_back(barrier);
                barrier.Transition.pResource = primitive.indicesBuffer->GetResource();
                bufferBarriers.push_back(barrier);
            }
            d3d.transferCmdList->ResourceBarrier((uint)bufferBarriers.size(), bufferBarriers.data());

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
                        .IndexCount = (uint)primitive.indices.size(),
                        .VertexCount = (uint)primitive.vertices.size(),
                        .IndexBuffer = primitive.indicesBuffer->GetResource()->GetGPUVirtualAddress(),
                        .VertexBuffer = {primitive.verticesBuffer->GetResource()->GetGPUVirtualAddress(), sizeof(Vertex)},
                    },
                };
            }
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE, .NumDescs = (uint)geometryDescs.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .pGeometryDescs = geometryDescs.data()};
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
            d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC blasBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            blasBufferDesc.Width = prebuildInfo.ResultDataMaxSizeInBytes;
            assert(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &mesh.blas, {}, nullptr) == S_OK);
            blasBufferDesc.Width = prebuildInfo.ScratchDataSizeInBytes;
            assert(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &mesh.blasScratch, {}, nullptr) == S_OK);
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
                if (gltfNode.has_scale) localTransform.scale = float3(gltfNode.scale);
                if (gltfNode.has_rotation) localTransform.rotate = float4(gltfNode.rotation);
                if (gltfNode.has_translation) localTransform.translate = float3(gltfNode.translation);
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
        for (cgltf_node* gltfNode : std::span(gltfData->scene[0].nodes, gltfData->scene[0].nodes_count)) {
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
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }

    ModelInstance modelInstance = {};
    modelInstance.index = (uint)(modelIter - scene.models.begin());

    if (modelIter->joints.size() > 0) {
        D3D12MA::ALLOCATION_DESC allocDesc = {.HeapType = D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = sizeof(struct Joint) * modelIter->joints.size(), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        assert(d3d.allocator->CreateResource(&allocDesc, &resourceDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &modelInstance.animationState.skinJointsBuffer, {}, nullptr) == S_OK);
        Joint* skinJointsBufferPtr = nullptr;
        assert(modelInstance.animationState.skinJointsBuffer->GetResource()->Map(0, nullptr, (void**)&skinJointsBufferPtr) == S_OK);
        for (uint jointIndex = 0; jointIndex < modelIter->joints.size(); jointIndex++) {
            ModelJoint& modelJoint = modelIter->joints[jointIndex];
            skinJointsBufferPtr[jointIndex] = {XMMatrixTranspose(modelJoint.node->globalTransform), XMMatrixTranspose(modelJoint.inverseBindMat)};
        }
        modelInstance.animationState.skinJointsBuffer->GetResource()->Unmap(0, nullptr);

        for (ModelMesh& mesh : modelIter->meshes) {
            std::vector<D3D12MA::Allocation*>& meshPrimitivesVerticesBuffers = modelInstance.animationState.meshPrimitivesVerticesBuffers.emplace_back();
            for (ModelPrimitive& primitive : mesh.primitives) {
                D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_UPLOAD};
                D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = primitive.verticesBuffer->GetSize(), .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
                D3D12MA::Allocation* verticesBuffer;
                assert(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, &verticesBuffer, {}, nullptr) == S_OK);
                meshPrimitivesVerticesBuffers.push_back(verticesBuffer);
            }
            {
                D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC blasBufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
                D3D12MA::Allocation* blasBuffer;
                D3D12MA::Allocation* blasScratchBuffer;
                blasBufferDesc.Width = mesh.blas->GetSize();
                assert(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &blasBuffer, {}, nullptr) == S_OK);
                blasBufferDesc.Width = mesh.blasScratch->GetSize();
                assert(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &blasScratchBuffer, {}, nullptr) == S_OK);
                modelInstance.animationState.meshBlases.push_back(blasBuffer);
                modelInstance.animationState.meshBlasScratches.push_back(blasScratchBuffer);
            }
        }
    }

    return modelInstance;
}

ModelInstance sceneLoadModel(const std::filesystem::path& filePath) {
    if (filePath.extension() == ".gltf") {
        return sceneLoadModelGLTF(filePath);
    } else {
        assert(false);
        return {};
    }
}

void sceneLoad(const std::filesystem::path& path) {
    std::string yamlStr = fileReadStr(path);
    ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
    ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
    scene.filePath = path;
    {
        ryml::ConstNodeRef cameraYaml = yamlRoot["editorCamera"];
        scene.editor.camera.position << cameraYaml["position"];
        scene.editor.camera.pitchYaw << cameraYaml["pitchYaw"];
        scene.editor.camera.updateLookAt();
        cameraYaml["fovVertical"] >> scene.editor.camera.fovVertical;
        cameraYaml["sensitivity"] >> scene.editor.camera.sensitivity;
        cameraYaml["rotationSensitivity"] >> scene.editor.camera.rotationSensitivity;
        cameraYaml["controllerSensitivity"] >> scene.editor.camera.controllerSensitivity;
    }
    {
        ryml::ConstNodeRef playerYaml = yamlRoot["player"];
        std::string file;
        playerYaml["file"] >> file;
        scene.player.model = sceneLoadModel(file);
        scene.player.transform << playerYaml;
        scene.player.velocity << playerYaml["velocity"];
        scene.player.acceleration << playerYaml["acceleration"];
        scene.player.camera.lookAtOffset << playerYaml["cameraLookAtOffset"];
        scene.player.camera.rotation << playerYaml["cameraRotation"];
        playerYaml["cameraDistance"] >> scene.player.camera.distance;
        playerYaml["cameraSensitivity"] >> scene.player.camera.sensitivity;
        playerYaml["cameraControllerSensitivity"] >> scene.player.camera.controllerSensitivity;
        scene.player.camera.lookAt = scene.player.transform.translate + scene.player.camera.lookAtOffset;
    }
    ryml::ConstNodeRef staticObjectsYaml = yamlRoot["staticObjects"];
    for (ryml::ConstNodeRef const& staticObjectYaml : staticObjectsYaml) {
        StaticObject& obj = scene.staticObjects.emplace_back();
        staticObjectYaml["name"] >> obj.name;
        std::string file;
        staticObjectYaml["file"] >> file;
        obj.model = sceneLoadModel(file);
        obj.transform << staticObjectYaml;
    }
    ryml::ConstNodeRef dynamicObjectsYaml = yamlRoot["dynamicObjects"];
    for (ryml::ConstNodeRef const& dynamicObjectYaml : dynamicObjectsYaml) {
        DynamicObject& obj = scene.dynamicObjects.emplace_back();
        dynamicObjectYaml["name"] >> obj.name;
        std::string file;
        dynamicObjectYaml["file"] >> file;
        obj.model = sceneLoadModel(file);
        obj.transform << dynamicObjectYaml;
    }
    {
        ryml::ConstNodeRef skyboxYaml = yamlRoot["skybox"];
        std::string file;
        skyboxYaml["file"] >> file;
        scene.skybox.hdriTextureFilePath = file;
        ScratchImage scratchImage;
        assert(LoadFromDDSFile((assetsDir / scene.skybox.hdriTextureFilePath).c_str(), DDS_FLAGS_NONE, nullptr, scratchImage) == S_OK);
        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint)scratchImage.GetMetadata().width, .Height = (uint)scratchImage.GetMetadata().height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = scratchImage.GetMetadata().format, .SampleDesc = {.Count = 1}};
        assert(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &scene.skybox.hdriTexture, {}, nullptr) == S_OK);
        assert(scene.skybox.hdriTexture->GetResource()->SetName(L"skyboxHDRITexture") == S_OK);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT copyableFootprint;
        uint64 totalSize;
        d3d.device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, &copyableFootprint, nullptr, nullptr, &totalSize);
        assert(copyableFootprint.Footprint.RowPitch == scratchImage.GetImages()->rowPitch);
        assert(scratchImage.GetImages()->slicePitch == totalSize);
        assert(totalSize < d3d.stagingBuffer->GetSize());
        memcpy(d3d.stagingBufferPtr, scratchImage.GetImages()->pixels, totalSize);

        d3dTransferQueueStartRecording();
        D3D12_TEXTURE_COPY_LOCATION copyDst = {.pResource = scene.skybox.hdriTexture->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION copySrc = {.pResource = d3d.stagingBuffer->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = {.Footprint = copyableFootprint.Footprint}};
        d3d.transferCmdList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);
        D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = copyDst.pResource, .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
        d3d.transferCmdList->ResourceBarrier(1, &barrier);
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }
}

void sceneSave() {
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;

    ryml::NodeRef cameraYaml = yamlRoot["editorCamera"];
    cameraYaml |= ryml::MAP;
    scene.editor.camera.position >> cameraYaml["position"];
    scene.editor.camera.pitchYaw >> cameraYaml["pitchYaw"];
    cameraYaml["fovVertical"] << scene.editor.camera.fovVertical;
    cameraYaml["sensitivity"] << scene.editor.camera.sensitivity;
    cameraYaml["rotationSensitivity"] << scene.editor.camera.rotationSensitivity;
    cameraYaml["controllerSensitivity"] << scene.editor.camera.controllerSensitivity;

    ryml::NodeRef playerYaml = yamlRoot["player"];
    playerYaml |= ryml::MAP;
    playerYaml["file"] << scene.models[scene.player.model.index].filePath.string();
    scene.player.transform >> playerYaml;
    scene.player.velocity >> playerYaml["velocity"];
    scene.player.acceleration >> playerYaml["acceleration"];
    scene.player.camera.lookAtOffset >> playerYaml["cameraLookAtOffset"];
    scene.player.camera.rotation >> playerYaml["cameraRotation"];
    playerYaml["cameraDistance"] << scene.player.camera.distance;
    playerYaml["cameraSensitivity"] << scene.player.camera.sensitivity;
    playerYaml["cameraControllerSensitivity"] << scene.player.camera.controllerSensitivity;

    ryml::NodeRef skyboxYaml = yamlRoot["skybox"];
    skyboxYaml |= ryml::MAP;
    skyboxYaml["file"] << scene.skybox.hdriTextureFilePath.string();

    ryml::NodeRef staticObjectsYaml = yamlRoot["staticObjects"];
    staticObjectsYaml |= ryml::SEQ;
    for (StaticObject& staticObject : scene.staticObjects) {
        ryml::NodeRef staticObjectYaml = staticObjectsYaml.append_child();
        staticObjectYaml |= ryml::MAP;
        staticObjectYaml["name"] << staticObject.name;
        staticObjectYaml["file"] << scene.models[staticObject.model.index].filePath.string();
        staticObject.transform >> staticObjectYaml;
    }

    ryml::NodeRef dynamicObjectsYaml = yamlRoot["dynamicObjects"];
    dynamicObjectsYaml |= ryml::SEQ;
    for (DynamicObject& dynamicObject : scene.dynamicObjects) {
        ryml::NodeRef dynamicObjectYaml = dynamicObjectsYaml.append_child();
        dynamicObjectYaml |= ryml::MAP;
        dynamicObjectYaml["name"] << dynamicObject.name;
        dynamicObjectYaml["file"] << scene.models[dynamicObject.model.index].filePath.string();
        dynamicObject.transform >> dynamicObjectYaml;
    }

    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    assert(fileWriteStr(scene.filePath, yamlStr));
}

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
static bool playing = false;
static LARGE_INTEGER perfFrequency = {};
static LARGE_INTEGER perfCounters[2] = {};
static double frameTime = 0;
static int mouseDeltaRaw[2] = {};
static float mouseWheel = 0;
static uint mouseSelectX = UINT_MAX;
static uint mouseSelectY = UINT_MAX;

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
        windowUpdateSizes();
        if (d3d.swapChain && settings.renderW > 0 && settings.renderH > 0 && (prevRenderW != settings.renderW || prevRenderH != settings.renderH)) {
            d3dResizeSwapChain(settings.renderW, settings.renderH);
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
};

static Controller controller = {};

void controllerGetInputs() {
    static XINPUT_STATE prevState = {};
    XINPUT_STATE state = {};
    DWORD result = XInputGetState(0, &state);
    if (result == ERROR_SUCCESS) {
        controller.a = state.Gamepad.wButtons & XINPUT_GAMEPAD_A;
        controller.b = state.Gamepad.wButtons & XINPUT_GAMEPAD_B;
        controller.x = state.Gamepad.wButtons & XINPUT_GAMEPAD_X;
        controller.y = state.Gamepad.wButtons & XINPUT_GAMEPAD_Y;
        controller.up = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP;
        controller.down = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN;
        controller.left = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT;
        controller.right = state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT;
        controller.lb = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER;
        controller.rb = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER;
        controller.ls = state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB;
        controller.rs = state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB;
        controller.back = state.Gamepad.wButtons & XINPUT_GAMEPAD_BACK;
        controller.start = state.Gamepad.wButtons & XINPUT_GAMEPAD_START;
        controller.lt = state.Gamepad.bLeftTrigger / 255.0f;
        controller.rt = state.Gamepad.bRightTrigger / 255.0f;
        controller.lStick.x = state.Gamepad.sThumbLX / 32767.0f;
        controller.lStick.y = state.Gamepad.sThumbLY / 32767.0f;
        controller.rStick.x = state.Gamepad.sThumbRX / 32767.0f;
        controller.rStick.y = state.Gamepad.sThumbRY / 32767.0f;

        float lDistance = controller.lStick.length();
        if (lDistance > 0) {
            float lDistanceNew = std::max(0.0f, lDistance - controller.deadZone) / (1.0f - controller.deadZone);
            controller.lStick = controller.lStick / lDistance * lDistanceNew;
        }
        float rDistance = controller.rStick.length();
        if (rDistance > 0) {
            float rDistanceNew = std::max(0.0f, rDistance - controller.deadZone) / (1.0f - controller.deadZone);
            controller.rStick = controller.rStick / rDistance * rDistanceNew;
        }
    } else {
        controller = {};
    }
    prevState = state;
}

bool controllerStickMoved() {
    return controller.lStick != float2(0, 0) || controller.rStick != float2(0, 0);
}

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

void update() {
    ImGui::GetIO().DeltaTime = (float)frameTime;
    ImGui::GetIO().DisplaySize = ImVec2((float)settings.renderW, (float)settings.renderH);
    ImGui::NewFrame();
    ImGuizmo::SetRect(0, 0, (float)settings.renderW, (float)settings.renderH);
    ImGuizmo::BeginFrame();

    static ImVec2 mousePosPrev = ImGui::GetMousePos();
    ImVec2 mousePos = ImGui::GetMousePos();
    ImVec2 mouseDelta = mousePos - mousePosPrev;
    mousePosPrev = mousePos;

    if (d3d.graphicsQueueFenceCounter > 0) {
        ReadBackBuffer* readBackBuffer = (ReadBackBuffer*)d3d.readBackBufferPtr;
        uint mouseSelectInstanceIndex = readBackBuffer->mouseSelectInstanceIndex;
        if (mouseSelectInstanceIndex < scene.tlasInstancesInfos.size()) {
            TLASInstanceInfo& info = scene.tlasInstancesInfos[mouseSelectInstanceIndex];
            scene.editor.selectedObjectType = info.objectType;
            scene.editor.selectedObjectIndex = info.objectIndex;
        } else {
            if (mouseSelectX != UINT_MAX && mouseSelectY != UINT_MAX) {
                scene.editor.selectedObjectType = SceneObjectTypeNone;
            }
        }
        // scene.player.transform.translate = readBackBuffer->playerPosition;
        // scene.player.velocity = readBackBuffer->playerVelocity;
        // scene.player.acceleration = readBackBuffer->playerAcceleration;
    }
    {
        Model& model = scene.models[scene.player.model.index];
        ModelAnimation& animation = model.animations[scene.player.model.animationState.index];
        scene.player.model.animationState.time += frameTime;
        if (scene.player.model.animationState.time > animation.timeLength) {
            scene.player.model.animationState.time -= animation.timeLength;
        }
    }

    static std::vector<std::string> logs = {};
    static bool addObjectPopup = false;
    ImVec2 mainMenuBarPos;
    ImVec2 mainMenuBarSize;
    if (ImGui::BeginMainMenuBar()) {
        mainMenuBarPos = ImGui::GetWindowPos();
        mainMenuBarSize = ImGui::GetWindowSize();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::BeginMenu("New")) {
                if (ImGui::MenuItem("Scene")) {
                    // addScene("New Scene", "");
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Open")) {
                if (ImGui::MenuItem("Scene")) {
                    char filePath[256] = {};
                    OPENFILENAMEA openFileName = {.lStructSize = sizeof(openFileName), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                    if (GetOpenFileNameA(&openFileName)) {
                        // addScene("New Scene", filePath);
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
                applySettings(settingsHDR);
            } else if (ImGui::MenuItem("Windowed")) {
                settings.windowMode = WindowModeWindowed;
                applySettings(settingWindowMode);
            } else if (ImGui::MenuItem("Borderless Fullscreen")) {
                settings.windowMode = WindowModeBorderless;
                applySettings(settingWindowMode);
            }
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
            ImGui::Text("Exclusive Fullscreen");
            for (DisplayMode& mode : d3d.displayModes) {
                std::string text = std::format("{}x{}", mode.resolution.width, mode.resolution.height);
                if (ImGui::BeginMenu(text.c_str())) {
                    for (DXGI_RATIONAL& refreshRate : mode.refreshRates) {
                        text = std::format("{:.2f}hz", (float)refreshRate.Numerator / (float)refreshRate.Denominator);
                        if (ImGui::MenuItem(text.c_str())) {
                            settings.windowMode = WindowModeFullscreen;
                            settings.windowW = mode.resolution.width;
                            settings.windowH = mode.resolution.height;
                            settings.refreshRate = refreshRate;
                            applySettings(settingWindowMode);
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Editor")) {
            if (ImGui::BeginMenu("Camera")) {
                ImGui::SliderFloat("Sensitivity", &scene.editor.camera.sensitivity, 0.0f, 100.0f);
                ImGui::SliderFloat("RotationSensitivity", &scene.editor.camera.rotationSensitivity, 0.1f, 10.0f);
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
                playing = !playing;
                hideCursor(playing);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    ImVec2 objectWindowPos = ImVec2(settings.renderW * 0.8f, mainMenuBarSize.y);
    ImVec2 objectWindowSize = ImVec2(settings.renderW * 0.2f, settings.renderH * 0.3f);
    ImGui::SetNextWindowPos(objectWindowPos);
    ImGui::SetNextWindowSize(objectWindowSize);
    if (ImGui::Begin("Objects")) {
        if (ImGui::Selectable("Player", scene.editor.selectedObjectType == SceneObjectTypePlayer)) {
            scene.editor.selectedObjectType = SceneObjectTypePlayer;
        }
        if (scene.editor.selectedObjectType == SceneObjectTypePlayer && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("player edit");
        }
        if (scene.editor.selectedObjectType == SceneObjectTypePlayer && ImGui::BeginPopup("player edit")) {
            if (ImGui::Selectable("focus")) scene.editor.camera.focus(scene.player.transform.translate, 1);
            ImGui::EndPopup();
        }
        if (ImGui::TreeNode("Static Objects")) {
            for (uint index = 0; StaticObject & object : scene.staticObjects) {
                if (ImGui::Selectable(object.name.c_str(), scene.editor.selectedObjectType == SceneObjectTypeStaticObject && scene.editor.selectedObjectIndex == index)) {
                    scene.editor.selectedObjectType = SceneObjectTypeStaticObject;
                    scene.editor.selectedObjectIndex = index;
                }
                if (scene.editor.selectedObjectType == SceneObjectTypeStaticObject && scene.editor.selectedObjectIndex == index &&
                    ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImGui::OpenPopup("static object edit");
                }
                if (scene.editor.selectedObjectType == SceneObjectTypeStaticObject && scene.editor.selectedObjectIndex == index && ImGui::BeginPopup("static object edit")) {
                    if (ImGui::Selectable("focus")) scene.editor.camera.focus(object.transform.translate, 1);
                    if (ImGui::Selectable("delete")) {
                        object.toBeDeleted = true;
                        scene.editor.selectedObjectType = SceneObjectTypeNone;
                    }
                    ImGui::EndPopup();
                }
                index += 1;
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Dyanmic Objects")) {
            for (uint index = 0; DynamicObject & object : scene.dynamicObjects) {
                if (ImGui::Selectable(object.name.c_str(), scene.editor.selectedObjectType == SceneObjectTypeDynamicObject && scene.editor.selectedObjectIndex == index)) {
                    scene.editor.selectedObjectType = SceneObjectTypeDynamicObject;
                    scene.editor.selectedObjectIndex = index;
                }
                if (scene.editor.selectedObjectType == SceneObjectTypeDynamicObject && scene.editor.selectedObjectIndex == index &&
                    ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImGui::OpenPopup("dynamic object edit");
                }
                if (scene.editor.selectedObjectType == SceneObjectTypeDynamicObject && scene.editor.selectedObjectIndex == index && ImGui::BeginPopup("dynamic object edit")) {
                    if (ImGui::Selectable("focus")) scene.editor.camera.focus(object.transform.translate, 1);
                    if (ImGui::Selectable("delete")) {
                        object.toBeDeleted = true;
                        scene.editor.selectedObjectType = SceneObjectTypeNone;
                    }
                    ImGui::EndPopup();
                }
                index += 1;
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
    auto propertyTransform = [](Transform* transform) {
        if (ImGui::TreeNode("Transform")) {
            ImGui::InputFloat3("S", &transform->scale.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##scale")) { transform->scale = float3(1, 1, 1); }
            ImGui::InputFloat4("R", &transform->rotate.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##rotate")) { transform->rotate = float4(0, 0, 0, 1); }
            ImGui::InputFloat3("T", &transform->translate.x);
            ImGui::SameLine();
            if (ImGui::Button("reset##translate")) { transform->translate = float3(0, 0, 0); }
            ImGui::TreePop();
        }
    };
    auto propertyModel = [](Model* model) {
        if (ImGui::TreeNode("Model")) {
            ImGui::Text(std::format("File: {}", model->filePath.string()).c_str());
            if (ImGui::TreeNode("Animations")) {
                for (uint animationIndex = 0; animationIndex < model->animations.size(); animationIndex++) {
                    ModelAnimation& animation = model->animations[animationIndex];
                    ImGui::Text(std::format("#{}: {}", animationIndex, animation.name).c_str());
                    ImGui::SameLine(ImGui::GetWindowWidth() * 0.8f);
                    ImGui::PushID(animationIndex);
                    if (ImGui::Button("play")) {
                        scene.player.model.animationState.index = animationIndex;
                        scene.player.model.animationState.time = 0;
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
        if (scene.editor.selectedObjectType == SceneObjectTypePlayer) {
            ImGui::Text("Player");
            propertyTransform(&scene.player.transform);
            if (ImGui::TreeNode("Movement")) {
                ImGui::InputFloat3("Velocity", &scene.player.velocity.x);
                ImGui::InputFloat3("Acceleration", &scene.player.acceleration.x);
                ImGui::TreePop();
            }
            propertyModel(&scene.models[scene.player.model.index]);
        } else if (scene.editor.selectedObjectType == SceneObjectTypeStaticObject) {
            StaticObject& staticObject = scene.staticObjects[scene.editor.selectedObjectIndex];
            ImGui::Text("Static Object #%d", scene.editor.selectedObjectIndex);
            ImGui::Text("Name \"%s\"", staticObject.name.c_str());
            propertyTransform(&staticObject.transform);
            propertyModel(&scene.models[staticObject.model.index]);
        } else if (scene.editor.selectedObjectType == SceneObjectTypeDynamicObject) {
            DynamicObject& dynamicObject = scene.dynamicObjects[scene.editor.selectedObjectIndex];
            ImGui::Text("Dynamic Object #%d", scene.editor.selectedObjectIndex);
            ImGui::Text("Name \"%s\"", dynamicObject.name.c_str());
            propertyTransform(&dynamicObject.transform);
            propertyModel(&scene.models[dynamicObject.model.index]);
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
            if (ImGui::Button("browse")) {
                OPENFILENAMEA openfileName = {.lStructSize = sizeof(OPENFILENAMEA), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                GetOpenFileNameA(&openfileName);
            }
        }
        if (ImGui::Button("Add")) {
            if (objectName[0] == '\0') {
                logs.push_back("error: object name is empty");
            } else {
                bool duplicatedName = false;
                for (StaticObject& object : scene.staticObjects) {
                    if (object.name == objectName) {
                        duplicatedName = true;
                    }
                }
                for (DynamicObject& object : scene.dynamicObjects) {
                    if (object.name == objectName) {
                        duplicatedName = true;
                    }
                }
                if (duplicatedName) {
                    logs.push_back(std::format("error: object name \"{}\" already exists", objectName));
                } else {
                    std::filesystem::path path = std::filesystem::relative(filePath, assetsDir);
                    ModelInstance model = sceneLoadModel(path);
                    if (model.index >= scene.models.size()) {
                        logs.push_back(std::format("error: failed to load model file: \"{}\"", path.string()));
                    } else {
                        StaticObject staticObject = {.name = objectName, .model = model};
                        scene.staticObjects.push_back(std::move(staticObject));
                    }
                }
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_P, false) &&
        ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
        playing = !playing;
        hideCursor(playing);
    }
    if (playing) {
        float xRotate = -mouseDeltaRaw[1] / 500.0f * scene.player.camera.sensitivity;
        float yRotate = mouseDeltaRaw[0] / 500.0f * scene.player.camera.sensitivity;
        xRotate += controller.rStick.y * (float)frameTime * scene.player.camera.controllerSensitivity;
        yRotate += controller.rStick.x * (float)frameTime * scene.player.camera.controllerSensitivity;
        float zoom = -mouseWheel * (float)frameTime * scene.player.camera.sensitivity;
        scene.player.camera.distance += zoom;
        scene.player.camera.rotation.x += xRotate;
        scene.player.camera.rotation.y += yRotate;
        scene.player.camera.lookAt = scene.player.transform.translate + scene.player.camera.lookAtOffset;
    } else {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
            mouseSelectX = (uint)mousePos.x;
            mouseSelectY = (uint)mousePos.y;
        } else {
            mouseSelectX = UINT_MAX;
            mouseSelectY = UINT_MAX;
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::GetIO().WantCaptureMouse) {
            scene.editor.cameraMoving = true;
            hideCursor(true);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            scene.editor.cameraMoving = false;
            hideCursor(false);
        }
        if (scene.editor.cameraMoving || controllerStickMoved()) {
            float pitch = (mouseDeltaRaw[1] / 500.0f * scene.editor.camera.rotationSensitivity) - (controller.rStick.y * (float)frameTime * scene.editor.camera.controllerSensitivity);
            float yaw = (mouseDeltaRaw[0] / 500.0f * scene.editor.camera.rotationSensitivity) + (controller.rStick.x * (float)frameTime * scene.editor.camera.controllerSensitivity);
            scene.editor.camera.sensitivity = std::clamp(scene.editor.camera.sensitivity + ImGui::GetIO().MouseWheel, 0.0f, 100.0f);
            float distance = (float)frameTime / 5.0f * scene.editor.camera.sensitivity;
            float3 translate = {0, 0, 0};
            if (ImGui::IsKeyDown(ImGuiKey_W)) translate.z = distance;
            if (ImGui::IsKeyDown(ImGuiKey_S)) translate.z = -distance;
            if (ImGui::IsKeyDown(ImGuiKey_A)) translate.x = distance;
            if (ImGui::IsKeyDown(ImGuiKey_D)) translate.x = -distance;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) translate.y = distance;
            if (ImGui::IsKeyDown(ImGuiKey_E)) translate.y = -distance;
            // scene.editor.camera.position += scene.editor.camera.dir.cross(float3(0, 1, 0)) * distance * -controller.lStick.x;
            // scene.editor.camera.position += scene.editor.camera.dir * distance * controller.lStick.y;
            scene.editor.camera.rotate(pitch, yaw);
            scene.editor.camera.translate(translate);
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
        const XMMATRIX lookAtMat = XMMatrixLookAtLH(scene.editor.camera.position.toXMVector(), scene.editor.camera.lookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
        const XMMATRIX perspectiveMat = XMMatrixPerspectiveFovLH(radian(scene.editor.camera.fovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f);
        auto transformGizmo = [&](Transform* transform) {
            XMMATRIX transformMat = transform->toMat();
            if (ImGui::IsKeyPressed(ImGuiKey_T)) gizmoOperation = ImGuizmo::TRANSLATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_R)) gizmoOperation = ImGuizmo::ROTATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_S)) gizmoOperation = ImGuizmo::SCALE;
            if (ImGuizmo::Manipulate((const float*)&lookAtMat, (const float*)&perspectiveMat, gizmoOperation, gizmoMode, (float*)&transformMat)) {
                XMVECTOR scale, rotate, translate;
                if (XMMatrixDecompose(&scale, &rotate, &translate, transformMat)) {
                    transform->scale = scale, transform->rotate = rotate, transform->translate = translate;
                }
            }
        };
        if (scene.editor.selectedObjectType == SceneObjectTypePlayer) {
            transformGizmo(&scene.player.transform);
        } else if (scene.editor.selectedObjectType == SceneObjectTypeStaticObject && scene.editor.selectedObjectIndex < scene.staticObjects.size()) {
            StaticObject& staticObject = scene.staticObjects[scene.editor.selectedObjectIndex];
            transformGizmo(&staticObject.transform);
        } else if (scene.editor.selectedObjectType == SceneObjectTypeDynamicObject && scene.editor.selectedObjectIndex < scene.dynamicObjects.size()) {
            DynamicObject& dynamicObject = scene.dynamicObjects[scene.editor.selectedObjectIndex];
            transformGizmo(&dynamicObject.transform);
        }
        // static const XMMATRIX gridMat = XMMatrixIdentity();
        // ImGuizmo::DrawGrid((const float*)&lookAtMat, (const float*)&perspectiveMat, (const float*)&gridMat, 10);
    }

    vectorDeleteElements(scene.staticObjects);
    vectorDeleteElements(scene.dynamicObjects);

    ImGui::Render();
}

void updateAnimatedModel(ModelInstance& modelInstance) {
    Model& model = scene.models[modelInstance.index];
    if (!modelInstance.animationState.skinJointsBuffer || model.animations.size() == 0) return;
    if (modelInstance.animationState.index >= model.animations.size()) return;
    ModelAnimation& animation = model.animations[modelInstance.animationState.index];

    std::vector<Transform> nodeLocalTransforms(model.nodes.size());
    std::vector<XMMATRIX> nodeLocalTransformMats(model.nodes.size());
    std::vector<XMMATRIX> nodeGlobalTransformMats(model.nodes.size());
    std::vector<XMMATRIX> jointTransformMats(model.joints.size());

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
                nodeLocalTransforms[nodeIndex].translate = lerp(frame0.xyz(), frame1.xyz(), progress);
            } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                nodeLocalTransforms[nodeIndex].translate = progress < 1.0f ? frame0.xyz() : frame1.xyz();
            }
        } else if (channel.type == AnimationChannelTypeRotate) {
            if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                nodeLocalTransforms[nodeIndex].rotate = slerp(frame0, frame1, progress);
            } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                nodeLocalTransforms[nodeIndex].rotate = progress < 1.0f ? frame0 : frame1;
            }
        } else if (channel.type == AnimationChannelTypeScale) {
            if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                nodeLocalTransforms[nodeIndex].scale = lerp(frame0.xyz(), frame1.xyz(), progress);
            } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                nodeLocalTransforms[nodeIndex].scale = progress < 1.0f ? frame0.xyz() : frame1.xyz();
            }
        }
    }
    for (uint nodeIndex = 0; nodeIndex < model.nodes.size(); nodeIndex++) {
        nodeLocalTransformMats[nodeIndex] = nodeLocalTransforms[nodeIndex].toMat();
        nodeGlobalTransformMats[nodeIndex] = XMMatrixIdentity();
    }
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
        jointTransformMats[jointIndex] = XMMatrixMultiply(model.joints[jointIndex].inverseBindMat, nodeGlobalTransformMats[nodeIndex]);
    }

    static std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.resize(model.meshes.size());
    for (uint meshIndex = 0; meshIndex < model.meshes.size(); meshIndex++) {
        ModelMesh& mesh = model.meshes[meshIndex];
        std::vector<D3D12MA::Allocation*>& verticesBuffers = modelInstance.animationState.meshPrimitivesVerticesBuffers[meshIndex];
        for (uint primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); primitiveIndex++) {
            ModelPrimitive& primitive = mesh.primitives[primitiveIndex];
            D3D12MA::Allocation* verticesBuffer = verticesBuffers[primitiveIndex];
            Vertex* verticesBufferPtr = nullptr;
            assert(verticesBuffer->GetResource()->Map(0, nullptr, (void**)&verticesBufferPtr) == S_OK);
            for (uint vertexIndex = 0; vertexIndex < primitive.vertices.size(); vertexIndex++) {
                Vertex vertex = primitive.vertices[vertexIndex];
                XMMATRIX skinMat0 = jointTransformMats[vertex.joints.x] * vertex.jointWeights.x;
                XMMATRIX skinMat1 = jointTransformMats[vertex.joints.y] * vertex.jointWeights.y;
                XMMATRIX skinMat2 = jointTransformMats[vertex.joints.z] * vertex.jointWeights.z;
                XMMATRIX skinMat3 = jointTransformMats[vertex.joints.w] * vertex.jointWeights.w;
                XMMATRIX skinMat = skinMat0 + skinMat1 + skinMat2 + skinMat3;
                vertex.position = XMVector3Transform(vertex.position.toXMVector(), skinMat);
                XMFLOAT3X3 normalMat;
                XMStoreFloat3x3(&normalMat, skinMat);
                vertex.normal = XMVector3Transform(vertex.normal.toXMVector(), XMLoadFloat3x3(&normalMat));
                verticesBufferPtr[vertexIndex] = vertex;
            }
            verticesBuffer->GetResource()->Unmap(0, nullptr);
        }
        std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
        geometryDescs.resize(verticesBuffers.size());
        for (uint bufferIndex = 0; bufferIndex < verticesBuffers.size(); bufferIndex++) {
            D3D12MA::Allocation* verticesBuffer = verticesBuffers[bufferIndex];
            geometryDescs[bufferIndex] = {
                .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                .Triangles = {
                    .Transform3x4 = 0,
                    .IndexFormat = DXGI_FORMAT_R32_UINT,
                    .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
                    .IndexCount = (uint)(mesh.primitives[bufferIndex].indices.size()),
                    .VertexCount = (uint)(mesh.primitives[bufferIndex].vertices.size()),
                    .IndexBuffer = mesh.primitives[bufferIndex].indicesBuffer->GetResource()->GetGPUVirtualAddress(),
                    .VertexBuffer = {.StartAddress = verticesBuffer->GetResource()->GetGPUVirtualAddress(), .StrideInBytes = sizeof(Vertex)},
                },
            };
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
        barriers[meshIndex] = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = modelInstance.animationState.meshBlases[meshIndex]->GetResource()}};
    }
    d3d.graphicsCmdList->ResourceBarrier((uint)barriers.size(), barriers.data());
}

void addTlasInstance(ModelInstance& modelInstance, const XMMATRIX& objectTransform, SceneObjectType objectType, uint objectIndex, uint selected) {
    Model& model = scene.models[modelInstance.index];
    TLASInstanceInfo tlasInstanceInfo = {.objectType = objectType, .objectIndex = objectIndex, .selected = selected, .skinJointsDescriptor = UINT32_MAX};
    if (!model.joints.empty()) {
        assertDebug(modelInstance.animationState.skinJointsBuffer);
        // tlasInstanceInfo.skinJointsDescriptor = d3d.cbvSrvUavDescriptorCount;
        // D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        // d3dAppendSrvDescriptor(&desc, modelInstance.animationState.skinJointsBuffer->GetResource());
        for (ModelNode& node : model.nodes) {
            if (node.mesh) {
                int64 meshIndex = node.mesh - &model.meshes[0];
                D3D12MA::Allocation* meshBlas = modelInstance.animationState.meshBlases[meshIndex];
                std::vector<D3D12MA::Allocation*> primitiveVerticesBuffers = modelInstance.animationState.meshPrimitivesVerticesBuffers[meshIndex];
                D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {.InstanceID = d3d.cbvSrvUavDescriptorCount, .InstanceMask = 0xff, .AccelerationStructure = meshBlas->GetResource()->GetGPUVirtualAddress()};
                XMMATRIX transform = node.globalTransform;
                transform = XMMatrixMultiply(transform, XMMatrixScaling(1, 1, -1)); // convert RH to LH
                transform = XMMatrixMultiply(transform, objectTransform);
                transform = XMMatrixTranspose(transform);
                memcpy(instanceDesc.Transform, &transform, sizeof(instanceDesc.Transform));
                scene.tlasInstancesBuildInfos.push_back(instanceDesc);
                scene.tlasInstancesInfos.push_back(tlasInstanceInfo);
                for (uint primitiveIndex = 0; primitiveIndex < node.mesh->primitives.size(); primitiveIndex++) {
                    ModelPrimitive& primitive = node.mesh->primitives[primitiveIndex];
                    D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)primitive.vertices.size(), .StructureByteStride = sizeof(struct Vertex)}};
                    D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)primitive.indices.size(), .StructureByteStride = sizeof(uint)}};
                    d3dAppendSrvDescriptor(&vertexBufferDesc, primitiveVerticesBuffers[primitiveIndex]->GetResource());
                    d3dAppendSrvDescriptor(&indexBufferDesc, primitive.indicesBuffer->GetResource());
                }
            }
        }
    } else {
        for (ModelNode& node : model.nodes) {
            if (node.mesh) {
                D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {.InstanceID = d3d.cbvSrvUavDescriptorCount, .InstanceMask = 0xff, .AccelerationStructure = node.mesh->blas->GetResource()->GetGPUVirtualAddress()};
                XMMATRIX transform = node.globalTransform;
                transform = XMMatrixMultiply(transform, XMMatrixScaling(1, 1, -1)); // convert RH to LH
                transform = XMMatrixMultiply(transform, objectTransform);
                transform = XMMatrixTranspose(transform);
                memcpy(instanceDesc.Transform, &transform, sizeof(instanceDesc.Transform));
                scene.tlasInstancesBuildInfos.push_back(instanceDesc);
                scene.tlasInstancesInfos.push_back(tlasInstanceInfo);
                for (ModelPrimitive& primitive : node.mesh->primitives) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)primitive.vertices.size(), .StructureByteStride = sizeof(struct Vertex)}};
                    D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)primitive.indices.size(), .StructureByteStride = sizeof(uint)}};
                    d3dAppendSrvDescriptor(&vertexBufferDesc, primitive.verticesBuffer->GetResource());
                    d3dAppendSrvDescriptor(&indexBufferDesc, primitive.indicesBuffer->GetResource());
                }
            }
        }
    }
}

void render() {
    d3dGraphicsQueueStartRecording();

    D3D12_RESOURCE_BARRIER renderTextureTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
    d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);

    d3d.cbvSrvUavDescriptorCount = 0;
    d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap);

    uint constantBufferOffset = 0;
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC renderTextureSRVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = 1}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC renderTextureUAVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3D12_CONSTANT_BUFFER_VIEW_DESC renderInfoCBVDesc = {.BufferLocation = d3d.constantBuffer->GetResource()->GetGPUVirtualAddress(), .SizeInBytes = align((uint)sizeof(struct RenderInfo), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT)};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasViewDesc = {.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .RaytracingAccelerationStructure = {.Location = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress()}};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasInstancesInfosDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.tlasInstancesInfosBuffer->GetSize() / sizeof(struct TLASInstanceInfo)), .StructureByteStride = sizeof(struct TLASInstanceInfo)}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC readBackBufferDesc = {.ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct ReadBackBuffer)}};
        D3D12_SHADER_RESOURCE_VIEW_DESC collisionQueriesDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQuery)}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC collisionQueryResultsDesc = {.ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQueryResult)}};

        D3DDescriptor renderTextureSRVDescriptor = d3dAppendSrvDescriptor(&renderTextureSRVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderTextureUAVDescriptor = d3dAppendUavDescriptor(&renderTextureUAVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderInfoDescriptor = d3dAppendCbvDescriptor(&renderInfoCBVDesc);
        D3DDescriptor tlasDescriptor = d3dAppendSrvDescriptor(&tlasViewDesc, nullptr);
        D3DDescriptor tlasInstancesInfosDescriptor = d3dAppendSrvDescriptor(&tlasInstancesInfosDesc, d3d.tlasInstancesInfosBuffer->GetResource());
        D3DDescriptor skyboxTextureDescriptor = d3dAppendSrvDescriptor(nullptr, scene.skybox.hdriTexture->GetResource());
        D3DDescriptor readBackBufferDescriptor = d3dAppendUavDescriptor(&readBackBufferDesc, d3d.readBackUavBuffer->GetResource());
        D3DDescriptor imguiTextureDescriptor = d3dAppendSrvDescriptor(nullptr, d3d.imguiTexture->GetResource());
        D3DDescriptor collisionQueriesDescriptor = d3dAppendSrvDescriptor(&collisionQueriesDesc, d3d.collisionQueriesBuffer->GetResource());
        D3DDescriptor collisionQueryResultsDescriptor = d3dAppendUavDescriptor(&collisionQueryResultsDesc, d3d.collisionQueryResultsUavBuffer->GetResource());
    }
    {
        float3 cameraPosition;
        float3 cameraLookAt;
        float cameraFovVertical;
        if (playing) {
            cameraPosition = scene.player.camera.position;
            cameraLookAt = scene.player.camera.lookAt;
            cameraFovVertical = 50;
        } else {
            cameraPosition = scene.editor.camera.position;
            cameraLookAt = scene.editor.camera.lookAt;
            cameraFovVertical = scene.editor.camera.fovVertical;
        }
        RenderInfo renderInfo = {
            .cameraViewMat = XMMatrixTranspose(XMMatrixInverse(nullptr, XMMatrixLookAtLH(cameraPosition.toXMVector(), cameraLookAt.toXMVector(), XMVectorSet(0, 1, 0, 0)))),
            .cameraProjMat = XMMatrixPerspectiveFovLH(radian(cameraFovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f),
            .resolution = {settings.renderW, settings.renderH},
            .mouseSelectPosition = {mouseSelectX, mouseSelectY},
            .hdr = settings.hdr,
            .frameTime = (float)frameTime,
            .playerPosition = scene.player.transform.translate,
            .playerVelocity = scene.player.velocity,
            .playerAcceleration = scene.player.acceleration,
        };
        assertDebug(constantBufferOffset == 0);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, &renderInfo, sizeof(renderInfo));
        constantBufferOffset += sizeof(renderInfo);
    }
    {
        updateAnimatedModel(scene.player.model);
        for (StaticObject& obj : scene.staticObjects) updateAnimatedModel(obj.model);
        for (DynamicObject& obj : scene.dynamicObjects) updateAnimatedModel(obj.model);

        scene.tlasInstancesBuildInfos.resize(0);
        scene.tlasInstancesInfos.resize(0);
        addTlasInstance(scene.player.model, scene.player.transform.toMat(), SceneObjectTypePlayer, 0, scene.editor.selectedObjectType == SceneObjectTypePlayer);
        for (uint objIndex = 0; objIndex < scene.staticObjects.size(); objIndex++) addTlasInstance(scene.staticObjects[objIndex].model, scene.staticObjects[objIndex].transform.toMat(), SceneObjectTypeStaticObject, objIndex, scene.editor.selectedObjectType == SceneObjectTypeStaticObject && scene.editor.selectedObjectIndex == objIndex);
        for (uint objIndex = 0; objIndex < scene.dynamicObjects.size(); objIndex++) addTlasInstance(scene.dynamicObjects[objIndex].model, scene.dynamicObjects[objIndex].transform.toMat(), SceneObjectTypeDynamicObject, objIndex, scene.editor.selectedObjectType == SceneObjectTypeDynamicObject && scene.editor.selectedObjectIndex == objIndex);

        assert(vectorSizeof(scene.tlasInstancesBuildInfos) < d3d.tlasInstancesBuildInfosBuffer->GetSize());
        assert(vectorSizeof(scene.tlasInstancesInfos) < d3d.tlasInstancesInfosBuffer->GetSize());
        memcpy(d3d.tlasInstancesBuildInfosBufferPtr, scene.tlasInstancesBuildInfos.data(), vectorSizeof(scene.tlasInstancesBuildInfos));
        memcpy(d3d.tlasInstancesInfosBufferPtr, scene.tlasInstancesInfos.data(), vectorSizeof(scene.tlasInstancesInfos));

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, .NumDescs = (uint)scene.tlasInstancesBuildInfos.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .InstanceDescs = d3d.tlasInstancesBuildInfosBuffer->GetResource()->GetGPUVirtualAddress()};
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);
        assert(prebuildInfo.ResultDataMaxSizeInBytes < d3d.tlasBuffer->GetSize());
        assert(prebuildInfo.ScratchDataSizeInBytes < d3d.tlasScratchBuffer->GetSize());

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress(), .Inputs = inputs, .ScratchAccelerationStructureData = d3d.tlasScratchBuffer->GetResource()->GetGPUVirtualAddress()};
        d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
        D3D12_RESOURCE_BARRIER tlasBarrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = d3d.tlasBuffer->GetResource()}};
        d3d.graphicsCmdList->ResourceBarrier(1, &tlasBarrier);

        D3D12_DISPATCH_RAYS_DESC dispatchDesc = {.Width = settings.renderW, .Height = settings.renderH, .Depth = 1};
        constantBufferOffset = align(constantBufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.renderSceneRayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.RayGenerationShaderRecord = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.renderScenePrimaryRayMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        memcpy(d3d.constantBufferPtr + constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, d3d.renderSceneSecondaryRayMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.MissShaderTable = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset = align(constantBufferOffset + 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.renderScenePrimaryRayHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        memcpy(d3d.constantBufferPtr + constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, d3d.renderSceneSecondaryRayHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.HitGroupTable = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset += 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        assert(constantBufferOffset < d3d.constantBuffer->GetSize());

        d3d.graphicsCmdList->SetPipelineState1(d3d.renderSceneSO);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.renderSceneRootSig);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);
    }
    {
        D3D12_RESOURCE_BARRIER readBackBufferTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.readBackUavBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        D3D12_RESOURCE_BARRIER collisionQueryResultsTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.collisionQueryResultsUavBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

        D3D12_DISPATCH_RAYS_DESC dispatchDesc = {.Width = 1, .Height = 1, .Depth = 1};
        constantBufferOffset = align(constantBufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.collisionDetectionRayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.RayGenerationShaderRecord = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.collisionDetectionMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.MissShaderTable = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
        memcpy(d3d.constantBufferPtr + constantBufferOffset, d3d.collisionDetectionHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        dispatchDesc.HitGroupTable = {d3d.constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
        constantBufferOffset += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        assert(constantBufferOffset < d3d.constantBuffer->GetSize());

        d3d.graphicsCmdList->SetPipelineState1(d3d.collisionDetection);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.collisionDetectionRootSig);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

        std::swap(readBackBufferTransition.Transition.StateBefore, readBackBufferTransition.Transition.StateAfter);
        std::swap(collisionQueryResultsTransition.Transition.StateBefore, collisionQueryResultsTransition.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

        d3d.graphicsCmdList->CopyBufferRegion(d3d.readBackBuffer->GetResource(), 0, d3d.readBackUavBuffer->GetResource(), 0, sizeof(struct ReadBackBuffer));
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
            d3d.graphicsCmdList->SetPipelineState(d3d.postProcess);
            d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.postProcessRootSig);
            std::swap(renderTextureTransition.Transition.StateBefore, renderTextureTransition.Transition.StateAfter);
            d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);
            d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
        }
        {
            d3d.graphicsCmdList->SetPipelineState(d3d.imgui);
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
    d3dGraphicsQueueSubmitRecording();
    assert(d3d.swapChain->Present(0, 0) == S_OK);
    d3dGraphicsQueueWait();
}

int main(int argc, char** argv) {
    assert(QueryPerformanceFrequency(&perfFrequency));
    if (commandLineContain(argc, argv, "showConsole")) { showConsole(); }
    settingsLoad();
    windowInit();
    d3dInit(commandLineContain(argc, argv, "d3ddebug"));
    d3dCompilePipelines();
    applySettings(settingAll);
    ShowWindow(window.hwnd, SW_SHOW);
    sceneLoad(assetsDir / "scenes/scene.yaml");
    while (!quit) {
        QueryPerformanceCounter(&perfCounters[0]);
        mouseDeltaRaw[0] = 0, mouseDeltaRaw[1] = 0, mouseWheel = 0;
        MSG windowMsg;
        while (PeekMessageA(&windowMsg, (HWND)window.hwnd, 0, 0, PM_REMOVE)) {
            TranslateMessage(&windowMsg);
            DispatchMessageA(&windowMsg);
        }
        controllerGetInputs();
        update();
        render();
        QueryPerformanceCounter(&perfCounters[1]);
        frameTime = (double)(perfCounters[1].QuadPart - perfCounters[0].QuadPart) / (double)perfFrequency.QuadPart;
    }
    sceneSave();
    settingsSave();
    return EXIT_SUCCESS;
}
