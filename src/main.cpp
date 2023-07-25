#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <windowsx.h>
#include <shellscalingapi.h>
#include <commdlg.h>
#include <cderr.h>
#include <atlbase.h>
#include <atlconv.h>
#include <xinput.h>
#undef near
#undef far

#include <span>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <format>
#include <iostream>
#include <fstream>
#include <streambuf>

#define CGLTF_IMPLEMENTATION
#include "external/cgltf/cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_OBSOLETE_KEYIO
#define IMGUI_USE_STB_SPRINTF
#include "external/imgui/imgui.cpp"
#include "external/imgui/imgui_draw.cpp"
#include "external/imgui/imgui_widgets.cpp"
#include "external/imgui/imgui_tables.cpp"
#include "external/imgui/imgui_demo.cpp"
#undef snprintf
#undef vsnprintf

#include "external/d3d12ma/D3D12MemAlloc.h"

#define RYML_SINGLE_HDR_DEFINE_NOW
#include "external/rapidyaml/rapidyaml-0.5.0.hpp"

#define _XM_SSE4_INTRINSICS_
#include <directxmath.h>
#include <directXTex.h>
using namespace DirectX;

typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef uint32_t uint;
typedef int64_t int64;
typedef uint64_t uint64;

static const double e = 2.71828182845904523536;
static const double pi = 3.14159265358979323846;

#define kilobytes(n) (1024 * (n))
#define megabytes(n) (1024 * 1024 * (n))
#define gigabytes(n) (1024 * 1024 * 1024 * (n))

#define identityMat4x4 { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }

template <typename T>
void check(T result) {
	assert(result);
}

void check(HRESULT result) {
	assert(result == S_OK);
}

template<typename T, uint N>
constexpr uint countof(const T(&)[N]) { return N; }

template<typename T>
uint64 vectorSizeof(const std::vector<T>& v) { return v.size() * sizeof(T); }

template<typename T>
T align(T x, T n) {
	T remainder = x % n;
	return remainder == 0 ? x : x + (n - remainder);
}

struct float2 {
	float x = 0; float y = 0;
};

struct float3 {
	float x = 0; float y = 0; float z = 0;
	float3() = default;
	float3(float x, float y, float z) : x(x), y(y), z(z) {}
	float3(XMVECTOR v) : x(XMVectorGetX(v)), y(XMVectorGetY(v)), z(XMVectorGetZ(v)) {}
	void operator=(XMVECTOR v) { x = XMVectorGetX(v); y = XMVectorGetY(v); z = XMVectorGetZ(v); }
	operator XMVECTOR() const { return XMVectorSet(x, y, z, 0); }
	float3 operator+(float3 v) const { return float3(x + v.x, y + v.y, z + v.z); }
	void operator+=(float3 v) { x += v.x; y += v.y; z += v.z; }
	float3 operator-() const { return float3(-x, -y, -z); }
	float3 operator-(float3 v) const { return float3(x - v.x, y - v.y, z - v.z); }
	void operator-=(float3 v) { x -= v.x; y -= v.y; z -= v.z; }
	float3 operator*(float scale) const { return float3(x * scale, y * scale, z * scale); }
	void operator*=(float scale) { x *= scale; y *= scale; z *= scale; };
	float3 operator/(float scale) const { return float3(x / scale, y / scale, z / scale); }
	void operator/=(float scale) { x /= scale; y /= scale; z /= scale; };
	float dot(float3 v) const { return x * v.x + y * v.y + z * v.z; }
	float3 cross(float3 v) const { return float3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
	float3 normalize() const { XMVECTOR v = XMVector3Normalize(XMVectorSet(x, y, z, 0)); return { XMVectorGetX(v), XMVectorGetY(v), XMVectorGetZ(v) }; }
	float length() const { return sqrtf(x * x + y * y + z * z); };
	std::string toString() const { return std::format("[{}, {}, {}]", x, y, z); };
};


// get pitch, yaw of a directional unit vector from vector (1, 0, 0)
void getPitchYaw(float3 dir, float* pitch, float* yaw) {
	*pitch = asinf(dir.y);
	float xMax = sqrtf(1 - dir.y * dir.y);
	*yaw = acos(dir.x / xMax) * (dir.z < 0 ? 1 : -1);
}

bool setCurrentDirToExeDir() {
	char path[512];
	DWORD n = GetModuleFileNameA(nullptr, path, sizeof(path));
	if (n < countof(path) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
		char* pathPtr = strrchr(path, '\\');
		if (pathPtr) {
			pathPtr[0] = '\0';
			if (SetCurrentDirectoryA(path)) {
				return true;
			}
		}
	}
	return false;
}

bool fileExists(const char* filePath) {
	DWORD dwAttrib = GetFileAttributesA(filePath);
	return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

std::string fileReadStr(const char* filePath) {
	std::ifstream t(filePath);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

bool fileWriteStr(const char* filePath, const std::string& str) {
	HANDLE hwnd = CreateFileA(filePath, GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hwnd != INVALID_HANDLE_VALUE) {
		DWORD bytesWritten = 0;
		if (WriteFile(hwnd, str.c_str(), (DWORD)str.length(), &bytesWritten, nullptr)) {
			CloseHandle(hwnd);
			return true;
		}
		else {
			CloseHandle(hwnd);
			return false;
		}
	}
	return false;
}

struct FileData {
	const char* data;
	uint size;
	~FileData() { delete[] data; }
};

FileData fileRead(const char* filePath) {
	HANDLE hwnd = CreateFileA(filePath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hwnd != INVALID_HANDLE_VALUE) {
		DWORD size = GetFileSize(hwnd, nullptr);
		if (size != INVALID_FILE_SIZE) {
			char* data = new char[size];
			DWORD byteRead;
			if (ReadFile(hwnd, data, size, &byteRead, nullptr) && byteRead == size) {
				CloseHandle(hwnd);
				return FileData{ .data = data, .size = size };
			}
			delete[] data;
		}
		CloseHandle(hwnd);
	}
	return {};
}

bool commandLineContain(int argc, char** argv, const char* str) {
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], str)) {
			return true;
		}
	}
	return false;
}

void showConsole() {
	if (AllocConsole()) {
		freopen_s((FILE**)stdin, "CONIN$", "r", stdin);
		freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
		freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);
		HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
		check(SetConsoleMode(hStdin, ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));
	}
}

enum class WindowMode {
	windowed,
	borderless,
	fullscreen
};

struct Settings {
	WindowMode windowMode = WindowMode::windowed;
	uint windowX = 0;
	uint windowY = 0;
	uint windowW = 1920;
	uint windowH = 1200;
	uint renderW;
	uint renderH;
	DXGI_RATIONAL refreshRate = { 60, 1 };
	bool hdr = false;
};

static Settings settings = {};

void settingsLoad() {
	if (fileExists("settings.yaml")) {
		std::string yamlStr = fileReadStr("settings.yaml");
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
	check(fileWriteStr("settings.yaml", yamlStr));
}

struct Window {
	HWND hwnd;
};

static Window window = {};

void windowInit() {
	check(SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE));

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
	check(RegisterClassA(&windowClass));

	DWORD windowStyle = WS_OVERLAPPEDWINDOW;
	window.hwnd = CreateWindowExA(
		0, windowClass.lpszClassName, nullptr, windowStyle,
		settings.windowX, settings.windowY, settings.windowW, settings.windowH,
		nullptr, nullptr, instanceHandle, nullptr
	);
	check(window.hwnd);

	RECT windowRect;
	GetWindowRect(window.hwnd, &windowRect);
	settings.windowX = windowRect.left;
	settings.windowY = windowRect.top;
	settings.windowW = windowRect.right - windowRect.left;
	settings.windowH = windowRect.bottom - windowRect.top;
	RECT clientRect;
	GetClientRect(window.hwnd, &clientRect);
	settings.renderW = clientRect.right - clientRect.left;
	settings.renderH = clientRect.bottom - clientRect.top;

	RAWINPUTDEVICE rawInputDevice = { .usUsagePage = 0x01, .usUsage = 0x02, /*.dwFlags = RIDEV_INPUTSINK, .hwndTarget = window.handle*/ };
	check(RegisterRawInputDevices(&rawInputDevice, 1, sizeof(rawInputDevice)));
}

struct DisplayMode {
	struct Resolution {
		uint width;
		uint height;
	};
	Resolution resolution;
	std::vector<DXGI_RATIONAL> refreshRates;
	bool operator==(Resolution res) {
		return res.width == resolution.width && res.height == resolution.height;
	}
	void addRefreshRate(DXGI_RATIONAL rate) {
		if (!std::any_of(refreshRates.begin(), refreshRates.end(), [rate](auto& r) { return r.Numerator == rate.Numerator && r.Denominator == rate.Denominator; })) {
			refreshRates.push_back(rate);
		}
	}
};

static const uint d3dBufferCount = 2;

struct D3D {
	IDXGIOutput6* dxgiOutput;
	IDXGIAdapter4* dxgiAdapter;
	std::vector<DisplayMode> displayModes;
	ID3D12Device5* device;
	ID3D12Debug1* debugController;

	uint bufferIndex;

	ID3D12CommandQueue* graphicsQueue;
	ID3D12Fence* graphicsQueueFence;
	HANDLE graphicsQueueFenceEvent;
	uint64 graphicsQueueFenceCounter;
	uint64 graphicsQueueFenceValues[d3dBufferCount];
	ID3D12CommandAllocator* graphicsCmdAllocators[d3dBufferCount];
	ID3D12GraphicsCommandList4* graphicsCmdList;

	ID3D12CommandQueue* transferQueue;
	ID3D12Fence* transferQueueFence;
	HANDLE transferQueueFenceEvent;
	uint64 transferQueueFenceCounter;
	ID3D12CommandAllocator* transferCmdAllocator;
	ID3D12GraphicsCommandList4* transferCmdList;

	IDXGISwapChain4* swapChain;
	DXGI_FORMAT swapChainFormat;
	ID3D12Resource* swapChainImages[d3dBufferCount];
	D3D12_CPU_DESCRIPTOR_HANDLE swapChainImageRTVDescriptors[d3dBufferCount];

	ID3D12DescriptorHeap* rtvDescriptorHeap;
	uint rtvDescriptorCount;
	uint rtvDescriptorSize;
	ID3D12DescriptorHeap* cbvSrvUavDescriptorHeaps[d3dBufferCount];
	uint cbvSrvUavDescriptorCounts[d3dBufferCount];
	uint cbvSrvUavDescriptorSize;

	D3D12MA::Allocator* allocator;

	D3D12MA::Allocation* stagingBuffer;
	char* stagingBufferPtr;
	D3D12MA::Allocation* constantBuffers[d3dBufferCount];
	char* constantBufferPtrs[d3dBufferCount];

	D3D12MA::Allocation* readBackBufferUavs[d3dBufferCount];
	D3D12MA::Allocation* readBackBuffers[d3dBufferCount];
	char* readBackBufferPtrs[d3dBufferCount];

	D3D12MA::Allocation* renderTexture;
	D3D12MA::Allocation* accumulationRenderTexture;

	D3D12MA::Allocation* imguiTexture;
	D3D12MA::Allocation* imguiVertexBuffers[d3dBufferCount];
	char* imguiVertexBufferPtrs[d3dBufferCount];
	D3D12MA::Allocation* imguiIndexBuffers[d3dBufferCount];
	char* imguiIndexBufferPtrs[d3dBufferCount];

	D3D12MA::Allocation* tlasInstancesBuildInfos[d3dBufferCount];
	char* tlasInstancesBuildInfosPtrs[d3dBufferCount];
	D3D12MA::Allocation* tlasInstancesExtraInfos[d3dBufferCount];
	char* tlasInstancesExtraInfosPtrs[d3dBufferCount];
	D3D12MA::Allocation* tlas[d3dBufferCount];
	D3D12MA::Allocation* tlasScratch[d3dBufferCount];

	D3D12MA::Allocation* collisionQueries[d3dBufferCount];
	char* collisionQueriesPtrs[d3dBufferCount];
	D3D12MA::Allocation* collisionQueryResultsUavs[d3dBufferCount];
	D3D12MA::Allocation* collisionQueryResults[d3dBufferCount];
	char* collisionQueryResultsPtrs[d3dBufferCount];

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

D3DDescriptor d3dAppendDescriptorCBV(D3D12_CONSTANT_BUFFER_VIEW_DESC* constantBufferViewDesc) {
	uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex];
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetCPUDescriptorHandleForHeapStart().ptr + offset };
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetGPUDescriptorHandleForHeapStart().ptr + offset };
	d3d.device->CreateConstantBufferView(constantBufferViewDesc, cpuHandle);
	d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex] += 1;
	return { cpuHandle, gpuHandle };
}

D3DDescriptor d3dAppendDescriptorSRV(D3D12_SHADER_RESOURCE_VIEW_DESC* resourceViewDesc, ID3D12Resource* resource) {
	uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex];
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetCPUDescriptorHandleForHeapStart().ptr + offset };
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetGPUDescriptorHandleForHeapStart().ptr + offset };
	d3d.device->CreateShaderResourceView(resource, resourceViewDesc, cpuHandle);
	d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex] += 1;
	return { cpuHandle, gpuHandle };
}

D3DDescriptor d3dAppendDescriptorUAV(D3D12_UNORDERED_ACCESS_VIEW_DESC* unorderedAccessViewDesc, ID3D12Resource* resource) {
	uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex];
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetCPUDescriptorHandleForHeapStart().ptr + offset };
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = { d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]->GetGPUDescriptorHandleForHeapStart().ptr + offset };
	d3d.device->CreateUnorderedAccessView(resource, nullptr, unorderedAccessViewDesc, cpuHandle);
	d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex] += 1;
	return { cpuHandle, gpuHandle };
}

void d3dGraphicsQueueStartRecording() {
	check(d3d.graphicsCmdAllocators[d3d.bufferIndex]->Reset());
	check(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocators[d3d.bufferIndex], nullptr));
}

void d3dGraphicsQueueSubmitRecording() {
	check(d3d.graphicsCmdList->Close());
	d3d.graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.graphicsCmdList);
}

void d3dGraphicsQueueWait() {
	d3d.graphicsQueueFenceCounter += 1;
	check(d3d.graphicsQueue->Signal(d3d.graphicsQueueFence, d3d.graphicsQueueFenceCounter));
	d3d.graphicsQueueFenceValues[d3d.bufferIndex] = d3d.graphicsQueueFenceCounter;
	d3d.bufferIndex = (d3d.bufferIndex + 1) % d3dBufferCount;
	uint64 fenceValue = d3d.graphicsQueueFenceValues[d3d.bufferIndex];
	if (d3d.graphicsQueueFence->GetCompletedValue() < fenceValue) {
		check(d3d.graphicsQueueFence->SetEventOnCompletion(fenceValue, d3d.graphicsQueueFenceEvent));
		check(WaitForSingleObjectEx(d3d.graphicsQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
	}
}

void d3dGraphicsQueueFlush() {
	d3d.graphicsQueueFenceCounter += 1;
	d3d.graphicsQueue->Signal(d3d.graphicsQueueFence, d3d.graphicsQueueFenceCounter);
	d3d.graphicsQueueFenceValues[d3d.bufferIndex] = d3d.graphicsQueueFenceCounter;
	d3d.bufferIndex = (d3d.bufferIndex + 1) % d3dBufferCount;
	uint64 fenceValue = d3d.graphicsQueueFenceCounter;
	if (d3d.graphicsQueueFence->GetCompletedValue() < fenceValue) {
		check(d3d.graphicsQueueFence->SetEventOnCompletion(fenceValue, d3d.graphicsQueueFenceEvent));
		check(WaitForSingleObjectEx(d3d.graphicsQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
	}
}

void d3dTransferQueueStartRecording() {
	check(d3d.transferCmdAllocator->Reset());
	check(d3d.transferCmdList->Reset(d3d.transferCmdAllocator, nullptr));
}

void d3dTransferQueueSubmitRecording() {
	check(d3d.transferCmdList->Close());
	d3d.transferQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.transferCmdList);
}

void d3dTransferQueueFlush() {
	d3d.transferQueueFenceCounter += 1;
	d3d.transferQueue->Signal(d3d.transferQueueFence, d3d.transferQueueFenceCounter);
	if (d3d.transferQueueFence->GetCompletedValue() < d3d.transferQueueFenceCounter) {
		check(d3d.transferQueueFence->SetEventOnCompletion(d3d.transferQueueFenceCounter, d3d.transferQueueFenceEvent));
		check(WaitForSingleObjectEx(d3d.transferQueueFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
	}
}

void d3dInit(bool debug) {
	uint factoryFlags = 0;
	if (debug) {
		factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
		check(D3D12GetDebugInterface(IID_PPV_ARGS(&d3d.debugController)));
		d3d.debugController->EnableDebugLayer();
		//d3d.debugController->SetEnableGPUBasedValidation(true);
		//d3d.debugController->SetEnableSynchronizedCommandQueueValidation(true);
	}

	IDXGIFactory7* dxgiFactory = nullptr;
	DXGI_ADAPTER_DESC dxgiAdapterDesc = {};
	DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};

	check(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&dxgiFactory)));
	check(dxgiFactory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&d3d.dxgiAdapter)));
	check(d3d.dxgiAdapter->GetDesc(&dxgiAdapterDesc));
	check(d3d.dxgiAdapter->EnumOutputs(0, (IDXGIOutput**)&d3d.dxgiOutput));
	check(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc));
	if (dxgiOutputDesc.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
		settings.hdr = false;
	}
	check(D3D12CreateDevice(d3d.dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d.device)));

	D3D12_FEATURE_DATA_D3D12_OPTIONS features = {};
	check(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &features, sizeof(features)));
	check(features.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_6 };
	check(d3d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)));
	check(shaderModel.HighestShaderModel == D3D_SHADER_MODEL_6_6);
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 rayTracingFeatures = {};
	check(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &rayTracingFeatures, sizeof(rayTracingFeatures)));
	check(rayTracingFeatures.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
	{
		D3D12_COMMAND_QUEUE_DESC graphicsQueueDesc = { .Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE };
		check(d3d.device->CreateCommandQueue(&graphicsQueueDesc, IID_PPV_ARGS(&d3d.graphicsQueue)));
		D3D12_COMMAND_QUEUE_DESC transferQueueDesc = { .Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE };
		check(d3d.device->CreateCommandQueue(&transferQueueDesc, IID_PPV_ARGS(&d3d.transferQueue)));

		for (auto& allocator : d3d.graphicsCmdAllocators) {
			check(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
		}
		check(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.transferCmdAllocator)));

		check(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.graphicsCmdAllocators[0], nullptr, IID_PPV_ARGS(&d3d.graphicsCmdList)));
		check(d3d.graphicsCmdList->Close());
		check(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.transferCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.transferCmdList)));
		check(d3d.transferCmdList->Close());

		check(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.graphicsQueueFence)));
		d3d.graphicsQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
		check(d3d.graphicsQueueFenceEvent);
		check(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.transferQueueFence)));
		d3d.transferQueueFenceEvent = CreateEventA(nullptr, false, false, nullptr);
		check(d3d.transferQueueFenceEvent);
	}
	{
		d3d.swapChainFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
		uint dxgiModeCount = 0;
		d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, nullptr);
		std::vector<DXGI_MODE_DESC> dxgiModes(dxgiModeCount);
		d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, dxgiModes.data());
		for (auto& dxgiMode : dxgiModes) {
			DisplayMode::Resolution res = { dxgiMode.Width, dxgiMode.Height };
			auto modeIter = std::find(d3d.displayModes.begin(), d3d.displayModes.end(), res);
			if (modeIter == d3d.displayModes.end()) {
				d3d.displayModes.push_back({ {dxgiMode.Width, dxgiMode.Height}, {dxgiMode.RefreshRate} });
			}
			else {
				modeIter->addRefreshRate(dxgiMode.RefreshRate);
			}
		}

		DXGI_SWAP_CHAIN_DESC1 desc = {
			.Width = (uint)settings.renderW, .Height = (uint)settings.renderH,
			.Format = d3d.swapChainFormat, .SampleDesc = {.Count = 1 },
			.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER,
			.BufferCount = d3dBufferCount,
			.Scaling = DXGI_SCALING_NONE,
			.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
			.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH,
		};
		check(dxgiFactory->CreateSwapChainForHwnd(d3d.graphicsQueue, window.hwnd, &desc, nullptr, nullptr, (IDXGISwapChain1**)&d3d.swapChain));

		DXGI_COLOR_SPACE_TYPE colorSpace = settings.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
		check(d3d.swapChain->SetColorSpace1(colorSpace));

		for (int i = 0; auto & image : d3d.swapChainImages) {
			check(d3d.swapChain->GetBuffer(i, IID_PPV_ARGS(&image)));
			image->SetName(std::format(L"swapChain{}", i).c_str());
			i += 1;
		}
	}
	{
		D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = { .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 16, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE };
		check(d3d.device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&d3d.rtvDescriptorHeap)));
		d3d.rtvDescriptorCount = 0;
		d3d.rtvDescriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		for (int i = 0; auto & image : d3d.swapChainImages) {
			uint offset = d3d.rtvDescriptorSize * d3d.rtvDescriptorCount;
			d3d.swapChainImageRTVDescriptors[i] = { d3d.rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset };
			d3d.device->CreateRenderTargetView(image, nullptr, d3d.swapChainImageRTVDescriptors[i]);
			d3d.rtvDescriptorCount += 1;
			i += 1;
		}

		D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavDescriptorHeapDesc = { .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = 1024, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
		for (auto& heap : d3d.cbvSrvUavDescriptorHeaps) {
			check(d3d.device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&heap)));
		}
		d3d.cbvSrvUavDescriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}
	{
		D3D12MA::ALLOCATOR_DESC allocatorDesc = {
			.Flags = D3D12MA::ALLOCATOR_FLAG_NONE,
			.pDevice = d3d.device,
			.PreferredBlockSize = 0,
			.pAllocationCallbacks = nullptr,
			.pAdapter = d3d.dxgiAdapter,
		};
		check(D3D12MA::CreateAllocator(&allocatorDesc, &d3d.allocator));
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
			{ &d3d.stagingBuffer, &d3d.stagingBufferPtr, megabytes(256), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_GENERIC_READ, L"stagingBuffer" },
			{ &d3d.constantBuffers[0], &d3d.constantBufferPtrs[0], megabytes(4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer0" },
			{ &d3d.constantBuffers[1], &d3d.constantBufferPtrs[1], megabytes(4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer1" },
			//{ &d3d.constantBuffers[2], &d3d.constantBufferPtrs[2], megabytes(4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer2" },
			{ &d3d.tlasInstancesBuildInfos[0], &d3d.tlasInstancesBuildInfosPtrs[0], megabytes(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfos0" },
			{ &d3d.tlasInstancesBuildInfos[1], &d3d.tlasInstancesBuildInfosPtrs[1], megabytes(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfos1" },
			//{ &d3d.tlasInstancesBuildInfos[2], &d3d.tlasInstancesBuildInfosPtrs[2], megabytes(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfos2" },
			{ &d3d.tlasInstancesExtraInfos[0], &d3d.tlasInstancesExtraInfosPtrs[0], megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesExtraInfos0" },
			{ &d3d.tlasInstancesExtraInfos[1], &d3d.tlasInstancesExtraInfosPtrs[1], megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesExtraInfos1" },
			//{ &d3d.tlasInstancesExtraInfos[2], &d3d.tlasInstancesExtraInfosPtrs[2], megabytes(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesExtraInfos2" },
			{ &d3d.tlas[0], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlas0" },
			{ &d3d.tlas[1], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlas1" },
			//{ &d3d.tlas[2], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlas2" },
			{ &d3d.tlasScratch[0], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratch0" },
			{ &d3d.tlasScratch[1], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratch1" },
			//{ &d3d.tlasScratch[2], nullptr, megabytes(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratch2" },
			{ &d3d.imguiVertexBuffers[0], &d3d.imguiVertexBufferPtrs[0], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer0" },
			{ &d3d.imguiVertexBuffers[1], &d3d.imguiVertexBufferPtrs[1], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer1" },
			//{ &d3d.imguiVertexBuffers[2], &d3d.imguiVertexBufferPtrs[2], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer2" },
			{ &d3d.imguiIndexBuffers[0], &d3d.imguiIndexBufferPtrs[0], megabytes(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer0" },
			{ &d3d.imguiIndexBuffers[1], &d3d.imguiIndexBufferPtrs[1], megabytes(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer1" },
			//{ &d3d.imguiIndexBuffers[2], &d3d.imguiIndexBufferPtrs[2], megabytes(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer2" },
			{ &d3d.readBackBufferUavs[0], nullptr, megabytes(2), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"readBackBufferUav0"},
			{ &d3d.readBackBufferUavs[1], nullptr, megabytes(2), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"readBackBufferUav1"},
			//{ &d3d.readBackBufferUavs[2], nullptr, megabytes(2), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"readBackBufferUav2"},
			{ &d3d.readBackBuffers[0], &d3d.readBackBufferPtrs[0], megabytes(2), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"readBackBuffer0"},
			{ &d3d.readBackBuffers[1], &d3d.readBackBufferPtrs[1], megabytes(2), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"readBackBuffer1"},
			//{ &d3d.readBackBuffers[2], &d3d.readBackBufferPtrs[2], megabytes(2), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"readBackBuffer2"},
			{ &d3d.collisionQueries[0], &d3d.collisionQueriesPtrs[0], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueries0"},
			{ &d3d.collisionQueries[1], &d3d.collisionQueriesPtrs[1], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueries1"},
			//{ &d3d.collisionQueries[2], &d3d.collisionQueriesPtrs[2], megabytes(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueries2"},
			{ &d3d.collisionQueryResultsUavs[0], nullptr, megabytes(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUav0"},
			{ &d3d.collisionQueryResultsUavs[1], nullptr, megabytes(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUav1"},
			//{ &d3d.collisionQueryResultsUavs[2], nullptr, megabytes(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUav2"},
			{ &d3d.collisionQueryResults[0], &d3d.collisionQueryResultsPtrs[0], megabytes(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResults0"},
			{ &d3d.collisionQueryResults[1], &d3d.collisionQueryResultsPtrs[1], megabytes(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResults1"},
			//{ &d3d.collisionQueryResults[2], &d3d.collisionQueryResultsPtrs[2], megabytes(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResults2"},
		};
		for (auto& desc : descs) {
			D3D12_RESOURCE_DESC bufferDesc = {
				.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Width = desc.size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
				.SampleDesc = {.Count = 1, }, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
				.Flags = desc.flags
			};
			D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = desc.heapType };
			check(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, desc.initState, nullptr, desc.buffer, {}, nullptr));
			(*desc.buffer)->GetResource()->SetName(desc.name);
			if (desc.bufferPtr) {
				check((*desc.buffer)->GetResource()->Map(0, nullptr, (void**)desc.bufferPtr));
			}
		}
	}
	{
		D3D12_RESOURCE_DESC renderTextureDesc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
			.Width = settings.renderW, .Height = settings.renderH, .DepthOrArraySize = 1, .MipLevels = 1,
			.Format = DXGI_FORMAT_R16G16B16A16_FLOAT, .SampleDesc = {.Count = 1 },
			.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		};
		D3D12_RESOURCE_DESC accumulationRenderTextureDesc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
			.Width = settings.renderW, .Height = settings.renderH, .DepthOrArraySize = 1, .MipLevels = 1,
			.Format = DXGI_FORMAT_R32G32B32A32_FLOAT, .SampleDesc = {.Count = 1 },
			.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		};

		D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
		check(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr));
		d3d.renderTexture->GetResource()->SetName(L"renderTexture");
		check(d3d.allocator->CreateResource(&allocationDesc, &accumulationRenderTextureDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &d3d.accumulationRenderTexture, {}, nullptr));
		d3d.accumulationRenderTexture->GetResource()->SetName(L"accumulationRenderTexture");
	}
	{
		uint8* imguiTextureData;
		int imguiTextureWidth;
		int imguiTextureHeight;
		ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&imguiTextureData, &imguiTextureWidth, &imguiTextureHeight);

		struct TextureDesc {
			D3D12MA::Allocation** texture;
			D3D12_RESOURCE_DESC desc;
			uint8* data;
			const wchar_t* name;
		} descs[] = {
			{
				.texture = &d3d.imguiTexture,
				.desc = {
					.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
					.Width = (uint64)imguiTextureWidth, .Height = (uint)imguiTextureHeight, .DepthOrArraySize = 1, .MipLevels = 1,
					.Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1 },
				},
				.data = imguiTextureData,
				.name = L"imguiTexture"
			}
		};

		for (auto& desc : descs) {
			D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
			check(d3d.allocator->CreateResource(&allocationDesc, &desc.desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, desc.texture, {}, nullptr));
			(*desc.texture)->GetResource()->SetName(desc.name);
		}
		uint64 stagingBufferOffset = 0;
		d3dTransferQueueStartRecording();
		for (auto& desc : descs) {
			const uint maxMipmapCount = 16;
			D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipmapFootPrints[maxMipmapCount] = {};
			uint mipmapRowCounts[maxMipmapCount] = {};
			uint64 mipmapRowSizes[maxMipmapCount] = {};
			uint64 textureSize = 0;
			d3d.device->GetCopyableFootprints(&desc.desc, 0, desc.desc.MipLevels, 0, mipmapFootPrints, mipmapRowCounts, mipmapRowSizes, &textureSize);
			check(textureSize < d3d.stagingBuffer->GetSize());
			if ((stagingBufferOffset + textureSize) >= d3d.stagingBuffer->GetSize()) {
				d3dTransferQueueSubmitRecording();
				d3dTransferQueueFlush();
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
				D3D12_TEXTURE_COPY_LOCATION dstCopyLocation = {
					.pResource = (*desc.texture)->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, .SubresourceIndex = mip
				};
				D3D12_TEXTURE_COPY_LOCATION srcCopyLocation = {
					.pResource = d3d.stagingBuffer->GetResource(), .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
					.PlacedFootprint = {.Offset = stagingBufferOffset + mipmapFootPrints[mip].Offset, .Footprint = mipmapFootPrints[mip].Footprint },
				};
				d3d.transferCmdList->CopyTextureRegion(&dstCopyLocation, 0, 0, 0, &srcCopyLocation, nullptr);
				D3D12_RESOURCE_BARRIER textureBarrier = {
					.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
					.Transition = {
						.pResource = (*desc.texture)->GetResource(), .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
						.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
					},
				};
				d3d.transferCmdList->ResourceBarrier(1, &textureBarrier);
			}
			stagingBufferOffset = align(stagingBufferOffset + textureSize, (uint64)D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
		}
		d3dTransferQueueSubmitRecording();
		d3dTransferQueueFlush();
	}
}

void d3dCompilePipelines() {
	{
		FileData rtByteCode = fileRead("renderScene.cso");
		check(rtByteCode.data);
		check(d3d.device->CreateRootSignature(0, rtByteCode.data, rtByteCode.size, IID_PPV_ARGS(&d3d.renderSceneRootSig)));
		D3D12_EXPORT_DESC exportDescs[] = {
			{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"},
			{L"rayGen"},
			{L"primaryRayMiss"}, {L"primaryRayHitGroup"}, {L"primaryRayClosestHit"},
			{L"secondaryRayMiss"}, {L"secondaryRayHitGroup"}, {L"secondaryRayClosestHit"},
		};
		D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {
			.DXILLibrary = {.pShaderBytecode = rtByteCode.data, .BytecodeLength = rtByteCode.size },
			.NumExports = countof(exportDescs), .pExports = exportDescs,
		};
		D3D12_STATE_SUBOBJECT stateSubobjects[] = { {.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc } };
		D3D12_STATE_OBJECT_DESC stateObjectDesc = { .Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects };
		check(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.renderSceneSO)));
		check(d3d.renderSceneSO->QueryInterface(IID_PPV_ARGS(&d3d.renderSceneSOProps)));
		check(d3d.renderSceneRayGenID = d3d.renderSceneSOProps->GetShaderIdentifier(L"rayGen"));
		check(d3d.renderScenePrimaryRayMissID = d3d.renderSceneSOProps->GetShaderIdentifier(L"primaryRayMiss"));
		check(d3d.renderScenePrimaryRayHitGroupID = d3d.renderSceneSOProps->GetShaderIdentifier(L"primaryRayHitGroup"));
		check(d3d.renderSceneSecondaryRayMissID = d3d.renderSceneSOProps->GetShaderIdentifier(L"secondaryRayMiss"));
		check(d3d.renderSceneSecondaryRayHitGroupID = d3d.renderSceneSOProps->GetShaderIdentifier(L"secondaryRayHitGroup"));
	}
	{
		FileData rtByteCode = fileRead("collisionDetection.cso");
		check(rtByteCode.data);
		check(d3d.device->CreateRootSignature(0, rtByteCode.data, rtByteCode.size, IID_PPV_ARGS(&d3d.collisionDetectionRootSig)));
		D3D12_EXPORT_DESC exportDescs[] = {
			{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"},
			{L"rayGen"}, {L"miss"}, {L"hitGroup"}, {L"closestHit"},
		};
		D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {
			.DXILLibrary = {.pShaderBytecode = rtByteCode.data, .BytecodeLength = rtByteCode.size },
			.NumExports = countof(exportDescs), .pExports = exportDescs,
		};
		D3D12_STATE_SUBOBJECT stateSubobjects[] = { {.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc } };
		D3D12_STATE_OBJECT_DESC stateObjectDesc = { .Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects };
		check(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.collisionDetection)));
		check(d3d.collisionDetection->QueryInterface(IID_PPV_ARGS(&d3d.collisionDetectionProps)));
		check(d3d.collisionDetectionRayGenID = d3d.collisionDetectionProps->GetShaderIdentifier(L"rayGen"));
		check(d3d.collisionDetectionMissID = d3d.collisionDetectionProps->GetShaderIdentifier(L"miss"));
		check(d3d.collisionDetectionHitGroupID = d3d.collisionDetectionProps->GetShaderIdentifier(L"hitGroup"));
	}
	{
		FileData vsByteCode = fileRead("postProcessVS.cso");
		FileData psByteCode = fileRead("postProcessPS.cso");
		check(vsByteCode.data);
		check(psByteCode.data);
		check(d3d.device->CreateRootSignature(0, psByteCode.data, psByteCode.size, IID_PPV_ARGS(&d3d.postProcessRootSig)));
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
			.VS = { vsByteCode.data, vsByteCode.size }, .PS = { psByteCode.data, psByteCode.size },
			.BlendState = {.RenderTarget = { {.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL } } },
			.SampleMask = 0xffffffff,
			.RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_BACK },
			.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
			.NumRenderTargets = 1, .RTVFormats = { d3d.swapChainFormat },
			.SampleDesc = {.Count = 1 },
		};
		check(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.postProcess)));
	}
	{
		FileData vsByteCode = fileRead("ImGuiVS.cso");
		FileData psByteCode = fileRead("ImGuiPS.cso");
		check(vsByteCode.data);
		check(psByteCode.data);
		check(d3d.device->CreateRootSignature(0, vsByteCode.data, vsByteCode.size, IID_PPV_ARGS(&d3d.imguiRootSig)));
		D3D12_INPUT_ELEMENT_DESC inputElemDescs[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
			.VS = { vsByteCode.data, vsByteCode.size },
			.PS = { psByteCode.data, psByteCode.size },
			.BlendState = {
				.RenderTarget = {
					{
						.BlendEnable = true,
						.SrcBlend = D3D12_BLEND_SRC_ALPHA, .DestBlend = D3D12_BLEND_INV_SRC_ALPHA, .BlendOp = D3D12_BLEND_OP_ADD,
						.SrcBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA, .DestBlendAlpha = D3D12_BLEND_ZERO, .BlendOpAlpha = D3D12_BLEND_OP_ADD,
						.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL,
					}
				}
			},
			.SampleMask = 0xffffffff,
			.RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = true },
			.InputLayout = { inputElemDescs, countof(inputElemDescs) },
			.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
			.NumRenderTargets = 1, .RTVFormats = { d3d.swapChainFormat },
			.SampleDesc = {.Count = 1 },
		};
		check(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.imgui)));
	}
}

void d3dResizeSwapChain(uint width, uint height) {
	d3dGraphicsQueueFlush();

	for (auto& image : d3d.swapChainImages) {
		image->Release();
	}
	check(d3d.swapChain->ResizeBuffers(countof(d3d.swapChainImages), width, height, d3d.swapChainFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));
	for (int i = 0; auto & image : d3d.swapChainImages) {
		check(d3d.swapChain->GetBuffer(i, IID_PPV_ARGS(&image)));
		image->SetName(std::format(L"swapChain{}", i).c_str());
		d3d.device->CreateRenderTargetView(image, nullptr, d3d.swapChainImageRTVDescriptors[i]);
		i += 1;
	}

	d3d.renderTexture->Release();
	d3d.accumulationRenderTexture->Release();
	D3D12_RESOURCE_DESC renderTextureDesc = {
		.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		.Width = width, .Height = height, .DepthOrArraySize = 1, .MipLevels = 1,
		.Format = DXGI_FORMAT_R16G16B16A16_FLOAT, .SampleDesc = {.Count = 1 },
		.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	};
	D3D12_RESOURCE_DESC accumulationRenderTextureDesc = {
		.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		.Width = width, .Height = height, .DepthOrArraySize = 1, .MipLevels = 1,
		.Format = DXGI_FORMAT_R32G32B32A32_FLOAT, .SampleDesc = {.Count = 1 },
		.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	};
	D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
	check(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr));
	d3d.renderTexture->GetResource()->SetName(L"renderTexture");
	check(d3d.allocator->CreateResource(&allocationDesc, &accumulationRenderTextureDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &d3d.accumulationRenderTexture, {}, nullptr));
	d3d.accumulationRenderTexture->GetResource()->SetName(L"accumulationRenderTexture");
}

static const uint64 settingsHDR = 1;
static const uint64 settingWindowMode = 1 << 1;
static const uint64 settingAll = ~0;

void applySettings(uint64 settingBits) {
	if (settingBits & settingsHDR) {
		DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};
		check(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc));
		if (settings.hdr && dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
			check(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020));
			settings.hdr = true;
		}
		else {
			check(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709));
			settings.hdr = false;
		}
	}
	if (settingBits & settingWindowMode) {
		if (settings.windowMode == WindowMode::windowed) {
			check(d3d.swapChain->SetFullscreenState(false, nullptr));
			DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
			MONITORINFO mi = { sizeof(mi) };
			check(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
			check(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW) != 0);
			check(SetWindowPos(window.hwnd, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
		}
		else if (settings.windowMode == WindowMode::borderless) {
			check(d3d.swapChain->SetFullscreenState(false, nullptr));
			DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
			MONITORINFO mi = { sizeof(mi) };
			check(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
			check(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW) != 0);
			check(SetWindowPos(window.hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
		}
		else if (settings.windowMode == WindowMode::fullscreen) {
			DXGI_MODE_DESC dxgiMode = {
				.Width = settings.windowW, .Height = settings.windowH,
				.RefreshRate = settings.refreshRate, .Format = d3d.swapChainFormat,
			};
			check(d3d.swapChain->ResizeTarget(&dxgiMode));
			check(d3d.swapChain->SetFullscreenState(true, nullptr));
			d3dResizeSwapChain(settings.windowW, settings.windowH);
		}
	}
}

#include "sceneStructs.h"

struct CameraEditor {
	float3 position;
	float3 rotation;
	float3 dir;
	void update() {
		rotation.x = std::clamp(rotation.x, float(-pi) * 0.4f, float(pi) * 0.4f);
		rotation.y = std::remainderf(rotation.y, float(pi) * 2.0f);
		XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(rotation.x, rotation.y, 0);
		dir = XMVector3Normalize(XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion));
	}
};

struct CameraThirdPerson {
	float3 lookAt;
	float3 rotation;
	float distance;
	float3 position;
	float3 dir;
	void update() {
		rotation.x = std::clamp(rotation.x, float(-pi) * 0.4f, float(pi) * 0.4f);
		rotation.y = std::remainderf(rotation.y, float(pi) * 2.0f);
		XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(rotation.x, rotation.y, 0);
		float3 d = XMVector3Normalize(XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion));
		position = lookAt + d * distance;
		dir = -d;
	}
};

struct Player {
	float3 position;
	float3 velocity;
	float3 acceleration;
	CameraThirdPerson camera;
	uint modelIndex;
};

struct ModelNode {
	XMMATRIX transform;
	uint meshIndex;
};

struct ModelPrimitive {
	std::vector<Vertex> vertices;
	std::vector<uint> indices;
	D3D12MA::Allocation* verticesBuffer;
	D3D12MA::Allocation* indicesBuffer;
};

struct ModelMesh {
	std::vector<ModelPrimitive> primitives;
	D3D12MA::Allocation* blas;
	D3D12MA::Allocation* blasScratch;
};

struct Model {
	std::string fileName;
	cgltf_data* gltfData;
	std::vector<ModelMesh> meshes;
	std::vector<ModelNode> nodes;
};

struct Terrain {
	uint modelIndex;
};

struct Building {
	std::string name;
	float3 position;
	uint modelIndex;
};

struct Skybox {
	std::string hdriTextureFileName;
	D3D12MA::Allocation* hdriTexture;
};

struct Scene {
	std::string fileName;
	CameraEditor camera;
	Player player;
	std::vector<Model> models;
	Terrain terrain;
	std::vector<Building> buildings;
	Skybox skybox;

	SceneObjectType selectedObjectType;
	uint selectedObjectIndex;
};

static Scene scene = {};
static std::vector<D3D12_RAYTRACING_INSTANCE_DESC> sceneTLASInstancesBuildInfos;
static std::vector<TLASInstanceInfo> sceneTLASInstancesExtraInfos;

Model loadModel(const char* fileName) {
	d3dTransferQueueStartRecording();

	FileData fileData = fileRead(fileName);
	check(fileData.data);
	cgltf_options options = {};
	cgltf_data* gltfData = nullptr;
	check(cgltf_parse(&options, fileData.data, fileData.size, &gltfData) == cgltf_result_success);
	check(cgltf_load_buffers(&options, gltfData, nullptr) == cgltf_result_success);
	Model model;
	model.fileName = fileName;
	model.gltfData = gltfData;
	uint64 stagingBufferOffset = 0;
	for (auto& gltfMesh : std::span(gltfData->meshes, gltfData->meshes_count)) {
		ModelMesh& mesh = model.meshes.emplace_back();
		for (auto& gltfPrimitive : std::span(gltfMesh.primitives, gltfMesh.primitives_count)) {
			cgltf_accessor* indices = gltfPrimitive.indices;
			cgltf_accessor* positions = nullptr;
			cgltf_accessor* normals = nullptr;
			for (auto& attribute : std::span(gltfPrimitive.attributes, gltfPrimitive.attributes_count)) {
				if (attribute.type == cgltf_attribute_type_position) {
					positions = attribute.data;
				}
				else if (attribute.type == cgltf_attribute_type_normal) {
					normals = attribute.data;
				}
			}
			check(gltfPrimitive.type == cgltf_primitive_type_triangles);
			check(positions && normals);
			check(positions->count == normals->count);
			check(positions->type == cgltf_type_vec3 && positions->component_type == cgltf_component_type_r_32f);
			check(normals->type == cgltf_type_vec3 && normals->component_type == cgltf_component_type_r_32f);
			check(indices->count % 3 == 0);
			check(indices->type == cgltf_type_scalar && (indices->component_type == cgltf_component_type_r_16u || indices->component_type == cgltf_component_type_r_32u));
			float* positionsBuffer = (float*)((char*)positions->buffer_view->buffer->data + positions->offset + positions->buffer_view->offset);
			float* normalsBuffer = (float*)((char*)normals->buffer_view->buffer->data + normals->offset + normals->buffer_view->offset);
			char* indicesBuffer = (char*)indices->buffer_view->buffer->data + indices->offset + indices->buffer_view->offset;

			ModelPrimitive& primitive = mesh.primitives.emplace_back();
			primitive.vertices.reserve(positions->count);
			primitive.indices.reserve(indices->count);
			for (uint i = 0; i < positions->count; i++) {
				Vertex vertex = {
					.position = { positionsBuffer[i * 3], positionsBuffer[i * 3 + 1], -positionsBuffer[i * 3 + 2] },
					.normal = { normalsBuffer[i * 3], normalsBuffer[i * 3 + 1], -normalsBuffer[i * 3 + 2] },
				};
				primitive.vertices.push_back(vertex);
			}
			if (indices->component_type == cgltf_component_type_r_16u) {
				primitive.indices.append_range(std::span((uint16*)indicesBuffer, indices->count));
			}
			else {
				primitive.indices.append_range(std::span((uint*)indicesBuffer, indices->count));
			}

			D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
			D3D12_RESOURCE_DESC resourceDesc = {
				.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Width = 0, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
				.SampleDesc = {.Count = 1 }, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
			};
			resourceDesc.Width = vectorSizeof(primitive.vertices);
			check(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &primitive.verticesBuffer, {}, nullptr));
			resourceDesc.Width = vectorSizeof(primitive.indices);
			check(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &primitive.indicesBuffer, {}, nullptr));

			memcpy(d3d.stagingBufferPtr + stagingBufferOffset, primitive.vertices.data(), vectorSizeof(primitive.vertices));
			d3d.transferCmdList->CopyBufferRegion(primitive.verticesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), stagingBufferOffset, vectorSizeof(primitive.vertices));
			stagingBufferOffset += vectorSizeof(primitive.vertices);
			memcpy(d3d.stagingBufferPtr + stagingBufferOffset, primitive.indices.data(), vectorSizeof(primitive.indices));
			d3d.transferCmdList->CopyBufferRegion(primitive.indicesBuffer->GetResource(), 0, d3d.stagingBuffer->GetResource(), stagingBufferOffset, vectorSizeof(primitive.indices));
			stagingBufferOffset += vectorSizeof(primitive.indices);
			check(stagingBufferOffset < d3d.stagingBuffer->GetSize());
		}
		std::vector<D3D12_RESOURCE_BARRIER> bufferBarriers;
		bufferBarriers.reserve(mesh.primitives.size() * 2);
		for (auto& primitive : mesh.primitives) {
			D3D12_RESOURCE_BARRIER barrier = {
				.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
				.Transition = {
					.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
					.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
				}
			};
			barrier.Transition.pResource = primitive.verticesBuffer->GetResource();
			bufferBarriers.push_back(barrier);
			barrier.Transition.pResource = primitive.indicesBuffer->GetResource();
			bufferBarriers.push_back(barrier);
		}
		d3d.transferCmdList->ResourceBarrier((uint)bufferBarriers.size(), bufferBarriers.data());
	}
	for (auto& gltfNode : std::span(gltfData->nodes, gltfData->nodes_count)) {
		if (gltfNode.mesh) {
			auto& node = model.nodes.emplace_back();
			int64 meshIndex = gltfNode.mesh - gltfData->meshes;
			check(meshIndex >= 0 && meshIndex < (int64)gltfData->meshes_count);
			node.meshIndex = (uint)meshIndex;
			float transformMat[16];
			cgltf_node_transform_world(&gltfNode, transformMat);
			node.transform = XMMATRIX(transformMat);
		}
	}
	for (auto& mesh : model.meshes) {
		std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
		for (auto& primitive : mesh.primitives) {
			D3D12_RAYTRACING_GEOMETRY_DESC& geometryDesc = geometryDescs.emplace_back();
			geometryDesc = {
				.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES, .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
				.Triangles = {
					.Transform3x4 = 0,
					.IndexFormat = DXGI_FORMAT_R32_UINT, .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
					.IndexCount = (uint)primitive.indices.size(), .VertexCount = (uint)primitive.vertices.size(),
					.IndexBuffer = primitive.indicesBuffer->GetResource()->GetGPUVirtualAddress(),
					.VertexBuffer = { primitive.verticesBuffer->GetResource()->GetGPUVirtualAddress(), sizeof(Vertex)},
				},
			};
		}
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS input = {
			.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
			.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
			.NumDescs = (uint)geometryDescs.size(),
			.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
			.pGeometryDescs = geometryDescs.data(),
		};
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
		d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &prebuildInfo);

		D3D12_RESOURCE_DESC blasBufferDesc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
			.Width = 0, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1},
			.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		};

		D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
		blasBufferDesc.Width = prebuildInfo.ResultDataMaxSizeInBytes;
		check(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &mesh.blas, {}, nullptr));
		mesh.blas->GetResource()->SetName(L"blas");
		blasBufferDesc.Width = prebuildInfo.ScratchDataSizeInBytes;
		check(d3d.allocator->CreateResource(&allocationDesc, &blasBufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &mesh.blasScratch, {}, nullptr));
		mesh.blasScratch->GetResource()->SetName(L"blasScratch");

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {
			.DestAccelerationStructureData = mesh.blas->GetResource()->GetGPUVirtualAddress(),
			.Inputs = input,
			.ScratchAccelerationStructureData = mesh.blasScratch->GetResource()->GetGPUVirtualAddress(),
		};
		d3d.transferCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
	}

	d3dTransferQueueSubmitRecording();
	d3dTransferQueueFlush();

	return model;
}

uint sceneGetModelIndex(const char* fileName) {
	auto modelIter = std::find_if(scene.models.begin(), scene.models.end(), [&](auto& model) { return model.fileName == fileName; });
	if (modelIter == scene.models.end()) {
		Model model = loadModel(fileName);
		scene.models.push_back(std::move(model));
		return (uint)scene.models.size() - 1;
	}
	else {
		return (uint)std::distance(scene.models.begin(), modelIter);
	}
}

void sceneInit(const char* fileName) {
	std::string yamlStr = fileReadStr(fileName);
	ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
	ryml::ConstNodeRef yamlRoot = yamlTree.rootref();

	scene.fileName = fileName;
	{
		ryml::ConstNodeRef cameraYaml = yamlRoot["camera"];
		ryml::ConstNodeRef cameraPositionYaml = cameraYaml["position"];
		ryml::ConstNodeRef cameraRotationYaml = cameraYaml["rotation"];
		cameraPositionYaml[0] >> scene.camera.position.x; cameraPositionYaml[1] >> scene.camera.position.y; cameraPositionYaml[2] >> scene.camera.position.z;
		cameraRotationYaml[0] >> scene.camera.rotation.x; cameraRotationYaml[1] >> scene.camera.rotation.y; cameraRotationYaml[2] >> scene.camera.rotation.z;
		scene.camera.update();
	}
	{
		ryml::ConstNodeRef playerYaml = yamlRoot["player"];
		ryml::ConstNodeRef playerPositionYaml = playerYaml["position"];
		ryml::ConstNodeRef playerVelocityYaml = playerYaml["velocity"];
		ryml::ConstNodeRef playerAccelerationYaml = playerYaml["acceleration"];
		ryml::ConstNodeRef playerCameraRotationYaml = playerYaml["cameraRotation"];
		ryml::ConstNodeRef playerCameraDistanceYaml = playerYaml["cameraDistance"];
		playerPositionYaml[0] >> scene.player.position.x; playerPositionYaml[1] >> scene.player.position.y; playerPositionYaml[2] >> scene.player.position.z;
		playerVelocityYaml[0] >> scene.player.velocity.x; playerVelocityYaml[1] >> scene.player.velocity.y; playerVelocityYaml[2] >> scene.player.velocity.z;
		playerAccelerationYaml[0] >> scene.player.acceleration.x; playerAccelerationYaml[1] >> scene.player.acceleration.y; playerAccelerationYaml[2] >> scene.player.acceleration.z;
		playerCameraRotationYaml[0] >> scene.player.camera.rotation.x; playerCameraRotationYaml[1] >> scene.player.camera.rotation.y; playerCameraRotationYaml[2] >> scene.player.camera.rotation.z;
		playerCameraDistanceYaml >> scene.player.camera.distance;
		scene.player.camera.lookAt = scene.player.position;
		scene.player.camera.update();
		std::string file;
		playerYaml["file"] >> file;
		scene.player.modelIndex = sceneGetModelIndex(file.c_str());
	}
	{
		ryml::ConstNodeRef terrainYaml = yamlRoot["terrain"];
		std::string terrainFile;
		terrainYaml["file"] >> terrainFile;
		scene.terrain.modelIndex = sceneGetModelIndex(terrainFile.c_str());
	}
	ryml::ConstNodeRef buildingsYaml = yamlRoot["buildings"];
	for (ryml::ConstNodeRef const& buildingYaml : buildingsYaml) {
		Building& building = scene.buildings.emplace_back();
		buildingYaml["name"] >> building.name;
		std::string buildingFile;
		buildingYaml["file"] >> buildingFile;
		building.modelIndex = sceneGetModelIndex(buildingFile.c_str());
		ryml::ConstNodeRef buildingPositionYaml = buildingYaml["position"];
		buildingPositionYaml[0] >> building.position.x; buildingPositionYaml[1] >> building.position.y; buildingPositionYaml[2] >> building.position.z;
	}
	{
		ryml::ConstNodeRef skyboxYaml = yamlRoot["skybox"];
		skyboxYaml["file"] >> scene.skybox.hdriTextureFileName;
		CA2W skyboxFileW(scene.skybox.hdriTextureFileName.c_str());
		TexMetadata textureMetaData;
		auto textureData = std::make_unique<ScratchImage>();
		check(LoadFromDDSFile(skyboxFileW, DDS_FLAGS_NONE, &textureMetaData, *textureData));
		D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = D3D12_HEAP_TYPE_DEFAULT };
		D3D12_RESOURCE_DESC resourceDesc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
			.Width = (uint)textureMetaData.width, .Height = (uint)textureMetaData.height, .DepthOrArraySize = 1, .MipLevels = 1,
			.Format = textureMetaData.format, .SampleDesc = {.Count = 1 }
		};
		check(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &scene.skybox.hdriTexture, {}, nullptr));
		check(scene.skybox.hdriTexture->GetResource()->SetName(L"skyboxHDRITexture"));
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT copyableFootprint; uint64 totalSize;
		d3d.device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, &copyableFootprint, nullptr, nullptr, &totalSize);
		check(copyableFootprint.Footprint.RowPitch == textureData->GetImages()->rowPitch);
		check(textureData->GetImages()->slicePitch == totalSize);
		check(totalSize < d3d.stagingBuffer->GetSize());
		memcpy(d3d.stagingBufferPtr, textureData->GetImages()->pixels, totalSize);

		d3dTransferQueueStartRecording();
		D3D12_TEXTURE_COPY_LOCATION copyDst = {
			.pResource = scene.skybox.hdriTexture->GetResource(),
			.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, .SubresourceIndex = 0,
		};
		D3D12_TEXTURE_COPY_LOCATION copySrc = {
			.pResource = d3d.stagingBuffer->GetResource(),
			.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = {.Offset = 0, .Footprint = copyableFootprint.Footprint }
		};
		d3d.transferCmdList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);
		D3D12_RESOURCE_BARRIER barrier = {
			.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
			.Transition = {
				.pResource = copyDst.pResource, .Subresource = 0,
				.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			}
		};
		d3d.transferCmdList->ResourceBarrier(1, &barrier);
		d3dTransferQueueSubmitRecording();
		d3dTransferQueueFlush();
	}
}

void sceneSave() {
	ryml::Tree yamlTree;
	ryml::NodeRef yamlRoot = yamlTree.rootref(); yamlRoot |= ryml::MAP;

	ryml::NodeRef cameraYaml = yamlRoot["camera"]; cameraYaml |= ryml::MAP;
	ryml::NodeRef cameraPositionYaml = cameraYaml["position"]; cameraPositionYaml |= ryml::SEQ; cameraPositionYaml |= ryml::_WIP_STYLE_FLOW_SL;
	cameraPositionYaml.append_child() << scene.camera.position.x; cameraPositionYaml.append_child() << scene.camera.position.y; cameraPositionYaml.append_child() << scene.camera.position.z;
	ryml::NodeRef cameraRotationYaml = cameraYaml["rotation"]; cameraRotationYaml |= ryml::SEQ; cameraRotationYaml |= ryml::_WIP_STYLE_FLOW_SL;
	cameraRotationYaml.append_child() << scene.camera.rotation.x; cameraRotationYaml.append_child() << scene.camera.rotation.y; cameraRotationYaml.append_child() << scene.camera.rotation.z;

	ryml::NodeRef playerYaml = yamlRoot["player"]; playerYaml |= ryml::MAP;
	ryml::NodeRef playerPositionYaml = playerYaml["position"]; playerPositionYaml |= ryml::SEQ; playerPositionYaml |= ryml::_WIP_STYLE_FLOW_SL;
	playerPositionYaml.append_child() << scene.player.position.x; playerPositionYaml.append_child() << scene.player.position.y; playerPositionYaml.append_child() << scene.player.position.z;
	ryml::NodeRef playerVelocityYaml = playerYaml["velocity"]; playerVelocityYaml |= ryml::SEQ; playerVelocityYaml |= ryml::_WIP_STYLE_FLOW_SL;
	playerVelocityYaml.append_child() << scene.player.velocity.x; playerVelocityYaml.append_child() << scene.player.velocity.y; playerVelocityYaml.append_child() << scene.player.velocity.z;
	ryml::NodeRef playerAccelerationYaml = playerYaml["acceleration"]; playerAccelerationYaml |= ryml::SEQ; playerAccelerationYaml |= ryml::_WIP_STYLE_FLOW_SL;
	playerAccelerationYaml.append_child() << scene.player.acceleration.x; playerAccelerationYaml.append_child() << scene.player.acceleration.y; playerAccelerationYaml.append_child() << scene.player.acceleration.z;
	ryml::NodeRef playerCameraRotationYaml = playerYaml["cameraRotation"]; playerCameraRotationYaml |= ryml::SEQ; playerCameraRotationYaml |= ryml::_WIP_STYLE_FLOW_SL;
	playerCameraRotationYaml.append_child() << scene.player.camera.rotation.x; playerCameraRotationYaml.append_child() << scene.player.camera.rotation.y; playerCameraRotationYaml.append_child() << scene.player.camera.rotation.z;
	playerYaml["cameraDistance"] << scene.player.camera.distance;
	playerYaml["file"] << scene.models[scene.player.modelIndex].fileName;

	ryml::NodeRef skyboxYaml = yamlRoot["skybox"]; skyboxYaml |= ryml::MAP;
	skyboxYaml["file"] << scene.skybox.hdriTextureFileName;

	ryml::NodeRef terrainYaml = yamlRoot["terrain"]; terrainYaml |= ryml::MAP;
	terrainYaml["file"] << scene.models[scene.terrain.modelIndex].fileName;

	ryml::NodeRef buildingsYaml = yamlRoot["buildings"]; buildingsYaml |= ryml::SEQ;
	for (auto& building : scene.buildings) {
		ryml::NodeRef buildingYaml = buildingsYaml.append_child(); buildingYaml |= ryml::MAP;
		buildingYaml["name"] << building.name;
		buildingYaml["file"] << scene.models[building.modelIndex].fileName;
		ryml::NodeRef buildingPositionYaml = buildingYaml["position"]; buildingPositionYaml |= ryml::SEQ; buildingPositionYaml |= ryml::_WIP_STYLE_FLOW_SL;
		buildingPositionYaml.append_child() << building.position.x; buildingPositionYaml.append_child() << building.position.y; buildingPositionYaml.append_child() << building.position.z;
		ryml::NodeRef buildingRotationYaml = buildingYaml["rotation"]; buildingRotationYaml |= ryml::SEQ; buildingRotationYaml |= ryml::_WIP_STYLE_FLOW_SL;
		buildingRotationYaml.append_child() << 0.0; buildingRotationYaml.append_child() << 0.0; buildingRotationYaml.append_child() << 0.0;
	}

	std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
	check(fileWriteStr(scene.fileName.c_str(), yamlStr));
}

void imguiInit() {
	check(ImGui::CreateContext());
	ImGui::StyleColorsLight();
	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = "imgui.ini";
	io.FontGlobalScale = 2;
	check(io.Fonts->AddFontDefault());
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
static bool cameraMoving = false;
static float controllerCameraSensitivity = 2;
static int mouseDeltaRaw[2] = {};
static uint mouseSelectX = UINT_MAX;
static uint mouseSelectY = UINT_MAX;
static SceneObjectType mouseSelectedObjectType = SceneObjectTypeNone;
static uint mouseSelectedObjectIndex = UINT_MAX;

LRESULT windowEventHandler(HWND hwnd, UINT eventType, WPARAM wParam, LPARAM lParam) {
	LRESULT result = 0;
	switch (eventType) {
	default: {
		result = DefWindowProcA(hwnd, eventType, wParam, lParam);
	} break;
	case WM_SHOWWINDOW: {
	} break;
	case WM_SIZE: {
		int width = LOWORD(lParam);
		int height = HIWORD(lParam);
		if (d3d.swapChain && width > 0 && height > 0 && (settings.renderW != width || settings.renderH != height)) {
			settings.renderW = width; settings.renderH = height;
			d3dResizeSwapChain(width, height);
		}
		RECT windowRect;
		GetWindowRect(window.hwnd, &windowRect);
		settings.windowX = windowRect.left; settings.windowY = windowRect.top;
		settings.windowW = windowRect.right - windowRect.left; settings.windowH = windowRect.bottom - windowRect.top;
	} break;
	case WM_MOVE: {
		RECT windowRect;
		GetWindowRect(window.hwnd, &windowRect);
		settings.windowX = windowRect.left; settings.windowY = windowRect.top;
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
		ImGui::GetIO().AddInputCharacter(uint(wParam));
	}break;
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
		ImGui::GetIO().AddMousePosEvent((float)GET_X_LPARAM(lParam), (float)GET_Y_LPARAM(lParam));
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
	float lsX, lsY, rsX, rsY;

	bool stickMoved() { return lsX != 0 || lsY != 0 || rsX != 0 || rsY != 0; }
};

static Controller controller = {};
static float controllerDeadZone = 0.1f;

void getControllerInputs() {
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
		controller.lt = float(state.Gamepad.bLeftTrigger) / 255.0f;
		controller.rt = float(state.Gamepad.bRightTrigger) / 255.0f;
		controller.lsX = float(state.Gamepad.sThumbLX) / 32767.0f;
		controller.lsY = float(state.Gamepad.sThumbLY) / 32767.0f;
		controller.rsX = float(state.Gamepad.sThumbRX) / 32767.0f;
		controller.rsY = float(state.Gamepad.sThumbRY) / 32767.0f;
	}
	else {
		controller = {};
	}
	prevState = state;
}

void hideCursor(bool hide) {
	if (hide) {
		POINT cursorP;
		GetCursorPos(&cursorP);
		RECT rect = { cursorP.x, cursorP.y, cursorP.x, cursorP.y };
		ClipCursor(&rect);
		ShowCursor(false);
	}
	else {
		ClipCursor(nullptr);
		ShowCursor(true);
	}
}

void update() {
	ImGui::GetIO().DeltaTime = (float)frameTime;
	ImGui::GetIO().DisplaySize = { (float)settings.renderW, (float)settings.renderH };
	ImGui::NewFrame();

	static ImVec2 mousePosPrev = ImGui::GetMousePos();
	ImVec2 mousePos = ImGui::GetMousePos();
	ImVec2 mouseDelta = mousePos - mousePosPrev;
	mousePosPrev = mousePos;

	if (d3d.graphicsQueueFenceValues[d3d.bufferIndex] > 0) {
		ReadBackBuffer* readBackBuffer = (ReadBackBuffer*)d3d.readBackBufferPtrs[d3d.bufferIndex];
		uint mouseSelectInstanceIndex = readBackBuffer->mouseSelectInstanceIndex;
		if (mouseSelectInstanceIndex < sceneTLASInstancesExtraInfos.size()) {
			TLASInstanceInfo& info = sceneTLASInstancesExtraInfos[mouseSelectInstanceIndex];
			mouseSelectedObjectType = info.objectType;
			mouseSelectedObjectIndex = info.objectIndex;
		}
		scene.player.position = readBackBuffer->playerPosition;
		scene.player.velocity = readBackBuffer->playerVelocity;
		scene.player.acceleration = readBackBuffer->playerAcceleration;
	}

	if (playing) {
		float xRotate = -float(mouseDeltaRaw[1]) / 500;
		float yRotate = float(mouseDeltaRaw[0]) / 500;
		xRotate -= controller.rsY * float(frameTime) * controllerCameraSensitivity;
		yRotate += controller.rsX * float(frameTime) * controllerCameraSensitivity;
		scene.player.camera.rotation.x += xRotate;
		scene.player.camera.rotation.y += yRotate;
		scene.player.camera.lookAt = scene.player.position;
		scene.player.camera.update();
	}
	else {
		if (!ImGui::GetIO().WantCaptureMouse and ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
			mouseSelectX = (uint)mousePos.x;
			mouseSelectY = (uint)mousePos.y;
		}
		else {
			mouseSelectX = UINT_MAX;
			mouseSelectY = UINT_MAX;
		}

		if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
			cameraMoving = true;
			hideCursor(true);
		}
		if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
			cameraMoving = false;
			hideCursor(false);
		}

		if (cameraMoving || controller.stickMoved()) {
			float xRotate = float(mouseDeltaRaw[1]) / 500;
			float yRotate = float(mouseDeltaRaw[0]) / 500;
			xRotate += controller.rsY * float(frameTime) * controllerCameraSensitivity;
			yRotate += controller.rsX * float(frameTime) * controllerCameraSensitivity;
			CameraEditor& camera = scene.camera;
			camera.rotation.x += xRotate;
			camera.rotation.y += yRotate;
			camera.update();
			float distance = float(frameTime * 50);
			if (ImGui::IsKeyDown(ImGuiKey_W)) { camera.position += camera.dir * distance; }
			else if (ImGui::IsKeyDown(ImGuiKey_S)) { camera.position += camera.dir * -distance; }
			else if (ImGui::IsKeyDown(ImGuiKey_A)) { camera.position += camera.dir.cross(float3{ 0, 1, 0 }) * distance; }
			else if (ImGui::IsKeyDown(ImGuiKey_D)) { camera.position += camera.dir.cross(float3{ 0, 1, 0 }) * -distance; }
			camera.position += camera.dir.cross(float3{ 0, 1, 0 }) * distance * -controller.lsX;
			camera.position += camera.dir * distance * controller.lsY;
		}
	}

	ImVec2 mainMenuBarPos;
	ImVec2 mainMenuBarSize;
	if (ImGui::BeginMainMenuBar()) {
		mainMenuBarPos = ImGui::GetWindowPos();
		mainMenuBarSize = ImGui::GetWindowSize();
		if (ImGui::BeginMenu("File")) {
			if (ImGui::BeginMenu("New")) {
				if (ImGui::MenuItem("Scene")) {
					//addScene("New Scene", "");
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Open")) {
				if (ImGui::MenuItem("Scene")) {
					char fileName[256] = {};
					OPENFILENAMEA openFileName = { .lStructSize = sizeof(openFileName), .hwndOwner = GetActiveWindow(), .lpstrFile = fileName, .nMaxFile = sizeof(fileName) };
					if (GetOpenFileNameA(&openFileName)) {
						//addScene("New Scene", filePath);
					}
					else {
						DWORD error = CommDlgExtendedError();
						if (error == FNERR_BUFFERTOOSMALL) {
						}
						else if (error == FNERR_INVALIDFILENAME) {
						}
						else if (error == FNERR_SUBCLASSFAILURE) {
						}
					}
				}
				ImGui::EndMenu();
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Quit")) {
				quit = true;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Display")) {
			if (ImGui::MenuItem(settings.hdr ? "HDR (On)" : "HDR (Off)")) {
				settings.hdr = !settings.hdr;
				applySettings(settingsHDR);
			}
			else if (ImGui::MenuItem("Windowed")) {
				settings.windowMode = WindowMode::windowed;
				applySettings(settingWindowMode);
			}
			else if (ImGui::MenuItem("Borderless Fullscreen")) {
				settings.windowMode = WindowMode::borderless;
				applySettings(settingWindowMode);
			}
			ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
			ImGui::Text("Exclusive Fullscreen");
			for (auto& mode : d3d.displayModes) {
				std::string text = std::format("{}x{}", mode.resolution.width, mode.resolution.height);
				if (ImGui::BeginMenu(text.c_str())) {
					for (auto& refreshRate : mode.refreshRates) {
						text = std::format("{:.2f}hz", (float)refreshRate.Numerator / refreshRate.Denominator);
						if (ImGui::MenuItem(text.c_str())) {
							settings.windowMode = WindowMode::fullscreen;
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
		if (ImGui::BeginMenu("Game")) {
			if (ImGui::MenuItem("Play", "CTRL+P")) {
				playing = !playing;
				hideCursor(playing);
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (ImGui::Begin("Properties")) {
		if (mouseSelectedObjectType == SceneObjectTypeTerrain) {
			ImGui::Text("Terrain");
		}
		else if (mouseSelectedObjectType == SceneObjectTypeBuilding) {
			Building& building = scene.buildings[mouseSelectedObjectIndex];
			ImGui::Text("Building #%d", mouseSelectedObjectIndex);
			ImGui::Text("Name \"%s\"", building.name.c_str());
			ImGui::InputFloat("X", &building.position.x, 0.01f);
			ImGui::InputFloat("Y", &building.position.y, 0.01f);
			ImGui::InputFloat("Z", &building.position.z, 0.01f);
		}
	}
	ImGui::End();

	if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyPressed(ImGuiKey_P, false)) {
		playing = !playing;
		hideCursor(playing);
	}

	ImGui::Render();
}

void render() {
	d3dGraphicsQueueStartRecording();

	D3D12_RESOURCE_BARRIER renderTextureTransition = {
		.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		.Transition = {
			.pResource = d3d.renderTexture->GetResource(),
			.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		}
	};
	d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);

	d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex] = 0;
	d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeaps[d3d.bufferIndex]);

	D3D12MA::Allocation* constantBuffer = d3d.constantBuffers[d3d.bufferIndex];
	char* constantBufferPtr = d3d.constantBufferPtrs[d3d.bufferIndex];
	uint constantBufferOffset = 0;
	{
		D3D12_CONSTANT_BUFFER_VIEW_DESC renderInfoCBVDesc = {
			.BufferLocation = constantBuffer->GetResource()->GetGPUVirtualAddress(),
			.SizeInBytes = align((uint)sizeof(struct RenderInfo), (uint)D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),
		};

		D3D12_SHADER_RESOURCE_VIEW_DESC tlasViewDesc = {
			.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
			.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
			.RaytracingAccelerationStructure = {.Location = d3d.tlas[d3d.bufferIndex]->GetResource()->GetGPUVirtualAddress() },
		};

		D3D12_SHADER_RESOURCE_VIEW_DESC tlasInstancesInfosDesc = {
			.ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
			.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
			.Buffer = {
				.NumElements = (uint)(d3d.tlasInstancesExtraInfos[d3d.bufferIndex]->GetSize() / sizeof(struct TLASInstanceInfo)),
				.StructureByteStride = (uint)sizeof(struct TLASInstanceInfo),
			},
		};

		D3D12_UNORDERED_ACCESS_VIEW_DESC readBackBufferDesc = {
			.ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
			.Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct ReadBackBuffer), }
		};

		D3D12_SHADER_RESOURCE_VIEW_DESC collisionQueriesDesc = {
			.ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
			.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
			.Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQuery), }
		};

		D3D12_UNORDERED_ACCESS_VIEW_DESC collisionQueryResultsDesc = {
			.ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
			.Buffer = {.NumElements = 1, .StructureByteStride = sizeof(struct CollisionQueryResult), }
		};

		D3DDescriptor renderTextureDescriptor = d3dAppendDescriptorSRV(nullptr, d3d.renderTexture->GetResource());
		D3DDescriptor accumulationTextureDescriptor = d3dAppendDescriptorSRV(nullptr, d3d.accumulationRenderTexture->GetResource());
		D3DDescriptor renderInfoDescriptor = d3dAppendDescriptorCBV(&renderInfoCBVDesc);
		D3DDescriptor tlasDescriptor = d3dAppendDescriptorSRV(&tlasViewDesc, nullptr);
		D3DDescriptor tlasInstancesInfosDescriptor = d3dAppendDescriptorSRV(&tlasInstancesInfosDesc, d3d.tlasInstancesExtraInfos[d3d.bufferIndex]->GetResource());
		D3DDescriptor skyboxTextureDescriptor = d3dAppendDescriptorSRV(nullptr, scene.skybox.hdriTexture->GetResource());
		D3DDescriptor readBackBufferDescriptor = d3dAppendDescriptorUAV(&readBackBufferDesc, d3d.readBackBufferUavs[d3d.bufferIndex]->GetResource());
		D3DDescriptor imguiTextureDescriptor = d3dAppendDescriptorSRV(nullptr, d3d.imguiTexture->GetResource());
		D3DDescriptor collisionQueriesDescriptor = d3dAppendDescriptorSRV(&collisionQueriesDesc, d3d.collisionQueries[d3d.bufferIndex]->GetResource());
		D3DDescriptor collisionQueryResultsDescriptor = d3dAppendDescriptorUAV(&collisionQueryResultsDesc, d3d.collisionQueryResultsUavs[d3d.bufferIndex]->GetResource());
	}
	{
		float3 cameraPosition;
		float3 cameraLookAt;
		if (playing) {
			cameraPosition = scene.player.camera.position;
			cameraLookAt = scene.player.camera.lookAt;
		}
		else {
			cameraPosition = scene.camera.position;
			cameraLookAt = scene.camera.position + scene.camera.dir;
		}
		RenderInfo renderInfo = {
			.cameraViewMat = XMMatrixTranspose(XMMatrixInverse(nullptr, XMMatrixLookAtLH(cameraPosition, cameraLookAt, XMVectorSet(0, 1, 0, 0)))),
			.cameraProjMat = XMMatrixPerspectiveFovLH(XMConvertToRadians(40), (float)settings.renderW / (float)settings.renderH, 0.001f, 1000.0f),
			.resolution = { settings.renderW, settings.renderH },
			.mouseSelectPosition = { mouseSelectX, mouseSelectY },
			.hdr = settings.hdr,
			.frameTime = (float)frameTime,
			.playerPosition = scene.player.position,
			.playerVelocity = scene.player.velocity,
			.playerAcceleration = scene.player.acceleration,
		};
		memcpy(constantBufferPtr + constantBufferOffset, &renderInfo, sizeof(renderInfo));
		constantBufferOffset += sizeof(renderInfo);

		sceneTLASInstancesBuildInfos.resize(0);
		sceneTLASInstancesExtraInfos.resize(0);
		auto addTlasInstance = [](uint modelIndex, XMMATRIX transform, const TLASInstanceInfo& info) {
			Model& model = scene.models[modelIndex];
			for (auto& node : model.nodes) {
				ModelMesh& mesh = model.meshes[node.meshIndex];
				sceneTLASInstancesExtraInfos.push_back(info);
				D3D12_RAYTRACING_INSTANCE_DESC& instanceDesc = sceneTLASInstancesBuildInfos.emplace_back();
				instanceDesc = {
					.InstanceID = d3d.cbvSrvUavDescriptorCounts[d3d.bufferIndex],
					.InstanceMask = 0xff,
					.AccelerationStructure = mesh.blas->GetResource()->GetGPUVirtualAddress(),
				};
				XMMATRIX finalTransform = XMMatrixTranspose(XMMatrixMultiply(transform, node.transform));
				memcpy(instanceDesc.Transform, &finalTransform, sizeof(instanceDesc.Transform));
				for (auto& primitive : mesh.primitives) {
					D3D12_SHADER_RESOURCE_VIEW_DESC desc = { .ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
					desc.Buffer = { .NumElements = (uint)primitive.vertices.size(), .StructureByteStride = sizeof(primitive.vertices[0]) };
					d3dAppendDescriptorSRV(&desc, primitive.verticesBuffer->GetResource());
					desc.Buffer = { .NumElements = (uint)primitive.indices.size(), .StructureByteStride = sizeof(primitive.indices[0]) };
					d3dAppendDescriptorSRV(&desc, primitive.indicesBuffer->GetResource());
				}
			}
		};
		addTlasInstance(scene.terrain.modelIndex, XMMatrixIdentity(), { SceneObjectTypeTerrain, 0, mouseSelectedObjectType == SceneObjectTypeTerrain });
		for (uint buildingIndex = 0; auto & building : scene.buildings) {
			TLASInstanceInfo tlasInstanceInfo = { SceneObjectTypeBuilding, buildingIndex, mouseSelectedObjectType == SceneObjectTypeBuilding && mouseSelectedObjectIndex == buildingIndex };
			addTlasInstance(building.modelIndex, XMMatrixTranslationFromVector(building.position), tlasInstanceInfo);
			buildingIndex += 1;
		}
		addTlasInstance(scene.player.modelIndex, XMMatrixTranslationFromVector(scene.player.position), { SceneObjectTypePlayer, 0, mouseSelectedObjectType == SceneObjectTypePlayer });

		check(vectorSizeof(sceneTLASInstancesBuildInfos) < d3d.tlasInstancesBuildInfos[0]->GetSize());
		check(vectorSizeof(sceneTLASInstancesExtraInfos) < d3d.tlasInstancesExtraInfos[0]->GetSize());
		memcpy(d3d.tlasInstancesBuildInfosPtrs[d3d.bufferIndex], sceneTLASInstancesBuildInfos.data(), vectorSizeof(sceneTLASInstancesBuildInfos));
		memcpy(d3d.tlasInstancesExtraInfosPtrs[d3d.bufferIndex], sceneTLASInstancesExtraInfos.data(), vectorSizeof(sceneTLASInstancesExtraInfos));

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {
			.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
			.NumDescs = (uint)sceneTLASInstancesBuildInfos.size(),
			.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
			.InstanceDescs = d3d.tlasInstancesBuildInfos[d3d.bufferIndex]->GetResource()->GetGPUVirtualAddress(),
		};
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
		d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);
		check(prebuildInfo.ResultDataMaxSizeInBytes < d3d.tlas[d3d.bufferIndex]->GetSize());
		check(prebuildInfo.ScratchDataSizeInBytes < d3d.tlasScratch[d3d.bufferIndex]->GetSize());

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {
			.DestAccelerationStructureData = d3d.tlas[d3d.bufferIndex]->GetResource()->GetGPUVirtualAddress(),
			.Inputs = inputs,
			.ScratchAccelerationStructureData = d3d.tlasScratch[d3d.bufferIndex]->GetResource()->GetGPUVirtualAddress(),
		};
		d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
		D3D12_RESOURCE_BARRIER tlasBarrier = {
			.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
			.UAV = {.pResource = d3d.tlas[d3d.bufferIndex]->GetResource() },
		};
		d3d.graphicsCmdList->ResourceBarrier(1, &tlasBarrier);

		D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = { .Width = settings.renderW, .Height = settings.renderH, .Depth = 1, };
		constantBufferOffset = align(constantBufferOffset, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.renderSceneRayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.RayGenerationShaderRecord = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.renderScenePrimaryRayMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		memcpy(constantBufferPtr + constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, d3d.renderSceneSecondaryRayMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.MissShaderTable = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset = align(constantBufferOffset + 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.renderScenePrimaryRayHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		memcpy(constantBufferPtr + constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, d3d.renderSceneSecondaryRayHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.HitGroupTable = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset += 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
		check(constantBufferOffset < d3d.constantBuffers[d3d.bufferIndex]->GetSize());

		d3d.graphicsCmdList->SetPipelineState1(d3d.renderSceneSO);
		d3d.graphicsCmdList->SetComputeRootSignature(d3d.renderSceneRootSig);
		d3d.graphicsCmdList->DispatchRays(&dispatchRaysDesc);
	}
	{
		D3D12_RESOURCE_BARRIER readBackBufferTransition = {
			.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
			.Transition = {
				.pResource = d3d.readBackBufferUavs[d3d.bufferIndex]->GetResource(),
				.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			}
		};
		D3D12_RESOURCE_BARRIER collisionQueryResultsTransition = {
			.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
			.Transition = {
				.pResource = d3d.collisionQueryResultsUavs[d3d.bufferIndex]->GetResource(),
				.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			}
		};
		d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
		d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

		D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = { .Width = 1, .Height = 1, .Depth = 1, };
		constantBufferOffset = align(constantBufferOffset, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.collisionDetectionRayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.RayGenerationShaderRecord = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.collisionDetectionMissID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.MissShaderTable = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset = align(constantBufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, (uint)D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		memcpy(constantBufferPtr + constantBufferOffset, d3d.collisionDetectionHitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		dispatchRaysDesc.HitGroupTable = { constantBuffer->GetResource()->GetGPUVirtualAddress() + constantBufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
		constantBufferOffset += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
		check(constantBufferOffset < d3d.constantBuffers[d3d.bufferIndex]->GetSize());

		d3d.graphicsCmdList->SetPipelineState1(d3d.collisionDetection);
		d3d.graphicsCmdList->SetComputeRootSignature(d3d.collisionDetectionRootSig);
		d3d.graphicsCmdList->DispatchRays(&dispatchRaysDesc);

		std::swap(readBackBufferTransition.Transition.StateBefore, readBackBufferTransition.Transition.StateAfter);
		std::swap(collisionQueryResultsTransition.Transition.StateBefore, collisionQueryResultsTransition.Transition.StateAfter);
		d3d.graphicsCmdList->ResourceBarrier(1, &readBackBufferTransition);
		d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsTransition);

		d3d.graphicsCmdList->CopyBufferRegion(d3d.readBackBuffers[d3d.bufferIndex]->GetResource(), 0, d3d.readBackBufferUavs[d3d.bufferIndex]->GetResource(), 0, sizeof(struct ReadBackBuffer));
		d3d.graphicsCmdList->CopyBufferRegion(d3d.collisionQueryResults[d3d.bufferIndex]->GetResource(), 0, d3d.collisionQueryResultsUavs[d3d.bufferIndex]->GetResource(), 0, d3d.collisionQueryResults[d3d.bufferIndex]->GetSize());
	}
	{
		uint swapChainBackBufferIndex = d3d.swapChain->GetCurrentBackBufferIndex();
		D3D12_RESOURCE_BARRIER swapChainImageTransition = {
			.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
			.Transition = {
				.pResource = d3d.swapChainImages[swapChainBackBufferIndex],
				.StateBefore = D3D12_RESOURCE_STATE_PRESENT, .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET,
			},
		};
		d3d.graphicsCmdList->ResourceBarrier(1, &swapChainImageTransition);
		d3d.graphicsCmdList->OMSetRenderTargets(1, &d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], false, nullptr);
		float swapChainClearColor[4] = { 0, 0, 0, 0 };
		d3d.graphicsCmdList->ClearRenderTargetView(d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], swapChainClearColor, 0, nullptr);
		D3D12_VIEWPORT viewport = { 0, 0, (float)settings.renderW, (float)settings.renderH, 0, 1 };
		RECT scissor = { 0, 0, (int)settings.renderW, (int)settings.renderH };
		d3d.graphicsCmdList->RSSetViewports(1, &viewport);
		d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
		{ // postProcess
			d3d.graphicsCmdList->SetPipelineState(d3d.postProcess);
			d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.postProcessRootSig);
			std::swap(renderTextureTransition.Transition.StateBefore, renderTextureTransition.Transition.StateAfter);
			d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);
			d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
		}
		{ // imgui
			d3d.graphicsCmdList->SetPipelineState(d3d.imgui);
			float blendFactor[] = { 0.f, 0.f, 0.f, 0.f };
			d3d.graphicsCmdList->OMSetBlendFactor(blendFactor);
			d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.imguiRootSig);

			D3D12MA::Allocation* vertBuffer = d3d.imguiVertexBuffers[d3d.bufferIndex];
			D3D12MA::Allocation* indexBuffer = d3d.imguiIndexBuffers[d3d.bufferIndex];
			char* vertBufferPtr = d3d.imguiVertexBufferPtrs[d3d.bufferIndex];
			char* indexBufferPtr = d3d.imguiIndexBufferPtrs[d3d.bufferIndex];
			D3D12_VERTEX_BUFFER_VIEW vertBufferView = { vertBuffer->GetResource()->GetGPUVirtualAddress(), (uint)vertBuffer->GetSize(), sizeof(ImDrawVert) };
			D3D12_INDEX_BUFFER_VIEW indexBufferView = { indexBuffer->GetResource()->GetGPUVirtualAddress(), (uint)indexBuffer->GetSize(), DXGI_FORMAT_R16_UINT };
			check(vertBuffer->GetResource()->Map(0, nullptr, (void**)&vertBufferPtr));
			check(indexBuffer->GetResource()->Map(0, nullptr, (void**)&indexBufferPtr));
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
				memcpy(vertBufferPtr + vertBufferOffset, dlist->VtxBuffer.Data, verticesSize);
				memcpy(indexBufferPtr + indexBufferOffset, dlist->IdxBuffer.Data, indicesSize);
				uint vertexIndex = vertBufferOffset / sizeof(ImDrawVert);
				uint indiceIndex = indexBufferOffset / sizeof(ImDrawIdx);
				for (int i = 0; i < dlist->CmdBuffer.Size; i++) {
					const ImDrawCmd* dcmd = &dlist->CmdBuffer[i];
					D3D12_RECT scissor = { (LONG)dcmd->ClipRect.x, (LONG)dcmd->ClipRect.y, (LONG)dcmd->ClipRect.z, (LONG)dcmd->ClipRect.w };
					d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
					d3d.graphicsCmdList->DrawIndexedInstanced(dcmd->ElemCount, 1, indiceIndex, vertexIndex, 0);
					indiceIndex += dcmd->ElemCount;
				}
				vertBufferOffset = vertBufferOffset + align(verticesSize, (uint)sizeof(ImDrawVert));
				indexBufferOffset = indexBufferOffset + align(indicesSize, (uint)sizeof(ImDrawIdx));
				check(vertBufferOffset < vertBuffer->GetSize());
				check(indexBufferOffset < indexBuffer->GetSize());
			}
		}
		std::swap(swapChainImageTransition.Transition.StateBefore, swapChainImageTransition.Transition.StateAfter);
		d3d.graphicsCmdList->ResourceBarrier(1, &swapChainImageTransition);
	}

	d3dGraphicsQueueSubmitRecording();
	check(d3d.swapChain->Present(0, 0));
	d3dGraphicsQueueWait();
}

int main(int argc, char** argv) {
	check(QueryPerformanceFrequency(&perfFrequency));
	check(setCurrentDirToExeDir());
	if (commandLineContain(argc, argv, "showConsole")) { showConsole(); }
	settingsLoad();
	imguiInit();
	windowInit();
	d3dInit(commandLineContain(argc, argv, "d3ddebug"));
	d3dCompilePipelines();
	applySettings(settingAll);
	ShowWindow(window.hwnd, SW_SHOW);
	sceneInit("../../assets/scenes/scene.yaml");
	while (!quit) {
		QueryPerformanceCounter(&perfCounters[0]);
		mouseDeltaRaw[0] = 0; mouseDeltaRaw[1] = 0;
		MSG windowMsg;
		while (PeekMessageA(&windowMsg, (HWND)window.hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&windowMsg);
			DispatchMessageA(&windowMsg);
		}
		getControllerInputs();
		update();
		render();
		QueryPerformanceCounter(&perfCounters[1]);
		frameTime = (double)(perfCounters[1].QuadPart - perfCounters[0].QuadPart) / (double)perfFrequency.QuadPart;
	}
	sceneSave();
	settingsSave();
	return EXIT_SUCCESS;
}
