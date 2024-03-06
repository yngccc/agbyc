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
#include <hidsdi.h>
#include <shellapi.h>
#include <shellscalingapi.h>
#include <shlobj.h>
#include <userenv.h>
#include <windows.h>
#include <windowsx.h>
#include <xinput.h>

#include <directxtex.h>
#include <pix3.h>

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

#include "common.h"
#include "structsHLSL.h"

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

struct Window {
    HWND hwnd;
};

struct DisplayMode {
    uint resolutionW;
    uint resolutionH;
    std::vector<DXGI_RATIONAL> refreshRates;
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
    uint rtvDescriptorCount;
    ID3D12DescriptorHeap* cbvSrvUavDescriptorHeap;
    uint cbvSrvUavDescriptorSize;
    uint cbvSrvUavDescriptorCapacity;
    uint cbvSrvUavDescriptorCount;

    D3D12MA::Allocator* allocator;

    D3D12MA::Allocation* stagingBuffer;
    uint8* stagingBufferPtr;
    uint stagingBufferOffset = 0;

    D3D12MA::Allocation* constantsBuffer;
    uint8* constantsBufferPtr;
    uint constantsBufferOffset = 0;

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

    ID3D12StateObject* renderScene;
    ID3D12StateObjectProperties* renderSceneProps;
    ID3D12RootSignature* renderSceneRootSig;
    void* renderSceneRayGenID;
    void* renderScenePrimaryRayMissID;
    void* renderScenePrimaryRayHitGroupID;
    void* renderSceneSecondaryRayMissID;
    void* renderSceneSecondaryRayHitGroupID;

    ID3D12StateObject* collisionQuery;
    ID3D12StateObjectProperties* collisionQueryProps;
    ID3D12RootSignature* collisionQueryRootSig;
    void* collisionQueryRayGenID;
    void* collisionQueryMissID;
    void* collisionQueryHitGroupID;

    ID3D12PipelineState* vertexSkinning;
    ID3D12RootSignature* vertexSkinningRootSig;

    ID3D12PipelineState* postProcess;
    ID3D12RootSignature* postProcessRootSig;

    ID3D12PipelineState* shapes;
    ID3D12RootSignature* shapesRootSig;

    ID3D12PipelineState* imgui;
    ID3D12RootSignature* imguiRootSig;
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
    uint verticesBufferOffset;
    uint verticesCount;
    uint indicesBufferOffset;
    uint indicesCount;
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
    uint idleAnimationIndex;
    uint walkAnimationIndex;
    uint runAnimationIndex;
    uint jumpAnimationIndex;
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
    uint selectedObjectIndex;
    std::stack<EditorUndo> undos;
};

static bool quit = false;
static LARGE_INTEGER perfFrequency = {};
static LARGE_INTEGER perfCounters[2] = {};
static uint64 frameCount = 0;
static double frameTime = 0;
static uint mouseSelectX = UINT_MAX;
static uint mouseSelectY = UINT_MAX;
static int2 mouseDeltaRaw = {0, 0};
static float mouseWheel = 0;
static float mouseSensitivity = 0.2f;
static float controllerSensitivity = 2.0f;

static Settings settings = {};
static Window window = {};
static D3D d3d = {};
static Controller controller = {};
static float controllerDeadZone = 0.25f;
static HANDLE controllerDualSenseHID;

static std::filesystem::path worldFilePath;
static std::list<Model> models;
static ModelInstance modelInstanceCube;
static ModelInstance modelInstanceCylinder;
static ModelInstance modelInstanceSphere;
static Player player;
static std::vector<GameObject> gameObjects;
static Skybox skybox;
static std::vector<Light> lights;
static std::vector<D3D12_RAYTRACING_INSTANCE_DESC> tlasInstancesBuildInfos;
static std::vector<TLASInstanceInfo> tlasInstancesInfos;
static std::vector<BLASGeometryInfo> blasGeometriesInfos;
static std::vector<ShapeCircle> shapeCircles;
static std::vector<ShapeLine> shapeLines;

static Editor* editor = new Editor();
static ImVec2 editorMainMenuBarPos = {};
static ImVec2 editorMainMenuBarSize = {};
static ImVec2 editorObjectWindowPos = {};
static ImVec2 editorObjectWindowSize = {};
static ImVec2 editorObjectPropertiesWindowPos = {};
static ImVec2 editorObjectPropertiesWindowSize = {};
static bool editorAddObjectPopupFlag = false;

static std::filesystem::path exeDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, countof(buf));
    assert(n < countof(buf));
    std::filesystem::path path(buf);
    return path.parent_path();
}();

static std::filesystem::path assetsDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, countof(buf));
    assert(n < countof(buf));
    std::filesystem::path path(buf);
    return path.parent_path().parent_path().parent_path() / "assets";
}();

static std::filesystem::path saveDir = [] {
    wchar_t* documentFolderPathStr;
    HRESULT result = SHGetKnownFolderPath(FOLDERID_SavedGames, KF_FLAG_DEFAULT, nullptr, &documentFolderPathStr);
    assert(result == S_OK);
    std::filesystem::path documentFolderPath(documentFolderPathStr);
    CoTaskMemFree(documentFolderPathStr);
    documentFolderPath = documentFolderPath / "AGBY_GAME_SAVES";
    if (!std::filesystem::exists(documentFolderPath)) {
        bool createSuccess = std::filesystem::create_directory(documentFolderPath);
        assert(createSuccess);
    }
    return documentFolderPath;
}();

std::string getLastErrorStr() {
    DWORD err = GetLastError();
    std::string message = std::system_category().message(err);
    return message;
}

bool fileExists(const std::filesystem::path& path) {
    DWORD dwAttrib = GetFileAttributesW(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

std::string fileReadStr(const std::filesystem::path& path) {
    std::ifstream file(path);
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
    std::ofstream file(path);
    file << str;
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

void settingsInit(const std::filesystem::path& settingsPath) {
    if (fileExists(settingsPath)) {
        std::string yamlStr = fileReadStr(settingsPath);
        ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
        ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
        yamlRoot["hdr"] >> settings.hdr;
        yamlRoot["windowX"] >> settings.windowX;
        yamlRoot["windowY"] >> settings.windowY;
        yamlRoot["windowW"] >> settings.windowW;
        yamlRoot["windowH"] >> settings.windowH;
    }
}

void settingsSave(const std::filesystem::path& settingsPath) {
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;
    yamlRoot["hdr"] << settings.hdr;
    yamlRoot["windowX"] << settings.windowX;
    yamlRoot["windowY"] << settings.windowY;
    yamlRoot["windowW"] << settings.windowW;
    yamlRoot["windowH"] << settings.windowH;
    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(settingsPath, yamlStr);
}

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
    assert(SUCCEEDED(SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)));

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

    RAWINPUTDEVICE rawInputDeviceMouse = {.usUsagePage = 0x0001, .usUsage = 0x0002, .hwndTarget = window.hwnd};
    assert(RegisterRawInputDevices(&rawInputDeviceMouse, 1, sizeof(rawInputDeviceMouse)));

    RAWINPUTDEVICE rawInputDeviceController = {.usUsagePage = 0x0001, .usUsage = 0x0005, .dwFlags = RIDEV_DEVNOTIFY, .hwndTarget = window.hwnd};
    assert(RegisterRawInputDevices(&rawInputDeviceController, 1, sizeof(rawInputDeviceController)));
}

void windowShow() {
    ShowWindow(window.hwnd, SW_SHOW);
}

void windowHideCursor(bool hide) {
    if (hide) {
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
        ShowCursor(false);
    } else {
        ClipCursor(nullptr);
        ShowCursor(true);
    }
}

void imguiInit() {
    assert(ImGui::CreateContext());
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();
    // ImGui::StyleColorsClassic();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "imgui.ini";
    io.FontGlobalScale = (float)settings.renderH / 800.0f;
    assert(io.Fonts->AddFontDefault());
}

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

// void controllerGetStateDualSense(uint8* packet, uint packetSize) {
//     controller = {};
//     uint n = (packetSize == 64) ? 0 : 1;
//     controller.lsX = (packet[n + 1] / 255.0f) * 2.0f - 1.0f;
//     controller.lsY = -((packet[n + 2] / 255.0f) * 2.0f - 1.0f);
//     controller.rsX = (packet[n + 3] / 255.0f) * 2.0f - 1.0f;
//     controller.rsY = -((packet[n + 4] / 255.0f) * 2.0f - 1.0f);
//     controller.lt = packet[n + 5] / 255.0f;
//     controller.rt = packet[n + 6] / 255.0f;
//     switch (packet[n + 8] & 0x0f) {
//     case 0x0: controller.up = true; break;
//     case 0x1: controller.up = controller.right = true; break;
//     case 0x2: controller.right = true; break;
//     case 0x3: controller.down = controller.right = true; break;
//     case 0x4: controller.down = true; break;
//     case 0x5: controller.down = controller.left = true; break;
//     case 0x6: controller.left = true; break;
//     case 0x7: controller.up = controller.left = true; break;
//     }
//     controller.x = packet[n + 8] & 0x10;
//     controller.a = packet[n + 8] & 0x20;
//     controller.b = packet[n + 8] & 0x40;
//     controller.y = packet[n + 8] & 0x80;
//     controller.lb = packet[n + 9] & 0x01;
//     controller.rb = packet[n + 9] & 0x02;
//     controller.back = packet[n + 9] & 0x10;
//     controller.start = packet[n + 9] & 0x20;
//     controller.ls = packet[n + 9] & 0x40;
//     controller.rs = packet[n + 9] & 0x80;
//     controllerApplyDeadZone();
// }

void controllerGetStateXInput() {
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
        controllerApplyDeadZone(&c);
        if (controller.back && c.back) c.backDownDuration = controller.backDownDuration + (float)frameTime;
        if (controller.start && c.start) c.startDownDuration = controller.startDownDuration + (float)frameTime;
        if (controller.a && c.a) c.aDownDuration = controller.aDownDuration + (float)frameTime;
        if (controller.b && c.b) c.bDownDuration = controller.bDownDuration + (float)frameTime;
        if (controller.x && c.x) c.xDownDuration = controller.xDownDuration + (float)frameTime;
        if (controller.y && c.y) c.yDownDuration = controller.yDownDuration + (float)frameTime;
        if (controller.up && c.up) c.upDownDuration = controller.upDownDuration + (float)frameTime;
        if (controller.down && c.down) c.downDownDuration = controller.downDownDuration + (float)frameTime;
        if (controller.left && c.left) c.leftDownDuration = controller.leftDownDuration + (float)frameTime;
        if (controller.right && c.right) c.rightDownDuration = controller.rightDownDuration + (float)frameTime;
        if (controller.lb && c.lb) c.lbDownDuration = controller.lbDownDuration + (float)frameTime;
        if (controller.rb && c.rb) c.rbDownDuration = controller.rbDownDuration + (float)frameTime;
        if (controller.ls && c.ls) c.lsDownDuration = controller.lsDownDuration + (float)frameTime;
        if (controller.rs && c.rs) c.rsDownDuration = controller.rsDownDuration + (float)frameTime;
    }
    controller = c;
}

void d3dMessageCallback(D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context) {
    if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR) {
        __debugbreak();
    }
}

void d3dTransferQueueStartRecording() {
    assert(SUCCEEDED(d3d.transferCmdAllocator->Reset()));
    assert(SUCCEEDED(d3d.transferCmdList->Reset(d3d.transferCmdAllocator, nullptr)));
}

void d3dTransferQueueSubmitRecording() {
    assert(SUCCEEDED(d3d.transferCmdList->Close()));
    d3d.transferQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.transferCmdList);
}

void d3dTransferQueueWait() {
    d3d.transferDoneFenceCounter += 1;
    d3d.transferQueue->Signal(d3d.transferDoneFence, d3d.transferDoneFenceCounter);
    if (d3d.transferDoneFence->GetCompletedValue() < d3d.transferDoneFenceCounter) {
        assert(SUCCEEDED(d3d.transferDoneFence->SetEventOnCompletion(d3d.transferDoneFenceCounter, d3d.transferDoneFenceEvent)));
        assert(WaitForSingleObjectEx(d3d.transferDoneFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
    }
}

D3D12MA::Allocation* d3dCreateImage(const D3D12_RESOURCE_DESC& resourceDesc, D3D12_SUBRESOURCE_DATA* imageMips, const wchar_t* name = nullptr) {
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr)));
    if (name) { image->GetResource()->SetName(name); }
    d3d.stagingBufferOffset = align(d3d.stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
    uint rowCounts[16];
    uint64 rowSizes[16];
    uint64 requiredSize;
    d3d.device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
    assert(d3d.stagingBufferOffset + requiredSize < d3d.stagingBuffer->GetSize());
    for (uint mipIndex = 0; mipIndex < resourceDesc.MipLevels; mipIndex++) {
        mipFootprints[mipIndex].Offset += d3d.stagingBufferOffset;
    }
    assert(UpdateSubresources(d3d.transferCmdList, image->GetResource(), d3d.stagingBuffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, imageMips) == requiredSize);
    d3d.stagingBufferOffset += (uint)requiredSize;
    return image;
}

D3D12MA::Allocation* d3dCreateImageSTB(const std::filesystem::path& ddsFilePath, const wchar_t* name = nullptr) {
    int width, height, channelCount;
    unsigned char* imageData = stbi_load(ddsFilePath.string().c_str(), &width, &height, &channelCount, 4);
    assert(imageData);
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint)width, .Height = (uint)height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, .SampleDesc = {.Count = 1}};
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr)));
    if (name) { image->GetResource()->SetName(name); }
    d3d.stagingBufferOffset = align(d3d.stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprint;
    uint rowCount;
    uint64 rowSize;
    uint64 requiredSize;
    d3d.device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, &mipFootprint, &rowCount, &rowSize, &requiredSize);
    assert(d3d.stagingBufferOffset + requiredSize < d3d.stagingBuffer->GetSize());
    mipFootprint.Offset += d3d.stagingBufferOffset;
    D3D12_SUBRESOURCE_DATA srcData = {.pData = imageData, .RowPitch = width * 4, .SlicePitch = width * height * 4};
    assert(UpdateSubresources(d3d.transferCmdList, image->GetResource(), d3d.stagingBuffer->GetResource(), 0, 1, requiredSize, &mipFootprint, &rowCount, &rowSize, &srcData) == requiredSize);
    d3d.stagingBufferOffset += (uint)requiredSize;
    stbi_image_free(imageData);
    return image;
}

D3D12MA::Allocation* d3dCreateImageDDS(const std::filesystem::path& ddsFilePath, const wchar_t* name = nullptr) {
    ScratchImage scratchImage;
    assert(SUCCEEDED(LoadFromDDSFile(ddsFilePath.c_str(), DDS_FLAGS_NONE, nullptr, scratchImage)));
    assert(scratchImage.GetImageCount() == scratchImage.GetMetadata().mipLevels);
    const TexMetadata& scratchImageInfo = scratchImage.GetMetadata();
    DXGI_FORMAT format = scratchImageInfo.format;
    if (format == DXGI_FORMAT_BC7_UNORM) format = DXGI_FORMAT_BC7_UNORM_SRGB;
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC resourceDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint)scratchImageInfo.width, .Height = (uint)scratchImageInfo.height, .DepthOrArraySize = (uint16)scratchImageInfo.arraySize, .MipLevels = (uint16)scratchImageInfo.mipLevels, .Format = format, .SampleDesc = {.Count = 1}};
    D3D12MA::Allocation* image;
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &image, {}, nullptr)));
    if (name) { image->GetResource()->SetName(name); }
    d3d.stagingBufferOffset = align(d3d.stagingBufferOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mipFootprints[16];
    uint rowCounts[16];
    uint64 rowSizes[16];
    uint64 requiredSize;
    D3D12_SUBRESOURCE_DATA srcData[16];
    d3d.device->GetCopyableFootprints(&resourceDesc, 0, resourceDesc.MipLevels, 0, mipFootprints, rowCounts, rowSizes, &requiredSize);
    assert(d3d.stagingBufferOffset + requiredSize < d3d.stagingBuffer->GetSize());
    for (uint mipIndex = 0; mipIndex < scratchImageInfo.mipLevels; mipIndex++) {
        mipFootprints[mipIndex].Offset += d3d.stagingBufferOffset;
        const Image& image = scratchImage.GetImages()[mipIndex];
        srcData[mipIndex] = {.pData = image.pixels, .RowPitch = (int64)image.rowPitch, .SlicePitch = (int64)image.slicePitch};
    }
    assert(UpdateSubresources(d3d.transferCmdList, image->GetResource(), d3d.stagingBuffer->GetResource(), 0, resourceDesc.MipLevels, requiredSize, mipFootprints, rowCounts, rowSizes, srcData) == requiredSize);
    d3d.stagingBufferOffset += (uint)requiredSize;
    return image;
}

void d3dWaitForRender() {
    if (d3d.renderDoneFence->GetCompletedValue() < d3d.renderDoneFenceValue) {
        assert(SUCCEEDED(d3d.renderDoneFence->SetEventOnCompletion(d3d.renderDoneFenceValue, d3d.renderDoneFenceEvent)));
        assert(WaitForSingleObjectEx(d3d.renderDoneFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
    }
}

void d3dWaitForCollisionQueries() {
    if (d3d.collisionQueriesFence->GetCompletedValue() < d3d.collisionQueriesFenceValue) {
        assert(SUCCEEDED(d3d.collisionQueriesFence->SetEventOnCompletion(d3d.collisionQueriesFenceValue, d3d.collisionQueriesFenceEvent)));
        assert(WaitForSingleObjectEx(d3d.collisionQueriesFenceEvent, INFINITE, false) == WAIT_OBJECT_0);
    }
}

void d3dRayTracingValidationCallBack(void* pUserData, NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY severity, const char* messageCode, const char* message, const char* messageDetails) {
    const char* severityString = "unknown";
    switch (severity) {
    case NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY_ERROR: severityString = "error"; break;
    case NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY_WARNING: severityString = "warning"; break;
    }
    fprintf(stderr, "Ray Tracing Validation message: %s: [%s] %s\n%s", severityString, messageCode, message, messageDetails);
    fflush(stderr);
}

void d3dInit() {
    bool debug = commandLineContain(L"d3ddebug");
    uint factoryFlags = 0;
    if (debug) {
        factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
        ID3D12Debug1* debug;
        assert(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))));
        debug->EnableDebugLayer();
        // debug->SetEnableGPUBasedValidation(true);
        // debug->SetEnableSynchronizedCommandQueueValidation(true);
    }

    DXGI_ADAPTER_DESC dxgiAdapterDesc = {};
    DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};
    assert(SUCCEEDED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&d3d.dxgiFactory))));
    assert(SUCCEEDED(d3d.dxgiFactory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&d3d.dxgiAdapter))));
    assert(SUCCEEDED(D3D12CreateDevice(d3d.dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d.device))));
    if (debug) {
        ID3D12InfoQueue1* infoQueue;
        assert(SUCCEEDED(d3d.device->QueryInterface(IID_PPV_ARGS(&infoQueue))));
        DWORD callbackCookie;
        assert(SUCCEEDED(infoQueue->RegisterMessageCallback(d3dMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &callbackCookie)));
    }
    assert(SUCCEEDED(d3d.dxgiAdapter->GetDesc(&dxgiAdapterDesc)));
    assert(SUCCEEDED(d3d.dxgiAdapter->EnumOutputs(0, (IDXGIOutput**)&d3d.dxgiOutput)));
    assert(SUCCEEDED(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc)));
    settings.hdr = (dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS resourceBindingTier = {};
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 rayTracing = {};
        // D3D12_FEATURE_DATA_D3D12_OPTIONS16 gpuUploadHeap = {};
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &resourceBindingTier, sizeof(resourceBindingTier))));
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))));
        assert(SUCCEEDED(d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &rayTracing, sizeof(rayTracing))));
        // assert(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &gpuUploadHeap, sizeof(gpuUploadHeap))));
        assert(resourceBindingTier.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        assert(shaderModel.HighestShaderModel == D3D_SHADER_MODEL_6_6);
        assert(rayTracing.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
        // gpuUploadHeapSupported = gpuUploadHeap.GPUUploadHeapSupported;
    }
    {
        NvAPI_Status status = NvAPI_Initialize();
        if (status == NVAPI_OK) {
            status = NvAPI_D3D12_EnableRaytracingValidation(d3d.device, NVAPI_D3D12_RAYTRACING_VALIDATION_FLAG_NONE);
            if (status == NVAPI_OK) {
                void* unregisterHandle;
                status = NvAPI_D3D12_RegisterRaytracingValidationMessageCallback(d3d.device, d3dRayTracingValidationCallBack, nullptr, &unregisterHandle);
            }
        }
    }
    {
        D3D12_COMMAND_QUEUE_DESC graphicsQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
        assert(SUCCEEDED(d3d.device->CreateCommandQueue(&graphicsQueueDesc, IID_PPV_ARGS(&d3d.graphicsQueue))));
        D3D12_COMMAND_QUEUE_DESC transferQueueDesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT, .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE};
        assert(SUCCEEDED(d3d.device->CreateCommandQueue(&transferQueueDesc, IID_PPV_ARGS(&d3d.transferQueue))));

        assert(SUCCEEDED(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.graphicsCmdAllocator))));
        assert(SUCCEEDED(d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.transferCmdAllocator))));

        assert(SUCCEEDED(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.graphicsCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.graphicsCmdList))));
        assert(SUCCEEDED(d3d.graphicsCmdList->Close()));
        assert(SUCCEEDED(d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.transferCmdAllocator, nullptr, IID_PPV_ARGS(&d3d.transferCmdList))));
        assert(SUCCEEDED(d3d.transferCmdList->Close()));

        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.renderDoneFence))));
        d3d.renderDoneFenceEvent = CreateEventA(nullptr, false, false, nullptr);
        assert(d3d.renderDoneFenceEvent);

        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.collisionQueriesFence))));
        d3d.collisionQueriesFenceEvent = CreateEventA(nullptr, false, false, nullptr);
        assert(d3d.collisionQueriesFenceEvent);

        assert(SUCCEEDED(d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.transferDoneFence))));
        d3d.transferDoneFenceEvent = CreateEventA(nullptr, false, false, nullptr);
        assert(d3d.transferDoneFenceEvent);
    }
    {
        d3d.swapChainFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
        uint dxgiModeCount = 0;
        d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, nullptr);
        std::vector<DXGI_MODE_DESC> dxgiModes(dxgiModeCount);
        d3d.dxgiOutput->GetDisplayModeList(d3d.swapChainFormat, 0, &dxgiModeCount, dxgiModes.data());
        for (DXGI_MODE_DESC& dxgiMode : dxgiModes) {
            bool hasResolution = false;
            for (DisplayMode& displayMode : d3d.displayModes) {
                if (displayMode.resolutionW == dxgiMode.Width && displayMode.resolutionH == dxgiMode.Height) {
                    hasResolution = true;
                    bool hasRefreshRate = false;
                    for (DXGI_RATIONAL& refreshRate : displayMode.refreshRates) {
                        if (refreshRate.Numerator == dxgiMode.RefreshRate.Numerator && refreshRate.Denominator == dxgiMode.RefreshRate.Denominator) {
                            hasRefreshRate = true;
                            break;
                        }
                    }
                    if (!hasRefreshRate) {
                        displayMode.refreshRates.push_back(dxgiMode.RefreshRate);
                    }
                    break;
                }
            }
            if (!hasResolution) d3d.displayModes.push_back(DisplayMode{dxgiMode.Width, dxgiMode.Height, {dxgiMode.RefreshRate}});
        }
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
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
        assert(SUCCEEDED(d3d.dxgiFactory->CreateSwapChainForHwnd(d3d.graphicsQueue, window.hwnd, &swapChainDesc, nullptr, nullptr, (IDXGISwapChain1**)&d3d.swapChain)));

        DXGI_COLOR_SPACE_TYPE colorSpace = settings.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(colorSpace)));
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
            assert(SUCCEEDED(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image))));
            (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
        }
        d3d.dxgiFactory->MakeWindowAssociation(window.hwnd, DXGI_MWA_NO_WINDOW_CHANGES); // disable alt-enter
    }
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 16, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        D3D12_DESCRIPTOR_HEAP_DESC uavDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = 16, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        assert(SUCCEEDED(d3d.device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&d3d.rtvDescriptorHeap))));
        d3d.rtvDescriptorCount = 0;
        for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
            ID3D12Resource* image = d3d.swapChainImages[imageIndex];
            d3d.swapChainImageRTVDescriptors[imageIndex] = {d3d.rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) * d3d.rtvDescriptorCount};
            d3d.device->CreateRenderTargetView(image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
            d3d.rtvDescriptorCount += 1;
        }

        d3d.cbvSrvUavDescriptorCapacity = 1024;
        d3d.cbvSrvUavDescriptorSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavDescriptorHeapDesc = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = d3d.cbvSrvUavDescriptorCapacity, .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        assert(SUCCEEDED(d3d.device->CreateDescriptorHeap(&cbvSrvUavDescriptorHeapDesc, IID_PPV_ARGS(&d3d.cbvSrvUavDescriptorHeap))));
    }
    {
        D3D12MA::ALLOCATOR_DESC allocatorDesc = {.Flags = D3D12MA::ALLOCATOR_FLAG_NONE, .pDevice = d3d.device, .pAdapter = d3d.dxgiAdapter};
        assert(SUCCEEDED(D3D12MA::CreateAllocator(&allocatorDesc, &d3d.allocator)));
    }
    {
        struct BufferDesc {
            D3D12MA::Allocation** buffer;
            void** bufferPtr;
            uint size;
            D3D12_HEAP_TYPE heapType;
            D3D12_RESOURCE_FLAGS flags;
            D3D12_RESOURCE_STATES initState;
            const wchar_t* name;
        } descs[] = {
            {&d3d.stagingBuffer, (void**)&d3d.stagingBufferPtr, MEGABYTES(512), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"stagingBuffer"},
            {&d3d.constantsBuffer, (void**)&d3d.constantsBufferPtr, MEGABYTES(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ, L"constantBuffer"},
            {&d3d.tlasInstancesBuildInfosBuffer, (void**)&d3d.tlasInstancesBuildInfosBufferPtr, MEGABYTES(32), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesBuildInfosBuffer"},
            {&d3d.tlasInstancesInfosBuffer, (void**)&d3d.tlasInstancesInfosBufferPtr, MEGABYTES(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"tlasInstancesInfosBuffer"},
            {&d3d.blasGeometriesInfosBuffer, (void**)&d3d.blasGeometriesInfosBufferPtr, MEGABYTES(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"blasGeometriesInfosBuffer"},
            {&d3d.tlasBuffer, nullptr, MEGABYTES(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"tlasBuffer"},
            {&d3d.tlasScratchBuffer, nullptr, MEGABYTES(32), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"tlasScratchBuffer"},
            {&d3d.shapeCirclesBuffer, (void**)&d3d.shapeCirclesBufferPtr, MEGABYTES(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, L"shapeCirclesBuffer"},
            {&d3d.shapeLinesBuffer, (void**)&d3d.shapeLinesBufferPtr, MEGABYTES(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, L"shapeLinesBuffer"},
            {&d3d.imguiVertexBuffer, (void**)&d3d.imguiVertexBufferPtr, MEGABYTES(2), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiVertexBuffer"},
            {&d3d.imguiIndexBuffer, (void**)&d3d.imguiIndexBufferPtr, MEGABYTES(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_GENERIC_READ, L"imguiIndexBuffer"},
            {&d3d.collisionQueriesBuffer, (void**)&d3d.collisionQueriesBufferPtr, MEGABYTES(1), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, L"collisionQueriesBuffer"},
            {&d3d.collisionQueryResultsUAVBuffer, nullptr, MEGABYTES(1), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, L"collisionQueryResultsUAVBuffer"},
            {&d3d.collisionQueryResultsBuffer, (void**)&d3d.collisionQueryResultsBufferPtr, MEGABYTES(1), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, L"collisionQueryResultsBuffer"},
        };
        for (BufferDesc& desc : descs) {
            D3D12_RESOURCE_DESC bufferDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = desc.size, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = desc.flags};
            D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = desc.heapType};
            assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &bufferDesc, desc.initState, nullptr, desc.buffer, {}, nullptr)));
            (*desc.buffer)->GetResource()->SetName(desc.name);
            if (desc.bufferPtr) {
                assert(SUCCEEDED((*desc.buffer)->GetResource()->Map(0, nullptr, desc.bufferPtr)));
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
            .Format = DXGI_FORMAT_R16G16B16A16_FLOAT,
            .SampleDesc = {.Count = 1},
            .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
            .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        };

        D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr)));
        assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexturePrevFrame, {}, nullptr)));
        d3d.renderTexture->GetResource()->SetName(L"renderTexture");
        d3d.renderTexturePrevFrame->GetResource()->SetName(L"renderTexturePrevFrame");
        d3d.renderTextureFormat = renderTextureDesc.Format;
    }
    {
        d3dTransferQueueStartRecording();
        d3d.stagingBufferOffset = 0;
        {
            uint8* imguiTextureData;
            int imguiTextureWidth, imguiTextureHeight;
            ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&imguiTextureData, &imguiTextureWidth, &imguiTextureHeight);
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = (uint64)imguiTextureWidth, .Height = (uint)imguiTextureHeight, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = imguiTextureData, .RowPitch = imguiTextureWidth * 4, .SlicePitch = imguiTextureWidth * imguiTextureHeight * 4};
            d3d.imguiImage = d3dCreateImage(desc, &data);
        }
        {
            uint8_4 defaultMaterialBaseColorImageData[4] = {{255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}};
            D3D12_RESOURCE_DESC desc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 2, .Height = 2, .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1}};
            D3D12_SUBRESOURCE_DATA data = {.pData = defaultMaterialBaseColorImageData, .RowPitch = 8, .SlicePitch = 16};
            d3d.defaultMaterialBaseColorImage = d3dCreateImage(desc, &data);
            d3d.defaultMaterialBaseColorImageSRVDesc = {.Format = desc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = desc.MipLevels}};
        }
        D3D12_RESOURCE_BARRIER barriers[2] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.imguiImage->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.defaultMaterialBaseColorImage->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}},
        };
        d3d.transferCmdList->ResourceBarrier(2, barriers);
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }
}

void d3dUpdateShaders() {
    {
        static std::filesystem::path shaderPath = exeDir / "renderScene.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitForRender();
            if (d3d.renderScene) d3d.renderScene->Release();
            if (d3d.renderSceneProps) d3d.renderSceneProps->Release();
            if (d3d.renderSceneRootSig) d3d.renderSceneRootSig->Release();
            std::vector<uint8> rtByteCode = fileReadBytes(shaderPath);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"primaryRayMiss"}, {L"primaryRayHitGroup"}, {L"primaryRayClosestHit"}, {L"secondaryRayMiss"}, {L"secondaryRayHitGroup"}, {L"secondaryRayClosestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(SUCCEEDED(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.renderScene))));
            assert(SUCCEEDED(d3d.renderScene->QueryInterface(IID_PPV_ARGS(&d3d.renderSceneProps))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, rtByteCode.data(), rtByteCode.size(), IID_PPV_ARGS(&d3d.renderSceneRootSig))));
            assert(d3d.renderSceneRayGenID = d3d.renderSceneProps->GetShaderIdentifier(L"rayGen"));
            assert(d3d.renderScenePrimaryRayMissID = d3d.renderSceneProps->GetShaderIdentifier(L"primaryRayMiss"));
            assert(d3d.renderScenePrimaryRayHitGroupID = d3d.renderSceneProps->GetShaderIdentifier(L"primaryRayHitGroup"));
            assert(d3d.renderSceneSecondaryRayMissID = d3d.renderSceneProps->GetShaderIdentifier(L"secondaryRayMiss"));
            assert(d3d.renderSceneSecondaryRayHitGroupID = d3d.renderSceneProps->GetShaderIdentifier(L"secondaryRayHitGroup"));
        }
    }
    {
        static std::filesystem::path shaderPath = exeDir / "collisionQuery.cso";
        static std::filesystem::file_time_type prevLastWriteTime = {};
        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(shaderPath);
        if (lastWriteTime > prevLastWriteTime) {
            prevLastWriteTime = lastWriteTime;
            d3dWaitForRender();
            if (d3d.collisionQuery) d3d.collisionQuery->Release();
            if (d3d.collisionQueryProps) d3d.collisionQueryProps->Release();
            if (d3d.collisionQueryRootSig) d3d.collisionQueryRootSig->Release();
            std::vector<uint8> rtByteCode = fileReadBytes(shaderPath);
            D3D12_EXPORT_DESC exportDescs[] = {{L"globalRootSig"}, {L"pipelineConfig"}, {L"shaderConfig"}, {L"rayGen"}, {L"miss"}, {L"hitGroup"}, {L"closestHit"}};
            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {.DXILLibrary = {.pShaderBytecode = rtByteCode.data(), .BytecodeLength = rtByteCode.size()}, .NumExports = countof(exportDescs), .pExports = exportDescs};
            D3D12_STATE_SUBOBJECT stateSubobjects[] = {{.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, .pDesc = &dxilLibDesc}};
            D3D12_STATE_OBJECT_DESC stateObjectDesc = {.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, .NumSubobjects = countof(stateSubobjects), .pSubobjects = stateSubobjects};
            assert(SUCCEEDED(d3d.device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&d3d.collisionQuery))));
            assert(SUCCEEDED(d3d.collisionQuery->QueryInterface(IID_PPV_ARGS(&d3d.collisionQueryProps))));
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
            d3dWaitForRender();
            if (d3d.vertexSkinning) d3d.vertexSkinning->Release();
            if (d3d.vertexSkinningRootSig) d3d.vertexSkinningRootSig->Release();
            std::vector<uint8> csByteCode = fileReadBytes(shaderPath);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {.pRootSignature = d3d.vertexSkinningRootSig, .CS = {.pShaderBytecode = csByteCode.data(), .BytecodeLength = csByteCode.size()}};
            assert(SUCCEEDED(d3d.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&d3d.vertexSkinning))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, csByteCode.data(), csByteCode.size(), IID_PPV_ARGS(&d3d.vertexSkinningRootSig))));
        }
    }
    {
        static std::filesystem::path shaderPathVS = exeDir / "postProcessVS.cso";
        static std::filesystem::path shaderPathPS = exeDir / "postProcessPS.cso";
        static std::filesystem::file_time_type prevLastWriteTimeVS = {};
        static std::filesystem::file_time_type prevLastWriteTimePS = {};
        std::filesystem::file_time_type lastWriteTimeVS = std::filesystem::last_write_time(shaderPathVS);
        std::filesystem::file_time_type lastWriteTimePS = std::filesystem::last_write_time(shaderPathPS);
        if (lastWriteTimeVS > prevLastWriteTimeVS || lastWriteTimePS > prevLastWriteTimePS) {
            prevLastWriteTimeVS = lastWriteTimeVS;
            prevLastWriteTimePS = lastWriteTimePS;
            d3dWaitForRender();
            if (d3d.postProcess) d3d.postProcess->Release();
            if (d3d.postProcessRootSig) d3d.postProcessRootSig->Release();
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
            assert(SUCCEEDED(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.postProcess))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, psByteCode.data(), psByteCode.size(), IID_PPV_ARGS(&d3d.postProcessRootSig))));
        }
    }
    {
        static std::filesystem::path shaderPathVS = exeDir / "shapesVS.cso";
        static std::filesystem::path shaderPathPS = exeDir / "shapesPS.cso";
        static std::filesystem::file_time_type prevLastWriteTimeVS = {};
        static std::filesystem::file_time_type prevLastWriteTimePS = {};
        std::filesystem::file_time_type lastWriteTimeVS = std::filesystem::last_write_time(shaderPathVS);
        std::filesystem::file_time_type lastWriteTimePS = std::filesystem::last_write_time(shaderPathPS);
        if (lastWriteTimeVS > prevLastWriteTimeVS || lastWriteTimePS > prevLastWriteTimePS) {
            prevLastWriteTimeVS = lastWriteTimeVS;
            prevLastWriteTimePS = lastWriteTimePS;
            d3dWaitForRender();
            if (d3d.shapes) d3d.shapes->Release();
            if (d3d.shapesRootSig) d3d.shapesRootSig->Release();
            std::vector<uint8> vsByteCode = fileReadBytes(shaderPathVS);
            std::vector<uint8> psByteCode = fileReadBytes(shaderPathPS);
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
                .VS = {vsByteCode.data(), vsByteCode.size()},
                .PS = {psByteCode.data(), psByteCode.size()},
                .BlendState = {
                    .RenderTarget = {
                        {
                            .BlendEnable = true,
                            .SrcBlend = D3D12_BLEND_SRC_ALPHA,
                            .DestBlend = D3D12_BLEND_INV_SRC_ALPHA,
                            .BlendOp = D3D12_BLEND_OP_ADD,
                            .SrcBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA,
                            .DestBlendAlpha = D3D12_BLEND_ZERO,
                            .BlendOpAlpha = D3D12_BLEND_OP_ADD,
                            .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL,
                        },
                    },
                },
                .SampleMask = 0xffffffff,
                .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_BACK},
                .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                .NumRenderTargets = 1,
                .RTVFormats = {d3d.swapChainFormat},
                .SampleDesc = {.Count = 1},
            };
            assert(SUCCEEDED(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.shapes))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, psByteCode.data(), psByteCode.size(), IID_PPV_ARGS(&d3d.shapesRootSig))));
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
            d3dWaitForRender();
            if (d3d.imgui) d3d.imgui->Release();
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
            assert(SUCCEEDED(d3d.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&d3d.imgui))));
            assert(SUCCEEDED(d3d.device->CreateRootSignature(0, vsByteCode.data(), vsByteCode.size(), IID_PPV_ARGS(&d3d.imguiRootSig))));
        }
    }
}

void d3dResizeSwapChain(uint width, uint height) {
    d3dWaitForRender();
    for (ID3D12Resource* image : d3d.swapChainImages) { image->Release(); }
    assert(SUCCEEDED(d3d.swapChain->ResizeBuffers(countof(d3d.swapChainImages), width, height, d3d.swapChainFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH)));
    for (uint imageIndex = 0; imageIndex < countof(d3d.swapChainImages); imageIndex++) {
        ID3D12Resource** image = &d3d.swapChainImages[imageIndex];
        assert(SUCCEEDED(d3d.swapChain->GetBuffer(imageIndex, IID_PPV_ARGS(image))));
        (*image)->SetName(std::format(L"swapChain{}", imageIndex).c_str());
        d3d.device->CreateRenderTargetView(*image, nullptr, d3d.swapChainImageRTVDescriptors[imageIndex]);
    }
    d3d.renderTexture->Release();
    d3d.renderTexturePrevFrame->Release();
    D3D12_RESOURCE_DESC renderTextureDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = width, .Height = height, .DepthOrArraySize = 1, .MipLevels = 1, .Format = d3d.renderTextureFormat, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexture, {}, nullptr)));
    assert(SUCCEEDED(d3d.allocator->CreateResource(&allocationDesc, &renderTextureDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, &d3d.renderTexturePrevFrame, {}, nullptr)));
    d3d.renderTexture->GetResource()->SetName(L"renderTexture");
    d3d.renderTexturePrevFrame->GetResource()->SetName(L"renderTexturePrevFrame");
}

void d3dApplySettings() {
    DXGI_OUTPUT_DESC1 dxgiOutputDesc = {};
    assert(SUCCEEDED(d3d.dxgiOutput->GetDesc1(&dxgiOutputDesc)));
    if (settings.hdr && dxgiOutputDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)));
    } else {
        assert(SUCCEEDED(d3d.swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)));
    }
    if (settings.windowMode == WindowModeWindowed) {
        assert(SUCCEEDED(d3d.swapChain->SetFullscreenState(false, nullptr)));
        DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
        MONITORINFO mi = {.cbSize = sizeof(mi)};
        assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
        assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW) != 0);
        assert(SetWindowPos(window.hwnd, NULL, settings.windowX, settings.windowY, settings.windowW, settings.windowH, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
    } else if (settings.windowMode == WindowModeBorderless) {
        assert(SUCCEEDED(d3d.swapChain->SetFullscreenState(false, nullptr)));
        DWORD dwStyle = GetWindowLong(window.hwnd, GWL_STYLE);
        MONITORINFO mi = {.cbSize = sizeof(mi)};
        assert(GetMonitorInfo(MonitorFromWindow(window.hwnd, MONITOR_DEFAULTTOPRIMARY), &mi));
        assert(SetWindowLong(window.hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW) != 0);
        assert(SetWindowPos(window.hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED));
    } else if (settings.windowMode == WindowModeFullscreen) {
        DXGI_MODE_DESC dxgiMode = {.Width = settings.windowW, .Height = settings.windowH, .RefreshRate = settings.refreshRate, .Format = d3d.swapChainFormat};
        assert(SUCCEEDED(d3d.swapChain->ResizeTarget(&dxgiMode)));
        assert(SUCCEEDED(d3d.swapChain->SetFullscreenState(true, nullptr)));
    }
}

D3DDescriptor d3dAppendCBVDescriptor(D3D12_CONSTANT_BUFFER_VIEW_DESC* constantBufferViewDesc) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateConstantBufferView(constantBufferViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendSRVDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC* resourceViewDesc, ID3D12Resource* resource) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateShaderResourceView(resource, resourceViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

D3DDescriptor d3dAppendUAVDescriptor(D3D12_UNORDERED_ACCESS_VIEW_DESC* unorderedAccessViewDesc, ID3D12Resource* resource) {
    assert(d3d.cbvSrvUavDescriptorCount < d3d.cbvSrvUavDescriptorCapacity);
    uint offset = d3d.cbvSrvUavDescriptorSize * d3d.cbvSrvUavDescriptorCount;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + offset};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = {d3d.cbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + offset};
    d3d.device->CreateUnorderedAccessView(resource, nullptr, unorderedAccessViewDesc, cpuHandle);
    d3d.cbvSrvUavDescriptorCount++;
    return {cpuHandle, gpuHandle};
}

void modelTraverseNodesImGui(ModelNode* node) {
    if (ImGui::TreeNode(node->name.c_str())) {
        for (ModelNode* childNode : node->children) {
            modelTraverseNodesImGui(childNode);
        }
        ImGui::TreePop();
    }
}

void modelTraverseNodesAndGetGlobalTransforms(Model* model, ModelNode* node, const XMMATRIX& parentMat, const std::vector<Transform>& nodeLocalTransforms, std::vector<XMMATRIX>& nodeGlobalTransformMats) {
    int64 nodeIndex = node - &model->nodes[0];
    XMMATRIX mat = XMMatrixMultiply(nodeLocalTransforms[nodeIndex].toMat(), parentMat);
    nodeGlobalTransformMats[nodeIndex] = mat;
    for (ModelNode* childNode : node->children) {
        modelTraverseNodesAndGetGlobalTransforms(model, childNode, mat, nodeLocalTransforms, nodeGlobalTransformMats);
    }
}

void modelTraverseNodesAndGetPositionAsCircles(Model* model, ModelNode* node, std::vector<XMMATRIX>& nodeGlobalTransformMats, std::vector<ShapeCircle>& circles) {
    int nodeIndex = node - &model->nodes[0];
    XMMATRIX& mat = nodeGlobalTransformMats[nodeIndex];
    XMVECTOR center = XMVector3Transform(XMVectorSet(0, 0, 0, 1), mat);
    circles.push_back(ShapeCircle{.center = float3(center), .radius = 0.01});
    for (ModelNode* childNode : node->children) {
        modelTraverseNodesAndGetPositionAsCircles(model, childNode, nodeGlobalTransformMats, circles);
    }
}

ModelInstance modelInstanceInit(const std::filesystem::path& filePath) {
    assert(filePath.extension() == ".gltf");
    Model* model = nullptr;
    for (Model& m : models) {
        if (m.filePath == filePath) model = &m;
    }
    if (!model) {
        model = &models.emplace_back();

        const std::filesystem::path gltfFilePath = assetsDir / filePath;
        const std::filesystem::path gltfFileFolderPath = gltfFilePath.parent_path();
        cgltf_options gltfOptions = {};
        cgltf_data* gltfData = nullptr;
        cgltf_result gltfParseFileResult = cgltf_parse_file(&gltfOptions, gltfFilePath.string().c_str(), &gltfData);
        assert(gltfParseFileResult == cgltf_result_success);
        cgltf_result gltfLoadBuffersResult = cgltf_load_buffers(&gltfOptions, gltfData, gltfFilePath.string().c_str());
        assert(gltfLoadBuffersResult == cgltf_result_success);
        assert(gltfData->scenes_count == 1);
        model->filePath = filePath;
        model->gltfData = gltfData;

        d3dTransferQueueStartRecording();
        d3d.stagingBufferOffset = 0;

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
            } else {
                node.parent = nullptr;
            }
            for (cgltf_node* child : std::span(gltfNode.children, gltfNode.children_count)) {
                uint childNodeIndex = (uint)(child - gltfData->nodes);
                assert(childNodeIndex >= 0 && childNodeIndex < gltfData->nodes_count);
                node.children.push_back(&model->nodes[childNodeIndex]);
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
                assert(meshIndex >= 0 && meshIndex < gltfData->meshes_count);
                node.mesh = &model->meshes[meshIndex];
            } else {
                node.mesh = nullptr;
            }
            if (gltfNode.skin) {
                uint skinIndex = (uint)(gltfNode.skin - gltfData->skins);
                assert(skinIndex >= 0 && skinIndex < gltfData->skins_count);
                node.skin = &model->skins[skinIndex];
            } else {
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
                if (indices->component_type == cgltf_component_type_r_16u) std::copy_n((uint16*)indicesBuffer, indices->count, std::back_inserter(mesh.indices));
                else if (indices->component_type == cgltf_component_type_r_32u) std::copy_n((uint*)indicesBuffer, indices->count, std::back_inserter(mesh.indices));
                if (gltfPrimitive.material) primitive.material = &model->materials[gltfPrimitive.material - gltfData->materials];
            }

            D3D12MA::ALLOCATION_DESC verticeIndicesBuffersAllocationDesc = {.HeapType = D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC verticeIndicesBuffersDesc = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.vertices);
            assert(SUCCEEDED(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.verticesBuffer, {}, nullptr)));
            mesh.verticesBuffer->GetResource()->SetName(std::format(L"{}Mesh{}VerticesBuffer", filePath.stem().wstring(), meshIndex).c_str());
            verticeIndicesBuffersDesc.Width = vectorSizeof(mesh.indices);
            assert(SUCCEEDED(d3d.allocator->CreateResource(&verticeIndicesBuffersAllocationDesc, &verticeIndicesBuffersDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &mesh.indicesBuffer, {}, nullptr)));
            mesh.verticesBuffer->GetResource()->SetName(std::format(L"{}Mesh{}IndicesBuffer", filePath.stem().wstring(), meshIndex).c_str());

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
            assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, &mesh.blas, {}, nullptr)));
            mesh.blas->GetResource()->SetName(std::format(L"{}Mesh{}Blas", filePath.stem().wstring(), meshIndex).c_str());
            blasDesc.Width = prebuildInfo.ScratchDataSizeInBytes;
            assert(SUCCEEDED(d3d.allocator->CreateResource(&blasAllocationDesc, &blasDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, &mesh.blasScratch, {}, nullptr)));
            mesh.blasScratch->GetResource()->SetName(std::format(L"{}Mesh{}BlasScratch", filePath.stem().wstring(), meshIndex).c_str());
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = mesh.blas->GetResource()->GetGPUVirtualAddress(), .Inputs = inputs, .ScratchAccelerationStructureData = mesh.blasScratch->GetResource()->GetGPUVirtualAddress()};
            d3d.transferCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
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
                skin.joints.push_back(ModelJoint{.node = node, .inverseBindMat = XMMATRIX(matsData)});
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
                } else if (gltfSampler.interpolation == cgltf_interpolation_type_step) {
                    sampler.interpolation = AnimationSamplerInterpolationStep;
                } else if (gltfSampler.interpolation == cgltf_interpolation_type_cubic_spline) {
                    sampler.interpolation = AnimationSamplerInterpolationCubicSpline;
                    assert(false);
                } else {
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
                assert(nodeIndex >= 0 && nodeIndex < gltfData->nodes_count);
                assert(samplerIndex >= 0 && samplerIndex < gltfAnimation.samplers_count);
                channel.node = &model->nodes[nodeIndex];
                channel.sampler = &animation.samplers[samplerIndex];
                if (gltfChannel.target_path == cgltf_animation_path_type_translation) {
                    assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                    channel.type = AnimationChannelTypeTranslate;
                } else if (gltfChannel.target_path == cgltf_animation_path_type_rotation) {
                    assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec4);
                    channel.type = AnimationChannelTypeRotate;
                } else if (gltfChannel.target_path == cgltf_animation_path_type_scale) {
                    assert(gltfAnimation.samplers[samplerIndex].output->type == cgltf_type_vec3);
                    channel.type = AnimationChannelTypeScale;
                } else {
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
                image.gpuData = d3dCreateImageDDS(imageDDSFilePath, std::format(L"{}Image{}", filePath.stem().wstring(), imageIndex).c_str());
            } else if (std::filesystem::exists(imageFilePath)) {
                image.gpuData = d3dCreateImageSTB(imageFilePath, std::format(L"{}Image{}", filePath.stem().wstring(), imageIndex).c_str());
            } else {
                assert(false);
            }
            D3D12_RESOURCE_DESC imageDesc = image.gpuData->GetResource()->GetDesc();
            image.srvDesc = {.Format = imageDesc.Format, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = imageDesc.MipLevels}};
            D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = image.gpuData->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
            d3d.transferCmdList->ResourceBarrier(1, &barrier);
        }
        for (uint textureIndex = 0; textureIndex < gltfData->textures_count; textureIndex++) {
            cgltf_texture& gltfTexture = gltfData->textures[textureIndex];
            ModelTexture& texture = model->textures[textureIndex];
            assert(gltfTexture.image);
            texture.image = &model->images[gltfTexture.image - &gltfData->images[0]];
        }
        for (uint materialIndex = 0; materialIndex < gltfData->materials_count; materialIndex++) {
            cgltf_material& gltfMaterial = gltfData->materials[materialIndex];
            ModelMaterial& material = model->materials[materialIndex];
            if (gltfMaterial.name) material.name = gltfMaterial.name;
            assert(gltfMaterial.has_pbr_metallic_roughness);
            material.baseColorFactor = float4(gltfMaterial.pbr_metallic_roughness.base_color_factor);
            if (gltfMaterial.pbr_metallic_roughness.base_color_texture.texture) {
                assert(gltfMaterial.pbr_metallic_roughness.base_color_texture.texcoord == 0);
                assert(!gltfMaterial.pbr_metallic_roughness.base_color_texture.has_transform);
                material.baseColorTexture = &model->textures[gltfMaterial.pbr_metallic_roughness.base_color_texture.texture - &gltfData->textures[0]];
            }
        }
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }

    ModelInstance modelInstance = {};
    modelInstance.model = model;
    modelInstance.meshNodes.resize(model->meshNodes.size());
    for (uint meshNodeIndex = 0; meshNodeIndex < model->meshNodes.size(); meshNodeIndex++) {
        ModelNode* meshNode = model->meshNodes[meshNodeIndex];
        ModelInstanceMeshNode& instanceMeshNode = modelInstance.meshNodes[meshNodeIndex];
        instanceMeshNode.transformMat = meshNode->globalTransform;
        if (!meshNode->skin || model->animations.size() == 0) {
            instanceMeshNode.verticesBuffer = meshNode->mesh->verticesBuffer;
            instanceMeshNode.blas = meshNode->mesh->blas;
            instanceMeshNode.blasScratch = meshNode->mesh->blasScratch;
        } else {
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

void modelInstanceRelease(ModelInstance* modelInstance) {
    for (ModelInstanceSkin& skin : modelInstance->skins) {
        skin.matsBuffer->Release();
    }
    for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance->model->meshNodes.size(); meshNodeIndex++) {
        if (modelInstance->model->meshNodes[meshNodeIndex]->skin) {
            modelInstance->meshNodes[meshNodeIndex].verticesBuffer->Release();
            modelInstance->meshNodes[meshNodeIndex].blas->Release();
            modelInstance->meshNodes[meshNodeIndex].blasScratch->Release();
        }
    }
}

void imguiTransform(Transform* transform) {
    if (ImGui::TreeNode("Transform")) {
        ImGui::InputFloat3("S", &transform->s.x), ImGui::SameLine();
        if (ImGui::Button("reset##scale")) transform->s = float3(1, 1, 1);
        ImGui::InputFloat4("R", &transform->r.x), ImGui::SameLine();
        if (ImGui::Button("reset##rotate")) transform->r = float4(0, 0, 0, 1);
        ImGui::InputFloat3("T", &transform->t.x), ImGui::SameLine();
        if (ImGui::Button("reset##translate")) transform->t = float3(0, 0, 0);
        ImGui::TreePop();
    }
}

void imguiModelInstance(ModelInstance* modelInstance) {
    if (ImGui::TreeNode("Model")) {
        ImGui::Text(std::format("File: {}", modelInstance->model->filePath.string()).c_str());
        imguiTransform(&modelInstance->transform);
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
    if (!modelInstance->animation) {
        for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance->meshNodes.size(); meshNodeIndex++) {
            modelInstance->meshNodes[meshNodeIndex].transformMat = modelInstance->model->meshNodes[meshNodeIndex]->globalTransform;
        }
    } else {
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
                    } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                        modelInstance->localTransforms[nodeIndex].t = percentage < 1.0f ? frame0.xyz() : frame1.xyz();
                    }
                } else if (channel.type == AnimationChannelTypeRotate) {
                    if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                        modelInstance->localTransforms[nodeIndex].r = slerp(frame0, frame1, percentage);
                    } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                        modelInstance->localTransforms[nodeIndex].r = percentage < 1.0f ? frame0 : frame1;
                    }
                } else if (channel.type == AnimationChannelTypeScale) {
                    if (channel.sampler->interpolation == AnimationSamplerInterpolationLinear) {
                        modelInstance->localTransforms[nodeIndex].s = lerp(frame0.xyz(), frame1.xyz(), percentage);
                    } else if (channel.sampler->interpolation == AnimationSamplerInterpolationStep) {
                        modelInstance->localTransforms[nodeIndex].s = percentage < 1.0f ? frame0.xyz() : frame1.xyz();
                    }
                }
            }
        }
        for (ModelNode* rootNode : modelInstance->model->rootNodes) {
            modelTraverseNodesAndGetGlobalTransforms(modelInstance->model, rootNode, XMMatrixIdentity(), modelInstance->localTransforms, modelInstance->globalTransformMats);
        }
        for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance->meshNodes.size(); meshNodeIndex++) {
            int64 nodeIndex = modelInstance->model->meshNodes[meshNodeIndex] - &modelInstance->model->nodes[0];
            modelInstance->meshNodes[meshNodeIndex].transformMat = XMMatrixMultiply(modelInstance->model->meshNodes[meshNodeIndex]->globalTransform, modelInstance->globalTransformMats[nodeIndex]);
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
}

void playerCameraSetPitchYaw(float2 pitchYawNew) {
    player.camera.pitchYaw.x = std::clamp(pitchYawNew.x, -pi * 0.1f, pi * 0.4f);
    player.camera.pitchYaw.y = std::remainderf(pitchYawNew.y, pi * 2.0f);
    XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(player.camera.pitchYaw.x, player.camera.pitchYaw.y, 0);
    float3 dir = float3(XMVector3Rotate(XMVectorSet(0, 0, -1, 0), quaternion)).normalize();
    player.camera.position = player.camera.lookAt + (dir * player.camera.distance);
}
void playerCameraTranslate(float3 translate) {
    player.camera.position += translate;
    player.camera.lookAt += translate;
}

void editorCameraRotate(float2 pitchYawDelta) {
    editor->camera.pitchYaw.x += pitchYawDelta.x;
    editor->camera.pitchYaw.y += pitchYawDelta.y;
    editor->camera.pitchYaw.x = std::clamp(editor->camera.pitchYaw.x, -pi * 0.4f, pi * 0.4f);
    editor->camera.pitchYaw.y = std::remainderf(editor->camera.pitchYaw.y, pi * 2.0f);
    XMVECTOR quaternion = XMQuaternionRotationRollPitchYaw(editor->camera.pitchYaw.x, editor->camera.pitchYaw.y, 0);
    float3 dir = XMVector3Rotate(XMVectorSet(0, 0, 1, 0), quaternion);
    editor->camera.lookAt = editor->camera.position + dir;
}
void editorCameraTranslate(float3 translate) {
    float3 dz = (editor->camera.lookAt - editor->camera.position).normalize();
    float3 dx = dz.cross({0, 1, 0});
    float3 dy = dz.cross({1, 0, 0});
    editor->camera.position += dx * translate.x;
    editor->camera.lookAt += dx * translate.x;
    editor->camera.position += dy * translate.y;
    editor->camera.lookAt += dy * translate.y;
    editor->camera.position += dz * translate.z;
    editor->camera.lookAt += dz * translate.z;
}

void editorCameraFocus(float3 position, float distance) {
}

void loadSimpleAssets() {
    modelInstanceCube = modelInstanceInit("models/cube/cube.gltf");
    modelInstanceCylinder = modelInstanceInit("models/cylinder/cylinder.gltf");
    modelInstanceSphere = modelInstanceInit("models/sphere/sphere.gltf");
}

void operator>>(ryml::ConstNodeRef node, float2& v) { node[0] >> v.x, node[1] >> v.y; }
void operator>>(ryml::ConstNodeRef node, float3& v) { node[0] >> v.x, node[1] >> v.y, node[2] >> v.z; }
void operator>>(ryml::ConstNodeRef node, float4& v) { node[0] >> v.x, node[1] >> v.y, node[2] >> v.z, node[3] >> v.w; }
void operator>>(ryml::ConstNodeRef node, Position& p) { node[0] >> p.x, node[1] >> p.y, node[2] >> p.z; }
void operator>>(ryml::ConstNodeRef node, Transform& t) { node["scale"] >> t.s, node["rotate"] >> t.r, node["translate"] >> t.t; }

void operator<<(ryml::NodeRef node, float2 v) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << v.x, node.append_child() << v.y; }
void operator<<(ryml::NodeRef node, float3 v) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << v.x, node.append_child() << v.y, node.append_child() << v.z; }
void operator<<(ryml::NodeRef node, float4 v) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << v.x, node.append_child() << v.y, node.append_child() << v.z, node.append_child() << v.w; }
void operator<<(ryml::NodeRef node, Position p) { node |= ryml::SEQ, node |= ryml::_WIP_STYLE_FLOW_SL, node.append_child() << p.x, node.append_child() << p.y, node.append_child() << p.z; }
void operator<<(ryml::NodeRef node, Transform t) { node["scale"] << t.s, node["rotate"] << t.r, node["translate"] << t.t; }

void worldInit(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) assert(false);

    std::string yamlStr = fileReadStr(path);
    ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
    ryml::ConstNodeRef yamlRoot = yamlTree.rootref();
    worldFilePath = path;

    if (editor) {
        if (yamlRoot.has_child("editorCamera")) {
            ryml::ConstNodeRef cameraYaml = yamlRoot["editorCamera"];
            cameraYaml["position"] >> editor->camera.position;
            cameraYaml["pitchYaw"] >> editor->camera.pitchYaw;
            editorCameraRotate({0, 0});
            cameraYaml["moveSpeed"] >> editor->camera.moveSpeed;
        } else {
            editor->camera = {};
        }
    }
    {
        ryml::ConstNodeRef skyboxYaml = yamlRoot["skybox"];
        std::string file;
        skyboxYaml["file"] >> file;
        skybox.hdriTextureFilePath = file;
        d3dTransferQueueStartRecording();
        d3d.stagingBufferOffset = 0;
        skybox.hdriTexture = d3dCreateImageDDS(assetsDir / skybox.hdriTextureFilePath);
        D3D12_RESOURCE_BARRIER barrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = skybox.hdriTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST, .StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}};
        d3d.transferCmdList->ResourceBarrier(1, &barrier);
        d3dTransferQueueSubmitRecording();
        d3dTransferQueueWait();
    }
    {
        ryml::ConstNodeRef playerYaml = yamlRoot["player"];
        std::string file;
        playerYaml["file"] >> file;
        player.modelInstance = modelInstanceInit(file);
        playerYaml >> player.modelInstance.transform;
        playerYaml["spawnPosition"] >> player.spawnPosition;
        player.position = player.spawnPosition;
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
        player.camera.lookAt = player.position + player.camera.lookAtOffset;
        playerCameraSetPitchYaw({0, 0});
    }
    ryml::ConstNodeRef gameObjectsYaml = yamlRoot["gameObjects"];
    for (ryml::ConstNodeRef const& gameObjectYaml : gameObjectsYaml) {
        GameObject& obj = gameObjects.emplace_back();
        gameObjectYaml["name"] >> obj.name;
        std::string file;
        gameObjectYaml["file"] >> file;
        obj.modelInstance = modelInstanceInit(file);
        gameObjectYaml >> obj.modelInstance.transform;
    }
}

void worldSave() {
    if (!editor) return;

    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;

    ryml::NodeRef cameraYaml = yamlRoot["editorCamera"];
    cameraYaml |= ryml::MAP;
    cameraYaml["position"] << editor->camera.position;
    cameraYaml["pitchYaw"] << editor->camera.pitchYaw;
    cameraYaml["moveSpeed"] << editor->camera.moveSpeed;

    ryml::NodeRef skyboxYaml = yamlRoot["skybox"];
    skyboxYaml |= ryml::MAP;
    skyboxYaml["file"] << skybox.hdriTextureFilePath.string();

    ryml::NodeRef playerYaml = yamlRoot["player"];
    playerYaml |= ryml::MAP;
    playerYaml["file"] << player.modelInstance.model->filePath.string();
    playerYaml << player.modelInstance.transform;
    playerYaml["spawnPosition"] << player.spawnPosition;
    playerYaml["walkSpeed"] << player.walkSpeed;
    playerYaml["runSpeed"] << player.runSpeed;
    playerYaml["idleAnimationIndex"] << player.idleAnimationIndex;
    playerYaml["walkAnimationIndex"] << player.walkAnimationIndex;
    playerYaml["runAnimationIndex"] << player.runAnimationIndex;
    playerYaml["jumpAnimationIndex"] << player.jumpAnimationIndex;
    playerYaml["cameraLookAtOffset"] << player.camera.lookAtOffset;
    playerYaml["cameraDistance"] << player.camera.distance;

    ryml::NodeRef gameObjectsYaml = yamlRoot["gameObjects"];
    gameObjectsYaml |= ryml::SEQ;
    for (GameObject& gameObject : gameObjects) {
        ryml::NodeRef gameObjectYaml = gameObjectsYaml.append_child();
        gameObjectYaml |= ryml::MAP;
        gameObjectYaml["name"] << gameObject.name;
        gameObjectYaml["file"] << gameObject.modelInstance.model->filePath.string();
        gameObjectYaml << gameObject.modelInstance.transform;
    }

    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(worldFilePath, yamlStr);
}

void worldReadSave(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return;

    std::string yamlStr = fileReadStr(path);
    ryml::Tree yamlTree = ryml::parse_in_arena(ryml::to_csubstr(yamlStr));
    ryml::ConstNodeRef yamlRoot = yamlTree.rootref();

    ryml::ConstNodeRef playerYaml = yamlRoot["player"];
    playerYaml["position"] >> player.position;
}

void worldWriteSave(const std::filesystem::path& path) {
    ryml::Tree yamlTree;
    ryml::NodeRef yamlRoot = yamlTree.rootref();
    yamlRoot |= ryml::MAP;

    ryml::NodeRef playerYaml = yamlRoot["player"];
    playerYaml |= ryml::MAP;
    playerYaml["position"] << player.position;

    std::string yamlStr = ryml::emitrs_yaml<std::string>(yamlTree);
    fileWriteStr(path, yamlStr);
}

void worldReset() {
    if (!editor) return;
    player.position = player.spawnPosition;
    player.camera.lookAt = player.spawnPosition + player.camera.lookAtOffset;
    playerCameraSetPitchYaw({0, 0});
}

void gameObjectRelease(GameObject* obj) {
    modelInstanceRelease(&obj->modelInstance);
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
    case WM_INPUT_DEVICE_CHANGE: {
        if (wParam == GIDC_ARRIVAL) {
            RID_DEVICE_INFO info;
            uint infoSize = sizeof(info);
            GetRawInputDeviceInfoA((HANDLE)lParam, RIDI_DEVICEINFO, &info, &infoSize);
            if (info.dwType == RIM_TYPEHID && info.hid.dwVendorId == 0x054c && info.hid.dwProductId == 0x0ce6) controllerDualSenseHID = (HANDLE)lParam;
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
            } else if (rawInput->header.dwType == RIM_TYPEHID && rawInput->header.hDevice == controllerDualSenseHID) {
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

void editorMainMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        editorMainMenuBarPos = ImGui::GetWindowPos();
        editorMainMenuBarSize = ImGui::GetWindowSize();
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
                d3dApplySettings();
            } else if (ImGui::MenuItem("Windowed")) {
                settings.windowMode = WindowModeWindowed;
                d3dApplySettings();
            } else if (ImGui::MenuItem("Borderless Fullscreen")) {
                settings.windowMode = WindowModeBorderless;
                d3dApplySettings();
            }
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Horizontal);
            ImGui::Text("Exclusive Fullscreen");
            for (DisplayMode& mode : d3d.displayModes) {
                std::string text = std::format("{}x{}", mode.resolutionW, mode.resolutionH);
                if (ImGui::BeginMenu(text.c_str())) {
                    for (DXGI_RATIONAL& refreshRate : mode.refreshRates) {
                        text = std::format("{:.2f}hz", (float)refreshRate.Numerator / (float)refreshRate.Denominator);
                        if (ImGui::MenuItem(text.c_str())) {
                            settings.windowMode = WindowModeFullscreen;
                            settings.windowW = mode.resolutionW;
                            settings.windowH = mode.resolutionH;
                            settings.refreshRate = refreshRate;
                            d3dApplySettings();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Editor")) {
            if (ImGui::BeginMenu("Camera")) {
                ImGui::SliderFloat("moveSpeed", &editor->camera.moveSpeed, 0.1f, 100.0f);
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Add")) {
                editorAddObjectPopupFlag = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Game")) {
            if (ImGui::MenuItem("Play", "CTRL+P")) {
                editor->active = false;
                windowHideCursor(true);
                worldReset();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void editorObjectsWindow() {
    editorObjectWindowPos = ImVec2(settings.renderW * 0.85f, editorMainMenuBarSize.y);
    editorObjectWindowSize = ImVec2(settings.renderW * 0.15f, settings.renderH * 0.3f);
    ImGui::SetNextWindowPos(editorObjectWindowPos);
    ImGui::SetNextWindowSize(editorObjectWindowSize);
    if (ImGui::Begin("Objects")) {
        if (ImGui::Selectable("Player", editor->selectedObjectType == ObjectTypePlayer)) {
            editor->selectedObjectType = ObjectTypePlayer;
            editor->selectedObjectIndex = 0;
        }
        if (editor->selectedObjectType == ObjectTypePlayer && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("player edit");
        }
        if (editor->selectedObjectType == ObjectTypePlayer && ImGui::BeginPopup("player edit")) {
            if (ImGui::Selectable("focus")) editorCameraFocus(player.modelInstance.transform.t, 1);
            ImGui::EndPopup();
        }
        if (ImGui::TreeNode("Game Objects")) {
            int objID = 0;
            for (uint objIndex = 0; objIndex < gameObjects.size(); objIndex++) {
                GameObject& object = gameObjects[objIndex];
                ImGui::PushID(objID++);
                if (ImGui::Selectable(object.name.c_str(), editor->selectedObjectType == ObjectTypeGameObject && editor->selectedObjectIndex == objIndex)) {
                    editor->selectedObjectType = ObjectTypeGameObject;
                    editor->selectedObjectIndex = objIndex;
                }
                ImGui::PopID();
                if (editor->selectedObjectType == ObjectTypeGameObject && editor->selectedObjectIndex == objIndex && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    ImGui::OpenPopup("game object edit");
                }
                if (editor->selectedObjectType == ObjectTypeGameObject && editor->selectedObjectIndex == objIndex && ImGui::BeginPopup("game object edit")) {
                    if (ImGui::Selectable("focus")) {
                        editorCameraFocus(object.modelInstance.transform.t, 1);
                    }
                    if (ImGui::Selectable("delete")) {
                        object.toBeDeleted = true;
                        editor->selectedObjectType = ObjectTypeNone;
                    }
                    ImGui::EndPopup();
                }
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

void editorObjectPropertiesWindow() {
    if (ImGui::Begin("Properties")) {
        editorObjectPropertiesWindowPos = ImGui::GetWindowPos();
        editorObjectPropertiesWindowSize = ImGui::GetWindowSize();
        if (editor->selectedObjectType == ObjectTypePlayer) {
            ImGui::Text("Player");
            imguiModelInstance(&player.modelInstance);
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
            if (ImGui::TreeNode("Movement")) {
                float3 spawnPoint = player.spawnPosition.toFloat3();
                if (ImGui::InputFloat3("SpawnPosition", &spawnPoint.x)) {
                    player.spawnPosition = spawnPoint;
                }
                ImGui::InputFloat3("Velocity", &player.velocity.x);
                ImGui::InputFloat3("Acceleration", &player.acceleration.x);
                ImGui::TreePop();
            }
        } else if (editor->selectedObjectType == ObjectTypeGameObject) {
            GameObject& object = gameObjects[editor->selectedObjectIndex];
            ImGui::Text("Game Object #%d", editor->selectedObjectIndex);
            ImGui::Text("Name \"%s\"", object.name.c_str());
            imguiModelInstance(&object.modelInstance);
        }
    }
    ImGui::End();
}

void editorAddObjectPopup() {
    if (editorAddObjectPopupFlag) {
        ImGui::OpenPopup("Add Object");
        editorAddObjectPopupFlag = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Add Object", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        static int objectType = 0;
        static char objectName[32] = {};
        static char filePath[256] = {};
        ImGui::Combo("Object Type", &objectType, "Game Object\0");
        if (objectType == 0) {
            ImGui::InputText("Name", objectName, sizeof(objectName));
            ImGui::InputText("File", filePath, sizeof(filePath)), ImGui::SameLine();
            if (ImGui::Button("Browse")) {
                OPENFILENAMEA openfileName = {.lStructSize = sizeof(OPENFILENAMEA), .hwndOwner = window.hwnd, .lpstrFile = filePath, .nMaxFile = sizeof(filePath)};
                GetOpenFileNameA(&openfileName);
            }
        }
        if (ImGui::Button("Add")) {
            if (objectName[0] == '\0') {
                ImGui::DebugLog("error: object name is empty");
            } else {
                std::filesystem::path path = std::filesystem::relative(filePath, assetsDir);
                if (objectType == 0) {
                    GameObject gameObject = {.name = objectName, .modelInstance = modelInstanceInit(path)};
                    gameObjects.push_back(std::move(gameObject));
                }
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

void editorUpdate() {
    if ((ImGui::IsKeyPressed(ImGuiKey_P, false) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) || (controller.back && controller.backDownDuration == 0)) {
        editor->active = false;
        windowHideCursor(true);
        worldReset();
        return;
    }
    if (d3d.collisionQueriesFenceValue > 0) {
        d3dWaitForCollisionQueries();
        if (mouseSelectX != UINT_MAX && mouseSelectY != UINT_MAX) {
            uint mouseSelectInstanceIndex = d3d.collisionQueryResultsBufferPtr[0].instanceIndex;
            if (mouseSelectInstanceIndex == UINT_MAX) {
                editor->selectedObjectType = ObjectTypeNone;
            } else {
                TLASInstanceInfo& info = tlasInstancesInfos[mouseSelectInstanceIndex];
                editor->selectedObjectType = info.objectType;
                editor->selectedObjectIndex = info.objectIndex;
            }
        }
    }
    {
        modelInstanceUpdateAnimation(&player.modelInstance, frameTime);
        for (GameObject& obj : gameObjects) modelInstanceUpdateAnimation(&obj.modelInstance, frameTime);
    }

    static ImVec2 mousePosPrev = ImGui::GetMousePos();
    ImVec2 mousePos = ImGui::GetMousePos();
    ImVec2 mouseDelta = {mousePos.x - mousePosPrev.x, mousePos.y - mousePosPrev.y};
    mousePosPrev = mousePos;

    editorMainMenuBar();
    editorObjectsWindow();
    editorObjectPropertiesWindow();
    editorAddObjectPopup();

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
        mouseSelectX = (uint)mousePos.x;
        mouseSelectY = (uint)mousePos.y;
    } else {
        mouseSelectX = UINT_MAX;
        mouseSelectY = UINT_MAX;
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::GetIO().WantCaptureMouse) {
        editor->cameraMoving = true;
        windowHideCursor(true);
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        editor->cameraMoving = false;
        windowHideCursor(false);
    }
    if (editor->cameraMoving || controllerStickMoved()) {
        float pitch = (mouseDeltaRaw.y * mouseSensitivity - controller.rsY * controllerSensitivity) * (float)frameTime;
        float yaw = (mouseDeltaRaw.x * mouseSensitivity + controller.rsX * controllerSensitivity) * (float)frameTime;
        editor->camera.moveSpeed = std::clamp(editor->camera.moveSpeed + ImGui::GetIO().MouseWheel, 0.1f, 100.0f);
        float distance = (float)frameTime / 5.0f * editor->camera.moveSpeed;
        float3 translate = {-controller.lsX * distance, 0, controller.lsY * distance};
        if (ImGui::IsKeyDown(ImGuiKey_W)) translate.z = distance;
        if (ImGui::IsKeyDown(ImGuiKey_S)) translate.z = -distance;
        if (ImGui::IsKeyDown(ImGuiKey_A)) translate.x = distance;
        if (ImGui::IsKeyDown(ImGuiKey_D)) translate.x = -distance;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) translate.y = distance;
        if (ImGui::IsKeyDown(ImGuiKey_E)) translate.y = -distance;
        editorCameraRotate({pitch, yaw});
        editorCameraTranslate(translate);
    }

    const XMMATRIX lookAtMat = XMMatrixLookAtLH(editor->camera.position.toXMVector(), editor->camera.lookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
    const XMMATRIX perspectiveMat = XMMatrixPerspectiveFovLH(RADIAN(editor->camera.fovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f);
    auto transformGizmo = [&](Transform* transform) {
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
            ImGui::EndPopup();
        }
        if (!ImGui::IsAnyItemActive()) {
            if (ImGui::IsKeyPressed(ImGuiKey_T)) gizmoOperation = ImGuizmo::TRANSLATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_R)) gizmoOperation = ImGuizmo::ROTATE;
            else if (ImGui::IsKeyPressed(ImGuiKey_S)) gizmoOperation = ImGuizmo::SCALE;
        }
        XMMATRIX transformMat = transform->toMat();
        if (ImGuizmo::Manipulate((const float*)&lookAtMat, (const float*)&perspectiveMat, gizmoOperation, gizmoMode, (float*)&transformMat)) {
            XMVECTOR scale, rotate, translate;
            if (XMMatrixDecompose(&scale, &rotate, &translate, transformMat)) {
                transform->s = scale, transform->r = rotate, transform->t = translate;
            }
        }
    };
    if (editor->selectedObjectType == ObjectTypePlayer) {
        Transform transform = player.modelInstance.transform;
        transform.t += player.spawnPosition.toFloat3();
        transformGizmo(&transform);
        transform.t -= player.spawnPosition.toFloat3();
        player.modelInstance.transform = transform;
    } else if (editor->selectedObjectType == ObjectTypeGameObject && editor->selectedObjectIndex < gameObjects.size()) {
        transformGizmo(&gameObjects[editor->selectedObjectIndex].modelInstance.transform);
    }

    ImGui::ShowDebugLogWindow();

    {
        auto objIter = gameObjects.begin();
        while (objIter != gameObjects.end()) {
            if (objIter->toBeDeleted) {
                gameObjectRelease(&*objIter);
                objIter = gameObjects.erase(objIter);
            } else {
                objIter++;
            }
        }
    }
}

void gameUpdate() {
    if (editor && ImGui::IsKeyPressed(ImGuiKey_P, false) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || (controller.back && controller.backDownDuration == 0)) {
        editor->active = true;
        windowHideCursor(false);
        return;
    }
    if (d3d.collisionQueriesFenceValue > 0) {
        d3dWaitForCollisionQueries();
        CollisionQueryResult queryResult = d3d.collisionQueryResultsBufferPtr[1];
        // std::string str = std::format("player Movement: {}\nqueryResultDistance: {}\nqueryResultInstance: {}\n", player.movement.toString(), queryResult.distance.toString(), queryResult.instanceIndex);
        // ImGui::DebugLog("%s", str.c_str());
        if (queryResult.instanceIndex == UINT_MAX) {
            // ImGui::DebugLog("null\n");
            if (player.movement != float3(0, 0, 0)) {
                player.position += player.movement;
                playerCameraTranslate(player.movement);
                float angle = acosf(player.movement.normalize().dot(float3(0, 0, -1)));
                if (player.movement.x > 0) angle = -angle;
                player.PitchYawRoll = float3(0, angle, 0);
            }
        } else {
            TLASInstanceInfo& instanceInfo = tlasInstancesInfos[queryResult.instanceIndex];
            if (instanceInfo.objectType == ObjectTypeGameObject) {
                // ImGui::DebugLog("%s\n", gameObjects[instanceInfo.objectIndex].name.c_str());
            } else {
                // ImGui::DebugLog("wtf\n");
            }
        }
    }
    {
        float pitch = (mouseDeltaRaw.y * mouseSensitivity - controller.rsY * controllerSensitivity) * (float)frameTime;
        float yaw = (mouseDeltaRaw.x * mouseSensitivity + controller.rsX * controllerSensitivity) * (float)frameTime;
        playerCameraSetPitchYaw(player.camera.pitchYaw + float2(pitch, yaw));

        float3 moveDir = {0, 0, 0};
        bool jump = false;
        {
            float3 forwardDir = player.camera.lookAt - player.camera.position;
            forwardDir.y = 0;
            forwardDir = forwardDir.normalize();
            float3 sideDir = forwardDir.cross(float3(0, 1, 0));
            if (ImGui::IsKeyDown(ImGuiKey_W)) moveDir += forwardDir;
            if (ImGui::IsKeyDown(ImGuiKey_S)) moveDir += -forwardDir;
            if (ImGui::IsKeyDown(ImGuiKey_A)) moveDir += sideDir;
            if (ImGui::IsKeyDown(ImGuiKey_D)) moveDir += -sideDir;
            moveDir += forwardDir * controller.lsY;
            moveDir += sideDir * -controller.lsX;
            moveDir = moveDir.normalize();

            jump = ImGui::IsKeyPressed(ImGuiKey_Space);
            if (jump) { ImGui::DebugLog("jump\n"); }
        }

        if (moveDir == float3{0, 0, 0}) {
            player.state = PlayerStateIdle;
            player.movement = {0, 0, 0};
            player.modelInstance.animation = &player.modelInstance.model->animations[player.idleAnimationIndex];
        } else {
            if (!ImGui::IsKeyDown(ImGuiKey_LeftShift) && sqrtf(controller.lsX * controller.lsX + controller.lsY * controller.lsY) < 0.8f) {
                player.state = PlayerStateWalk;
                player.movement = moveDir * player.walkSpeed * (float)frameTime;
                player.modelInstance.animation = &player.modelInstance.model->animations[player.walkAnimationIndex];
            } else {
                player.state = PlayerStateRun;
                player.movement = moveDir * player.runSpeed * (float)frameTime;
                player.modelInstance.animation = &player.modelInstance.model->animations[player.runAnimationIndex];
            }
        }
        // if (player.movement != float3(0, 0, 0)) {
        //     player.position += player.movement;
        //     playerCameraTranslate(player.movement);
        //     float angle = acosf(player.movement.normalize().dot(float3(0, 0, -1)));
        //     if (player.movement.x > 0) angle = -angle;
        //     player.PitchYawRoll = float3(0, angle, 0);
        // }
    }
    {
        modelInstanceUpdateAnimation(&player.modelInstance, frameTime);
        for (GameObject& obj : gameObjects) modelInstanceUpdateAnimation(&obj.modelInstance, frameTime);
    }
    ImGui::ShowDebugLogWindow();
}

void update() {
    ImGui::GetIO().DeltaTime = (float)frameTime;
    ImGui::GetIO().DisplaySize = ImVec2((float)settings.renderW, (float)settings.renderH);
    ImGui::NewFrame();
    ImGuizmo::SetRect(0, 0, (float)settings.renderW, (float)settings.renderH);
    ImGuizmo::BeginFrame();

    if (editor && editor->active) {
        editorUpdate();
    } else {
        gameUpdate();
    }

    ImGui::Render();
}

void addTLASInstance(ModelInstance& modelInstance, const XMMATRIX& objectTransform, ObjectType objectType, uint objectIndex) {
    TLASInstanceInfo tlasInstanceInfo = {.objectType = objectType, .objectIndex = objectIndex, .blasGeometriesOffset = (uint)blasGeometriesInfos.size()};
    if (editor && editor->active) {
        if (objectType != ObjectTypeNone && editor->selectedObjectType == objectType && editor->selectedObjectIndex == objectIndex) {
            tlasInstanceInfo.flags |= TLASInstanceFlagSelected;
        }
    }
    for (uint meshNodeIndex = 0; meshNodeIndex < modelInstance.meshNodes.size(); meshNodeIndex++) {
        ModelNode* meshNode = modelInstance.model->meshNodes[meshNodeIndex];
        ModelInstanceMeshNode& instanceMeshNode = modelInstance.meshNodes[meshNodeIndex];
        XMMATRIX transform = instanceMeshNode.transformMat;
        transform = XMMatrixMultiply(transform, XMMatrixScaling(1, 1, -1)); // convert RH to LH
        transform = XMMatrixMultiply(transform, modelInstance.transform.toMat());
        transform = XMMatrixMultiply(transform, objectTransform);
        transform = XMMatrixTranspose(transform);
        D3D12_RAYTRACING_INSTANCE_DESC tlasInstanceBuildInfo = {.InstanceID = d3d.cbvSrvUavDescriptorCount, .InstanceMask = objectType, .AccelerationStructure = instanceMeshNode.blas->GetResource()->GetGPUVirtualAddress()};
        memcpy(tlasInstanceBuildInfo.Transform, &transform, sizeof(tlasInstanceBuildInfo.Transform));
        tlasInstancesBuildInfos.push_back(tlasInstanceBuildInfo);
        tlasInstancesInfos.push_back(tlasInstanceInfo);
        for (uint primitiveIndex = 0; primitiveIndex < meshNode->mesh->primitives.size(); primitiveIndex++) {
            ModelPrimitive& primitive = meshNode->mesh->primitives[primitiveIndex];
            D3D12_SHADER_RESOURCE_VIEW_DESC vertexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.verticesBufferOffset, .NumElements = primitive.verticesCount, .StructureByteStride = sizeof(struct Vertex)}};
            D3D12_SHADER_RESOURCE_VIEW_DESC indexBufferDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.FirstElement = primitive.indicesBufferOffset, .NumElements = primitive.indicesCount, .StructureByteStride = sizeof(uint)}};
            d3dAppendSRVDescriptor(&vertexBufferDesc, instanceMeshNode.verticesBuffer->GetResource());
            d3dAppendSRVDescriptor(&indexBufferDesc, meshNode->mesh->indicesBuffer->GetResource());
            if (primitive.material) {
                blasGeometriesInfos.push_back(BLASGeometryInfo{.baseColorFactor = primitive.material->baseColorFactor});
                if (primitive.material->baseColorTexture) {
                    d3dAppendSRVDescriptor(&primitive.material->baseColorTexture->image->srvDesc, primitive.material->baseColorTexture->image->gpuData->GetResource());
                } else {
                    d3dAppendSRVDescriptor(&d3d.defaultMaterialBaseColorImageSRVDesc, d3d.defaultMaterialBaseColorImage->GetResource());
                }
            } else {
                blasGeometriesInfos.push_back(BLASGeometryInfo{.baseColorFactor = {0.7f, 0.7f, 0.7f, 1.0f}});
                d3dAppendSRVDescriptor(&d3d.defaultMaterialBaseColorImageSRVDesc, d3d.defaultMaterialBaseColorImage->GetResource());
            }
        }
    }
}

D3D12_DISPATCH_RAYS_DESC fillRayTracingShaderTable(ID3D12Resource* buffer, uint8* bufferPtr, uint* bufferOffset, void* rayGenID, std::span<void*> missIDs, std::span<void*> hitGroupIDs) {
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    *bufferOffset = align(*bufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    memcpy(bufferPtr + *bufferOffset, rayGenID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    dispatchDesc.RayGenerationShaderRecord = {buffer->GetGPUVirtualAddress() + *bufferOffset, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
    *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.MissShaderTable = {buffer->GetGPUVirtualAddress() + *bufferOffset, missIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* missID : missIDs) {
        memcpy(bufferPtr + *bufferOffset, missID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    *bufferOffset = align(*bufferOffset, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatchDesc.HitGroupTable = {buffer->GetGPUVirtualAddress() + *bufferOffset, hitGroupIDs.size() * D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT};
    for (void* hitGroupID : hitGroupIDs) {
        memcpy(bufferPtr + *bufferOffset, hitGroupID, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        *bufferOffset = align(*bufferOffset + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    }
    return dispatchDesc;
}

void render() {
    d3dWaitForRender();

    assert(SUCCEEDED(d3d.graphicsCmdAllocator->Reset()));
    assert(SUCCEEDED(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocator, nullptr)));

    d3d.cbvSrvUavDescriptorCount = 0;
    d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap);

    tlasInstancesBuildInfos.resize(0);
    tlasInstancesInfos.resize(0);
    blasGeometriesInfos.resize(0);

    d3d.stagingBufferOffset = 0;
    d3d.constantsBufferOffset = 0;

    float3 cameraLookAt = player.camera.lookAt - player.camera.position;
    float cameraFovVertical = player.camera.fovVertical;
    if (editor && editor->active) {
        cameraLookAt = editor->camera.lookAt - editor->camera.position;
        cameraFovVertical = editor->camera.fovVertical;
    }
    XMMATRIX cameraLookAtMat = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 0), cameraLookAt.toXMVector(), XMVectorSet(0, 1, 0, 0));
    XMMATRIX cameraLookAtMatInverseTranspose = XMMatrixTranspose(XMMatrixInverse(nullptr, cameraLookAtMat));
    XMMATRIX cameraProjectMat = XMMatrixPerspectiveFovLH(RADIAN(cameraFovVertical), (float)settings.renderW / (float)settings.renderH, 0.001f, 100.0f);
    XMMATRIX cameraProjectViewMat = XMMatrixMultiply(cameraLookAtMat, cameraProjectMat);
    XMMATRIX cameraProjectViewInverseMat = XMMatrixInverse(nullptr, cameraProjectViewMat);
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC renderTextureSRVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Texture2D = {.MipLevels = 1}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC renderTextureUAVDesc = {.Format = d3d.renderTextureFormat, .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0, .PlaneSlice = 0}};
        D3D12_CONSTANT_BUFFER_VIEW_DESC renderInfoCBVDesc = {.BufferLocation = d3d.constantsBuffer->GetResource()->GetGPUVirtualAddress(), .SizeInBytes = align((uint)sizeof(struct RenderInfo), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT)};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasViewDesc = {.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .RaytracingAccelerationStructure = {.Location = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress()}};
        D3D12_SHADER_RESOURCE_VIEW_DESC tlasInstancesInfosDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.tlasInstancesInfosBuffer->GetSize() / sizeof(struct TLASInstanceInfo)), .StructureByteStride = sizeof(struct TLASInstanceInfo)}};
        D3D12_SHADER_RESOURCE_VIEW_DESC blasGeometriesInfosDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.blasGeometriesInfosBuffer->GetSize() / sizeof(struct BLASGeometryInfo)), .StructureByteStride = sizeof(struct BLASGeometryInfo)}};
        D3D12_SHADER_RESOURCE_VIEW_DESC collisionQueriesDesc = {.ViewDimension = D3D12_SRV_DIMENSION_BUFFER, .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .Buffer = {.NumElements = (uint)(d3d.collisionQueriesBuffer->GetSize() / sizeof(struct CollisionQuery)), .StructureByteStride = sizeof(struct CollisionQuery)}};
        D3D12_UNORDERED_ACCESS_VIEW_DESC collisionQueryResultsDesc = {.ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = {.NumElements = (uint)(d3d.collisionQueryResultsBuffer->GetSize() / sizeof(struct CollisionQueryResult)), .StructureByteStride = sizeof(struct CollisionQueryResult)}};

        D3DDescriptor renderTextureSRVDescriptor = d3dAppendSRVDescriptor(&renderTextureSRVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderTextureUAVDescriptor = d3dAppendUAVDescriptor(&renderTextureUAVDesc, d3d.renderTexture->GetResource());
        D3DDescriptor renderInfoDescriptor = d3dAppendCBVDescriptor(&renderInfoCBVDesc);
        D3DDescriptor tlasDescriptor = d3dAppendSRVDescriptor(&tlasViewDesc, nullptr);
        D3DDescriptor tlasInstancesInfosDescriptor = d3dAppendSRVDescriptor(&tlasInstancesInfosDesc, d3d.tlasInstancesInfosBuffer->GetResource());
        D3DDescriptor blasGeometriesInfosDescriptor = d3dAppendSRVDescriptor(&blasGeometriesInfosDesc, d3d.blasGeometriesInfosBuffer->GetResource());
        D3DDescriptor skyboxTextureDescriptor = d3dAppendSRVDescriptor(nullptr, skybox.hdriTexture->GetResource());
        D3DDescriptor imguiImageDescriptor = d3dAppendSRVDescriptor(nullptr, d3d.imguiImage->GetResource());
        D3DDescriptor collisionQueriesDescriptor = d3dAppendSRVDescriptor(&collisionQueriesDesc, d3d.collisionQueriesBuffer->GetResource());
        D3DDescriptor collisionQueryResultsDescriptor = d3dAppendUAVDescriptor(&collisionQueryResultsDesc, d3d.collisionQueryResultsUAVBuffer->GetResource());
    }
    {
        RenderInfo renderInfo = {
            .cameraViewMat = cameraLookAtMat,
            .cameraViewMatInverseTranspose = cameraLookAtMatInverseTranspose,
            .cameraProjMat = cameraProjectMat,
            .cameraProjViewMat = cameraProjectViewMat,
            //.cameraProjViewInverseMat = cameraProjectViewInverseMat,
        };
        assert(d3d.constantsBufferOffset == 0);
        memcpy(d3d.constantsBufferPtr + d3d.constantsBufferOffset, &renderInfo, sizeof(renderInfo));
        d3d.constantsBufferOffset += sizeof(renderInfo);
    }
    {
        static std::vector<ModelInstance*> skinnedModelInstances;
        skinnedModelInstances.resize(0);
        skinnedModelInstances.push_back(&player.modelInstance);
        for (GameObject& object : gameObjects) {
            if (object.modelInstance.skins.size() > 0 && object.modelInstance.animation) {
                skinnedModelInstances.push_back(&object.modelInstance);
            }
        }
        struct MeshSkinningInfo {
            ModelNode* meshNode;
            ModelInstanceMeshNode* instanceMeshNode;
            D3D12_GPU_VIRTUAL_ADDRESS matsBuffer;
        };
        static std::vector<D3D12_RESOURCE_BARRIER> verticeBufferBarriers;
        static std::vector<MeshSkinningInfo> meshSkinningInfos;
        verticeBufferBarriers.resize(0);
        meshSkinningInfos.resize(0);
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
        d3d.graphicsCmdList->SetPipelineState(d3d.vertexSkinning);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.vertexSkinningRootSig);
        d3d.graphicsCmdList->ResourceBarrier((uint)verticeBufferBarriers.size(), &verticeBufferBarriers[0]);
        for (MeshSkinningInfo& info : meshSkinningInfos) {
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(0, info.matsBuffer);
            d3d.graphicsCmdList->SetComputeRootShaderResourceView(1, info.meshNode->mesh->verticesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRootUnorderedAccessView(2, info.instanceMeshNode->verticesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetComputeRoot32BitConstant(3, (uint)info.meshNode->mesh->vertices.size(), 0);
            d3d.graphicsCmdList->Dispatch((uint)info.meshNode->mesh->vertices.size() / 32 + 1, 1, 1);
        }
        for (D3D12_RESOURCE_BARRIER& barrier : verticeBufferBarriers) std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier((uint)verticeBufferBarriers.size(), &verticeBufferBarriers[0]);
        static std::vector<D3D12_RESOURCE_BARRIER> blasBarriers;
        blasBarriers.resize(0);
        PIXSetMarker(d3d.graphicsCmdList, 0, "build BLAS");
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
                        .IndexBuffer = info.meshNode->mesh->indicesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.indicesBufferOffset * sizeof(uint),
                        .VertexBuffer = {.StartAddress = info.instanceMeshNode->verticesBuffer->GetResource()->GetGPUVirtualAddress() + primitive.verticesBufferOffset * sizeof(struct Vertex), .StrideInBytes = sizeof(struct Vertex)},
                    },
                });
            }
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD, .NumDescs = (uint)geometryDescs.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .pGeometryDescs = geometryDescs.data()};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasDesc = {.DestAccelerationStructureData = info.instanceMeshNode->blas->GetResource()->GetGPUVirtualAddress(), .Inputs = blasInputs, .ScratchAccelerationStructureData = info.instanceMeshNode->blasScratch->GetResource()->GetGPUVirtualAddress()};
            d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&blasDesc, 0, nullptr);
            blasBarriers.push_back(D3D12_RESOURCE_BARRIER{.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = info.instanceMeshNode->blas->GetResource()}});
        }
        d3d.graphicsCmdList->ResourceBarrier((uint)blasBarriers.size(), &blasBarriers[0]);
    }
    {
        {
            XMVECTOR translate = (player.position - player.camera.position).toXMVector();
            XMVECTOR rotate = XMQuaternionRotationRollPitchYaw(0, player.PitchYawRoll.y, 0);
            if (editor && editor->active) {
                translate = (player.spawnPosition - editor->camera.position).toXMVector();
                rotate = XMQuaternionRotationRollPitchYaw(0, 0, 0);
            }
            XMMATRIX transformMat = XMMatrixAffineTransformation(XMVectorSet(1, 1, 1, 0), XMVectorSet(0, 0, 0, 0), rotate, translate);
            addTLASInstance(player.modelInstance, transformMat, ObjectTypePlayer, 0);
        }
        for (uint objIndex = 0; objIndex < gameObjects.size(); objIndex++) {
            XMVECTOR translate = (-player.camera.position).toXMVector();
            if (editor && editor->active) {
                translate = (-editor->camera.position).toXMVector();
            }
            addTLASInstance(gameObjects[objIndex].modelInstance, XMMatrixTranslationFromVector(translate), ObjectTypeGameObject, objIndex);
        }

        addTLASInstance(modelInstanceCylinder, XMMatrixAffineTransformation(XMVectorSet(0.05f, player.movement.length(), 0.05f, 0), XMVectorSet(0, 0, 0, 0), quaternionBetween(float3(0, 1, 0), player.movement), (player.position - player.camera.position).toXMVector()), ObjectTypeNone, 0);

        assert(vectorSizeof(tlasInstancesBuildInfos) < d3d.tlasInstancesBuildInfosBuffer->GetSize());
        assert(vectorSizeof(tlasInstancesInfos) < d3d.tlasInstancesInfosBuffer->GetSize());
        assert(vectorSizeof(blasGeometriesInfos) < d3d.blasGeometriesInfosBuffer->GetSize());
        memcpy(d3d.tlasInstancesBuildInfosBufferPtr, tlasInstancesBuildInfos.data(), vectorSizeof(tlasInstancesBuildInfos));
        memcpy(d3d.tlasInstancesInfosBufferPtr, tlasInstancesInfos.data(), vectorSizeof(tlasInstancesInfos));
        memcpy(d3d.blasGeometriesInfosBufferPtr, blasGeometriesInfos.data(), vectorSizeof(blasGeometriesInfos));

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, .NumDescs = (uint)tlasInstancesBuildInfos.size(), .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY, .InstanceDescs = d3d.tlasInstancesBuildInfosBuffer->GetResource()->GetGPUVirtualAddress()};
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &prebuildInfo);
        assert(prebuildInfo.ResultDataMaxSizeInBytes < d3d.tlasBuffer->GetSize());
        assert(prebuildInfo.ScratchDataSizeInBytes < d3d.tlasScratchBuffer->GetSize());

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {.DestAccelerationStructureData = d3d.tlasBuffer->GetResource()->GetGPUVirtualAddress(), .Inputs = tlasInputs, .ScratchAccelerationStructureData = d3d.tlasScratchBuffer->GetResource()->GetGPUVirtualAddress()};
        PIXSetMarker(d3d.graphicsCmdList, 0, "build TLAS");
        d3d.graphicsCmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
        D3D12_RESOURCE_BARRIER tlasBarrier = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = d3d.tlasBuffer->GetResource()}};
        d3d.graphicsCmdList->ResourceBarrier(1, &tlasBarrier);
    }
    {
        if (mouseSelectX == UINT_MAX) {
            d3d.collisionQueriesBufferPtr[0] = {.rayDesc = {.origin = {0, 0, 0}, .min = 0, .dir = {0, 0, 0}, .max = 0}};
        } else {
            XMFLOAT4X4 cameraViewMat;
            XMFLOAT4X4 cameraProjMat;
            XMStoreFloat4x4(&cameraViewMat, cameraLookAtMatInverseTranspose);
            XMStoreFloat4x4(&cameraProjMat, cameraProjectMat);
            float2 pixelCoord = ((float2((float)mouseSelectX, (float)mouseSelectY) + 0.5f) / float2((float)settings.renderW, (float)settings.renderH)) * 2.0f - 1.0f;
            RayDesc rayDesc = {.origin = {cameraViewMat.m[0][3], cameraViewMat.m[1][3], cameraViewMat.m[2][3]}, .min = 0.0f, .max = FLT_MAX};
            float aspect = cameraProjMat.m[1][1] / cameraProjMat.m[0][0];
            float tanHalfFovY = 1.0f / cameraProjMat.m[1][1];
            rayDesc.dir = (float3(cameraViewMat.m[0][0], cameraViewMat.m[1][0], cameraViewMat.m[2][0]) * pixelCoord.x * tanHalfFovY * aspect) - (float3(cameraViewMat.m[0][1], cameraViewMat.m[1][1], cameraViewMat.m[2][1]) * pixelCoord.y * tanHalfFovY) + (float3(cameraViewMat.m[0][2], cameraViewMat.m[1][2], cameraViewMat.m[2][2]));
            rayDesc.dir = rayDesc.dir.normalize();
            d3d.collisionQueriesBufferPtr[0] = {.rayDesc = rayDesc, .instanceInclusionMask = 0xff & ~ObjectTypeNone};
        }

        d3d.collisionQueriesBufferPtr[1] = {.rayDesc = {.origin = player.position - player.camera.position, .min = 0.0f, .dir = player.movement.normalize(), .max = player.movement.length()}, .instanceInclusionMask = 0xff & ~(ObjectTypeNone | ObjectTypePlayer)};

        D3D12_RESOURCE_BARRIER collisionQueryResultsBarriers[2] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.collisionQueryResultsUAVBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.collisionQueryResultsUAVBuffer->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE}},
        };
        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsBarriers[0]);

        void* missIDs[1] = {d3d.collisionQueryMissID};
        void* hitGroupIDs[1] = {d3d.collisionQueryHitGroupID};
        D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(d3d.constantsBuffer->GetResource(), d3d.constantsBufferPtr, &d3d.constantsBufferOffset, d3d.collisionQueryRayGenID, missIDs, hitGroupIDs);
        dispatchDesc.Width = 2, dispatchDesc.Height = 1, dispatchDesc.Depth = 1;
        assert(d3d.constantsBufferOffset < d3d.constantsBuffer->GetSize());

        PIXSetMarker(d3d.graphicsCmdList, 0, "collisionQuery");
        d3d.graphicsCmdList->SetPipelineState1(d3d.collisionQuery);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.collisionQueryRootSig);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);

        d3d.graphicsCmdList->ResourceBarrier(1, &collisionQueryResultsBarriers[1]);
        d3d.graphicsCmdList->CopyBufferRegion(d3d.collisionQueryResultsBuffer->GetResource(), 0, d3d.collisionQueryResultsUAVBuffer->GetResource(), 0, d3d.collisionQueryResultsBuffer->GetSize());

        assert(SUCCEEDED(d3d.graphicsCmdList->Close()));
        d3d.graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.graphicsCmdList);
        d3d.collisionQueriesFenceValue += 1;
        d3d.graphicsQueue->Signal(d3d.collisionQueriesFence, d3d.collisionQueriesFenceValue);
        assert(SUCCEEDED(d3d.graphicsCmdList->Reset(d3d.graphicsCmdAllocator, nullptr)));
        d3d.graphicsCmdList->SetDescriptorHeaps(1, &d3d.cbvSrvUavDescriptorHeap);
    }
    {
        void* missIDs[2] = {d3d.renderScenePrimaryRayMissID, d3d.renderSceneSecondaryRayMissID};
        void* hitGroupIDs[2] = {d3d.renderScenePrimaryRayHitGroupID, d3d.renderSceneSecondaryRayHitGroupID};
        D3D12_DISPATCH_RAYS_DESC dispatchDesc = fillRayTracingShaderTable(d3d.constantsBuffer->GetResource(), d3d.constantsBufferPtr, &d3d.constantsBufferOffset, d3d.renderSceneRayGenID, missIDs, hitGroupIDs);
        dispatchDesc.Width = settings.renderW, dispatchDesc.Height = settings.renderH, dispatchDesc.Depth = 1;
        assert(d3d.constantsBufferOffset < d3d.constantsBuffer->GetSize());

        D3D12_RESOURCE_BARRIER renderTextureTransition = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
        d3d.graphicsCmdList->ResourceBarrier(1, &renderTextureTransition);

        PIXSetMarker(d3d.graphicsCmdList, 0, "renderScene");
        d3d.graphicsCmdList->SetPipelineState1(d3d.renderScene);
        d3d.graphicsCmdList->SetComputeRootSignature(d3d.renderSceneRootSig);
        d3d.graphicsCmdList->DispatchRays(&dispatchDesc);
    }
    {
        uint swapChainBackBufferIndex = d3d.swapChain->GetCurrentBackBufferIndex();
        D3D12_RESOURCE_BARRIER imageTransitions[] = {
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.swapChainImages[swapChainBackBufferIndex], .StateBefore = D3D12_RESOURCE_STATE_PRESENT, .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET}},
            {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, .Transition = {.pResource = d3d.renderTexture->GetResource(), .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, .StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE}},
        };
        d3d.graphicsCmdList->ResourceBarrier(countof(imageTransitions), imageTransitions);
        d3d.graphicsCmdList->OMSetRenderTargets(1, &d3d.swapChainImageRTVDescriptors[swapChainBackBufferIndex], false, nullptr);
        D3D12_VIEWPORT viewport = {0, 0, (float)settings.renderW, (float)settings.renderH, 0, 1};
        RECT scissor = {0, 0, (long)settings.renderW, (long)settings.renderH};
        d3d.graphicsCmdList->RSSetViewports(1, &viewport);
        d3d.graphicsCmdList->RSSetScissorRects(1, &scissor);
        d3d.graphicsCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        {
            PIXSetMarker(d3d.graphicsCmdList, 0, "postProcess");
            d3d.graphicsCmdList->SetPipelineState(d3d.postProcess);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.postProcessRootSig);
            d3d.graphicsCmdList->SetGraphicsRoot32BitConstant(0, settings.hdr, 0);
            d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
        }
        {
            shapeCircles.resize(0);
            shapeLines.resize(0);
            if (editor && editor->active) {
                if (editor->selectedObjectType == ObjectTypePlayer) {
                    if (player.modelInstance.model->skeletonRootNode) {
                        XMMatrixScaling(1, 1, -1);
                        player.modelInstance.transform.toMat();
                        modelTraverseNodesAndGetPositionAsCircles(player.modelInstance.model, player.modelInstance.model->skeletonRootNode, player.modelInstance.globalTransformMats, shapeCircles);
                    }
                } else if (editor->selectedObjectType == ObjectTypeGameObject) {
                    GameObject& gameObject = gameObjects[editor->selectedObjectIndex];
                    if (gameObject.modelInstance.model->skeletonRootNode) {
                        modelTraverseNodesAndGetPositionAsCircles(gameObject.modelInstance.model, gameObject.modelInstance.model->skeletonRootNode, gameObject.modelInstance.globalTransformMats, shapeCircles);
                    }
                }
            }
            //shapeCircles.push_back(ShapeCircle{.center = player.position - player.camera.position, .radius = 0.02f});
            //shapeLines.push_back(ShapeLine{.p0 = player.position - player.camera.position, .thickness = 0.01f, .p1 = player.position + float3(0, 1, 0) - player.camera.position});
            memcpy(d3d.shapeCirclesBufferPtr, shapeCircles.data(), shapeCircles.size() * sizeof(ShapeCircle));
            memcpy(d3d.shapeLinesBufferPtr, shapeLines.data(), shapeLines.size() * sizeof(ShapeLine));
            PIXSetMarker(d3d.graphicsCmdList, 0, "shapes");
            d3d.graphicsCmdList->SetPipelineState(d3d.shapes);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.shapesRootSig);
            uint constants[4] = {settings.renderW, settings.renderH, (uint)shapeCircles.size(), (uint)shapeLines.size()};
            d3d.graphicsCmdList->SetGraphicsRoot32BitConstants(0, countof(constants), constants, 0);
            d3d.graphicsCmdList->SetGraphicsRootShaderResourceView(1, d3d.shapeCirclesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->SetGraphicsRootShaderResourceView(2, d3d.shapeLinesBuffer->GetResource()->GetGPUVirtualAddress());
            d3d.graphicsCmdList->DrawInstanced(3, 1, 0, 0);
        }
        {
            PIXSetMarker(d3d.graphicsCmdList, 0, "imgui");
            d3d.graphicsCmdList->SetPipelineState(d3d.imgui);
            float blendFactor[] = {0, 0, 0, 0};
            d3d.graphicsCmdList->OMSetBlendFactor(blendFactor);
            d3d.graphicsCmdList->SetGraphicsRootSignature(d3d.imguiRootSig);
            uint constants[3] = {settings.renderW, settings.renderH, settings.hdr};
            d3d.graphicsCmdList->SetGraphicsRoot32BitConstants(0, countof(constants), constants, 0);
            D3D12_VERTEX_BUFFER_VIEW vertBufferView = {d3d.imguiVertexBuffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiVertexBuffer->GetSize(), sizeof(ImDrawVert)};
            D3D12_INDEX_BUFFER_VIEW indexBufferView = {d3d.imguiIndexBuffer->GetResource()->GetGPUVirtualAddress(), (uint)d3d.imguiIndexBuffer->GetSize(), DXGI_FORMAT_R16_UINT};
            assert(SUCCEEDED(d3d.imguiVertexBuffer->GetResource()->Map(0, nullptr, (void**)&d3d.imguiVertexBufferPtr)));
            assert(SUCCEEDED(d3d.imguiIndexBuffer->GetResource()->Map(0, nullptr, (void**)&d3d.imguiIndexBufferPtr)));
            d3d.graphicsCmdList->IASetVertexBuffers(0, 1, &vertBufferView);
            d3d.graphicsCmdList->IASetIndexBuffer(&indexBufferView);
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
        std::swap(imageTransitions[0].Transition.StateBefore, imageTransitions[0].Transition.StateAfter);
        d3d.graphicsCmdList->ResourceBarrier(1, &imageTransitions[0]);

        assert(SUCCEEDED(d3d.graphicsCmdList->Close()));
        d3d.graphicsQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&d3d.graphicsCmdList);

        assert(SUCCEEDED(d3d.swapChain->Present(0, 0)));

        d3d.renderDoneFenceValue += 1;
        d3d.graphicsQueue->Signal(d3d.renderDoneFence, d3d.renderDoneFenceValue);

        std::swap(d3d.renderTexture, d3d.renderTexturePrevFrame);
    }
}

typedef int (*gameUpdateLiveReloadProc)(void* gameState);
gameUpdateLiveReloadProc gameUpdateLiveReload = nullptr;

void updateGameLiveReloadProcs() {
    static HMODULE gameLiveReloadDLL = nullptr;
    static std::filesystem::path gameLiveReloadDLLPath = exeDir / "gameLiveReload.dll";
    static std::filesystem::path gameLiveReloadDLLCopyPath = exeDir / "gameLiveReloadCopy.dll";
    static std::filesystem::file_time_type prevLastWriteTime = {};
    std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(gameLiveReloadDLLPath);
    if (lastWriteTime > prevLastWriteTime) {
        prevLastWriteTime = lastWriteTime;
        FreeLibrary(gameLiveReloadDLL);
        CopyFileW(gameLiveReloadDLLPath.c_str(), gameLiveReloadDLLCopyPath.c_str(), false);
        gameLiveReloadDLL = LoadLibraryW(gameLiveReloadDLLCopyPath.c_str());
        assert(gameLiveReloadDLL);
        gameUpdateLiveReload = (gameUpdateLiveReloadProc)GetProcAddress(gameLiveReloadDLL, "gameUpdateLiveReload");
        assert(gameUpdateLiveReload);
    }
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    assert(SetCurrentDirectoryW(exeDir.c_str()));
    showConsole();
    settingsInit(saveDir / "settings.yaml");
    windowInit();
    windowShow();
    imguiInit();
    d3dInit();
    d3dApplySettings();
    loadSimpleAssets();
    worldInit(assetsDir / "worlds/world.yaml");
    worldReadSave(saveDir / "save.yaml");
    assert(QueryPerformanceFrequency(&perfFrequency));
    while (!quit) {
        QueryPerformanceCounter(&perfCounters[0]);
        ZoneScoped;
        updateGameLiveReloadProcs();
        d3dUpdateShaders();
        mouseDeltaRaw = {0, 0};
        mouseWheel = 0;
        controllerGetStateXInput();
        MSG windowMsg;
        while (PeekMessageA(&windowMsg, (HWND)window.hwnd, 0, 0, PM_REMOVE)) {
            TranslateMessage(&windowMsg);
            DispatchMessageA(&windowMsg);
        }
        update();
        render();
        frameCount += 1;
        FrameMark;
        QueryPerformanceCounter(&perfCounters[1]);
        frameTime = (double)(perfCounters[1].QuadPart - perfCounters[0].QuadPart) / (double)perfFrequency.QuadPart;
    }
    assert(SetCurrentDirectoryW(exeDir.c_str()));
    worldSave();
    worldWriteSave(saveDir / "save.yaml");
    settingsSave(saveDir / "settings.yaml");
    return EXIT_SUCCESS;
}
