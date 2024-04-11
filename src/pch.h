#pragma once

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
#include <cassert>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <shellscalingapi.h>
#include <shlobj.h>
#include <cderr.h>
#include <commdlg.h>

#include <d3dx12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <d3d11on12.h>
#include <dwrite.h>
#include <d2d1_3.h>
#include <pix3.h>

#define _XM_SSE4_INTRINSICS_
#include <directxmath.h>
#include <directxtex.h>
#include <directxcollision.h>

#include <xinput.h>

#ifdef _DEBUG
#include <physx/pxphysicsapi.h>
#else
#define NDEBUG
#include <physx/pxphysicsapi.h>
#undef NDEBUG
#endif

using namespace DirectX;
using namespace physx;
