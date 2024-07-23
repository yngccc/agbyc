#pragma once

#include <cassert>
#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <iostream>
#include <fstream>
#include <list>
#include <span>
#include <stack>
#include <streambuf>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory_resource>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <shellscalingapi.h>
#include <shlobj.h>
#include <cderr.h>
#include <commdlg.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <d3dx12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <d2d1_3.h>
#include <pix3.h>
#include <xinput.h>
#include <gameInput.h>

#define _XM_SSE4_INTRINSICS_
#include <directxmath.h>
#include <directxtex.h>
#include <directxcollision.h>

#undef near
#undef far

typedef DirectX::XMMATRIX XMMatrix;
typedef DirectX::XMVECTOR XMVector;
typedef DirectX::XMFLOAT3X3 XMFloat3x3;
typedef DirectX::XMFLOAT4X3 XMFloat4x3;
typedef DirectX::XMFLOAT4X4 XMFloat4x4;
using DirectX::XMVectorSet;
using DirectX::XMVectorGetX;
using DirectX::XMVectorGetY;
using DirectX::XMVectorGetZ;
using DirectX::XMVectorGetW;
using DirectX::XMVectorNegate;
using DirectX::XMVectorAdd;
using DirectX::XMVectorSubtract;
using DirectX::XMVectorMultiply;
using DirectX::XMVector3Normalize;
using DirectX::XMVector3Rotate;
using DirectX::XMVector3Transform;
using DirectX::XMVector3Unproject;
using DirectX::XMMatrixIdentity;
using DirectX::XMMatrixScaling;
using DirectX::XMMatrixAffineTransformation;
using DirectX::XMMatrixLookAtLH;
using DirectX::XMMatrixPerspectiveFovLH;
using DirectX::XMMatrixInverse;
using DirectX::XMMatrixTranspose;
using DirectX::XMStoreFloat3x3;
using DirectX::XMStoreFloat4x3;
using DirectX::XMStoreFloat4x4;
using DirectX::XMLoadFloat3x3;
using DirectX::XMLoadFloat4x3;
using DirectX::XMLoadFloat4x4;
using DirectX::XMQuaternionSlerp;
using DirectX::XMQuaternionNormalize;
using DirectX::XMQuaternionRotationRollPitchYaw;

#ifdef _DEBUG
#include <physx/pxphysicsapi.h>
#else
#define NDEBUG
#include <physx/pxphysicsapi.h>
#undef NDEBUG
#endif

using namespace physx;

//#include <intrin.h>
//#pragma intrinsic(_umul128)
#include "hashTable/unordered_dense.h"

#include <iconfontcppheaders/iconsfontawesome6.h>

#include <concurrentqueue/concurrentqueue.h>
