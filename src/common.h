#include <cstdint>
#include <format>
#include <string>
#include <vector>

#define _XM_SSE4_INTRINSICS_
#include <directxmath.h>
using namespace DirectX;

typedef int8_t int8;
typedef int16_t int16;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint64_t uint64;
typedef uint32_t uint;

static const float euler = 2.71828182845904523536f;
static const float pi = 3.14159265358979323846f;
static const float sqrt2 = 1.41421356237309504880f;

#define KILOBYTES(n) (1024 * (n))
#define MEGABYTES(n) (1024 * 1024 * (n))
#define GIGABYTES(n) (1024 * 1024 * 1024 * (n))
#define RADIAN(d) (d * (pi / 180.0f))
#define DEGREE(r) (r * (180.0f / pi))

template <typename T, uint N>
constexpr uint countof(const T (&)[N]) { return N; }

template <typename T>
uint64 vectorSizeof(const std::vector<T>& v) { return v.size() * sizeof(T); }

template <typename T, typename T2>
T align(T x, T2 n) {
    T remainder = x % (T)n;
    return remainder == 0 ? x : x + ((T)n - remainder);
}

bool getBit(uint n, uint index) {
    return (n >> index) & 1;
}

uint setBit(uint n, uint index) {
    return n |= (1 << index);
}

uint unsetBit(uint n, uint index) {
    return n &= ~(1 << index);
}

uint toggleBit(uint n, uint index) {
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
    uint x = 0, y = 0, z = 0, w = 0;
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

struct float2 {
    float x = 0, y = 0;

    float2() = default;
    float2(float x, float y) : x(x), y(y) {}
    bool operator==(float2 v) const { return x == v.x && y == v.y; }
    bool operator!=(float2 v) const { return x != v.x || y != v.y; }
    float2 operator+(float v) const { return float2(x + v, y + v); }
    float2 operator+(float2 v) const { return float2(x + v.x, y + v.y); }
    float2 operator-(float v) const { return float2(x - v, y - v); }
    float2 operator-(float2 v) const { return float2(x - v.x, y - v.y); }
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
    float3 operator+(float3 v) const { return float3(x + v.x, y + v.y, z + v.z); }
    void operator+=(float3 v) { x += v.x, y += v.y, z += v.z; }
    float3 operator-() const { return float3(-x, -y, -z); }
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
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, 0); }
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
    float3 xyz() const { return float3(x, y, z); }
    XMVECTOR toXMVector() const { return XMVectorSet(x, y, z, w); }
    std::string toString() const { return std::format("[{}, {}, {}, {}]", x, y, z, w); }
};

#define scale  10000.0f // 0.1mm precision
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

float3 lerp(const float3& a, const float3& b, float t) { return a + ((b - a) * t); };

float4 slerp(const float4& a, const float4& b, float t) { return float4(XMQuaternionSlerp(a.toXMVector(), b.toXMVector(), t)); };

std::string toString(const XMVECTOR& vec) { return std::format("|{:+.3f}, {:+.3f}, {:+.3f}, {:+.3f}|\n", XMVectorGetX(vec), XMVectorGetY(vec), XMVectorGetZ(vec), XMVectorGetW(vec)); }

std::string toString(const XMMATRIX& mat) {
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
