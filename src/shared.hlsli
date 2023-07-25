#define PI 3.14159265358979323846
#define FLT_MAX 3.402823466e+38F
#define UINT_MAX 0xffffffff

#define RENDER_TEXTURE_DESCRIPTOR ResourceDescriptorHeap[0]
#define ACCUMULATION_RENDER_TEXTURE_DESCRIPTOR ResourceDescriptorHeap[1]
#define RENDER_INFO_DESCRIPTOR ResourceDescriptorHeap[2]
#define BVH_DESCRIPTOR ResourceDescriptorHeap[3]
#define TLAS_INSTANCE_INFOS_DESCRIPTOR ResourceDescriptorHeap[4]
#define SKYBOX_TEXTURE_DESCRIPTOR ResourceDescriptorHeap[5]
#define READBACK_BUFFER_DESCRIPTOR ResourceDescriptorHeap[6]
#define IMGUI_TEXTURE_DESCRIPTOR ResourceDescriptorHeap[7]
#define COLLISION_QUERIES_DESCRIPTOR ResourceDescriptorHeap[8]
#define COLLISION_QUERY_RESULTS_DESCRIPTOR ResourceDescriptorHeap[9]

float sRGBToLinear(float srgb) {
	if (srgb <= 0.04045) {
		return srgb / 12.92;
	}
	else {
		return pow((srgb + 0.055) / 1.055, 2.4);
	}
}

float3 sRGBToLinear(float3 srgb) {
	return float3(sRGBToLinear(srgb.x), sRGBToLinear(srgb.y), sRGBToLinear(srgb.z));
}

float linearToSRGB(float rgb) {
	if (rgb <= 0.0031308) {
		return rgb * 12.92;
	}
	else {
		return 1.055 * pow(rgb, 1.0 / 2.4) - 0.055;
	}
}

float3 linearToSRGB(float3 rgb) {
	return float3(linearToSRGB(rgb.x), linearToSRGB(rgb.y), linearToSRGB(rgb.z));
}

float3 bt709To2020(float3 rgb) {
	const float3x3 mat = { 
		0.6274, 0.3293, 0.0433,
		0.0691, 0.9195, 0.0114, 
		0.0164, 0.0880, 0.8956 
	};
	return mul(mat, rgb);
}

float3 linearToPQ(float3 rgb) {
	const float m1 = 0.1593017578125;
	const float m2 = 78.84375;
	const float c1 = 0.8359375;
	const float c2 = 18.8515625;
	const float c3 = 18.6875;
	const float3 rgbPow = pow(rgb, m1);
	return pow((c1 + c2 * rgbPow) / (1.0 + c3 * rgbPow), m2);
}

float3 barycentricsLerp(in float2 barycentrics, in float3 vertAttrib0, in float3 vertAttrib1, in float3 vertAttrib2) {
	return vertAttrib0 + barycentrics.x * (vertAttrib1 - vertAttrib0) + barycentrics.y * (vertAttrib2 - vertAttrib0);
}

uint xorshift(inout uint rngState) {
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

uint jenkinsHash(uint x) {
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;
	return x;
}

float uintToFloat(uint x) {
	return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

uint randInit(uint2 resolution, uint2 pixelIndex, uint frameCount) {
	uint seed = dot(pixelIndex, uint2(1, resolution.x)) ^ jenkinsHash(frameCount);
	return jenkinsHash(seed);
}

float rand(inout uint rngState) {
	return uintToFloat(xorshift(rngState));
}
