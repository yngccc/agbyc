#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), "\
"RootConstants(num32BitConstants=4, b0), "\
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

uint4 constants : register(b0);
sampler textureSampler : register(s0);

struct VSOutput {
    float4 translate : SV_POSITION;
    float2 texCoord : TEXCOORD;
    float4 color : COLOR;
};

[RootSignature(rootSig)]
VSOutput vertexShader(float2 translate : POSITION, float2 texCoord : TEXCOORD, float4 color : COLOR) {
    uint2 resolution = constants.xy;
    VSOutput output;
    output.translate.x = (translate.x / resolution.x) * 2 - 1;
    output.translate.y = -((translate.y / resolution.y) * 2 - 1);
    output.translate.zw = float2(0, 1);
    output.texCoord = texCoord;
    output.color = color;
    return output;
}

[RootSignature(rootSig)]
float4 pixelShader(VSOutput vsOutput) : SV_TARGET{
	IMGUI_IMAGE_DESCRIPTOR(imguiImage);
    float4 output = vsOutput.color * imguiImage.Sample(textureSampler, vsOutput.texCoord);
	bool hdr = constants.z;
    if (hdr) {
        output.rgb = srgbToLinear(output.rgb);
        output.rgb = bt709To2020(output.rgb);
        output.rgb *= HDR_SCALE_FACTOR;
        output.rgb = linearToPQ(output.rgb);
    }
    return output;
}
