#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig \
"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
"RootConstants(num32BitConstants=4, b0), "\
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

uint4 constants : register(b0);
sampler renderTextureSampler : register(s0);

struct VSOutput {
    float2 texCoord : TEXCOORD;
    float4 position : SV_POSITION;
};

[RootSignature(rootSig)]
VSOutput vertexShader(uint vertexID : SV_VertexID) {
    VSOutput output;
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(output.texCoord * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}

[RootSignature(rootSig)]
float4 pixelShader(VSOutput vsOutput) : SV_TARGET{
    RENDER_TEXTURE_SRV_DESCRIPTOR(renderTexture);
    float4 output = renderTexture.Sample(renderTextureSampler, vsOutput.texCoord);
    bool hdr = constants.x;
    if (hdr) {
        output.rgb *= HDR_SCALE_FACTOR;
        output.rgb = linearToPQ(output.rgb);
    } else{
        output.rgb = linearToSRGB(output.rgb);
    }
    return output;
}
