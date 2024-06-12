#include "shared.h"

#define rootSig \
"RootFlags(0)," \
"RootConstants(b0, num32BitConstants = 1)," \
"DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)," \
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

uint flags : register(b0);
Texture2D<float3> renderTexture : register(t0);
sampler textureSampler : register(s0);

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
float4 pixelShader(VSOutput vsOutput) : SV_TARGET {
    float3 output = renderTexture.Sample(textureSampler, vsOutput.texCoord);
    bool hdr = flags & CompositeFlagHDR;
    if (hdr) {
        output *= HDR_SCALE_FACTOR;
        output = linearToPQ(output);
    } else {
        output = linearToSRGB(output);
    }
    return float4(output, 0);
}
