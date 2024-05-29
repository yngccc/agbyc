#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig \
"RootFlags(0)," \
"RootConstants(num32BitConstants=1, b0)," \
"DescriptorTable(SRV(t0), SRV(t1), visibility = SHADER_VISIBILITY_PIXEL)," \
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

uint flags : register(b0);
Texture2D<float3> renderTexture : register(t0);
Texture2D<float4> directWriteImage : register(t1);
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
    float3 output;
    float3 renderTextureColor = renderTexture.Sample(textureSampler, vsOutput.texCoord);
    bool directWrite = flags & CompositeFlagDirectWrite;
    if (directWrite) {
        float4 directWriteColor = directWriteImage.Sample(textureSampler, vsOutput.texCoord);
        output = renderTextureColor * (1 - directWriteColor.a) + directWriteColor.rgb * directWriteColor.a;
    } else {
        output = renderTextureColor;
    }
    bool hdr = flags & CompositeFlagHDR;
    if (hdr) {
        output *= HDR_SCALE_FACTOR;
        output = linearToPQ(output);
    } else {
        output = linearToSRGB(output);
    }
    return float4(output, 0);
}
