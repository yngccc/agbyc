#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig \
"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

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
float4 pixelShader(VSOutput vsOutput) : SV_TARGET {
    RENDER_INFO_DESCRIPTOR(renderInfo);
    RENDER_TEXTURE_SRV_DESCRIPTOR(renderTexture);
    float4 output = renderTexture.Sample(renderTextureSampler, vsOutput.texCoord);
    if (renderInfo.hdr) {
        output.rgb *= HDR_SCALE_FACTOR;
        output.rgb = linearToPQ(output.rgb);
    }
    else {
        output.rgb = linearToSrgb(output.rgb);
    }
    return output;
}
