#include "shared.hlsli"
#include "sceneStructs.h"

#define rootSig \
"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

sampler renderTextureSampler : register(s0);

struct VSOutput {
	float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD;
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
	ConstantBuffer<RenderInfo> renderInfo = RENDER_INFO_DESCRIPTOR;
	Texture2D<float4> renderTexture = RENDER_TEXTURE_DESCRIPTOR;
	float4 output = renderTexture.Sample(renderTextureSampler, vsOutput.texCoord);
	if (renderInfo.hdr) {
		output.rgb /= 10; // 10 = 10000 nits
		output.rgb = linearToPQ(output.rgb);
	}
	else {
		output.rgb = linearToSRGB(output.rgb);
	}
	return output;
}
