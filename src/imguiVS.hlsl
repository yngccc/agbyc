#include "shared.hlsli"
#include "sceneStructs.h"

#define rootSig \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
"StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"

sampler textureSampler : register(s0);

struct VSOutput {
	float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD;
	float4 color : COLOR;
};

[RootSignature(rootSig)]
VSOutput vertexShader(float2 position : POSITION, float2 texCoord : TEXCOORD, float4 color : COLOR) {
	ConstantBuffer<RenderInfo> renderInfo = RENDER_INFO_DESCRIPTOR;
	VSOutput output;
	output.position.x = (position.x / renderInfo.resolution.x) * 2 - 1;
	output.position.y = -((position.y / renderInfo.resolution.y) * 2 - 1);
	output.position.zw = float2(0, 1);
	output.texCoord = texCoord;
	output.color = color;
	return output;
}

[RootSignature(rootSig)]
float4 pixelShader(VSOutput vsOutput) : SV_TARGET {
	ConstantBuffer<RenderInfo> renderInfo = RENDER_INFO_DESCRIPTOR;
	Texture2D<float4> imguiTexture = IMGUI_TEXTURE_DESCRIPTOR;
	float4 output = vsOutput.color * imguiTexture.Sample(textureSampler, vsOutput.texCoord);
	if (renderInfo.hdr) {
		output.rgb = sRGBToLinear(output.rgb);
		output.rgb = bt709To2020(output.rgb);
		output.rgb *= (100.0 / 10000.0);
		output.rgb = linearToPQ(output.rgb);
	}
	return output;
}
