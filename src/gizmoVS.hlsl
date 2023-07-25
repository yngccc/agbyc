#include "shared.hlsli"

#define rootSig \
"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)"

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
	return float4(0, 0, 0, 0);
}
