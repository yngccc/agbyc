#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1)"

uint4 constants : register(b0);
StructuredBuffer<ShapeCircle> shapeCircles : register(t0);
StructuredBuffer<ShapeLine> shapeLines : register(t1);

float drawCircle(in float3 center, in float radius, in uint2 pixelIndex, in uint2 renderTargetSize, in float4x4 cameraProjViewMat) {
    float4 ndc = mul(cameraProjViewMat, float4(center, 1));
    ndc.y = -ndc.y;
    ndc /= ndc.w;
    if (ndc.x >= -1.0 && ndc.x <= 1.0 && ndc.y >= -1.0 && ndc.y <= 1.0 && ndc.z >= -1.0 && ndc.z <= 1.0) {
        ndc = (ndc + 1.0) * 0.5;
        float2 centerPixel = renderTargetSize * ndc.xy;
        float d = distance(float2(pixelIndex), centerPixel);
        if (d <= renderTargetSize.y * radius) {
            return 1;
        } else{
            return 0;
        }
    } else{
        return 0;
    }
}

float drawLine(in float3 p0, in float3 p1, in float thickness, in uint2 pixelIndex, in uint2 renderTargetSize, in float4x4 cameraProjViewMat) {
    float4 p_0 = mul(cameraProjViewMat, float4(p0, 1));
    float4 p_1 = mul(cameraProjViewMat, float4(p1, 1));
    p_0.y = -p_0.y;
    p_1.y = -p_1.y;
    p_0 /= p_0.w;
    p_1 /= p_1.w;
    p_0.xy = clamp(p_0.xy, float2(-1, -1), float2(1, 1));
    p_1.xy = clamp(p_1.xy, float2(-1, -1), float2(1, 1));
    p_0.xy = (p_0.xy + 1.0) * 0.5;
    p_1.xy = (p_1.xy + 1.0) * 0.5;
    p_0.xy *= (float2)renderTargetSize;
    p_1.xy *= (float2)renderTargetSize;
    float2 l = p_1.xy - p_0.xy;
    float2 v = (float2)pixelIndex - p_0.xy;
    float2 vl = l * (dot(v, l) / dot(l, l));
    if (dot(vl, l) < 0) {
        return 0;
    }
    if (length(vl) > length(l)) {
        return 0;
    }
    if (length(vl - v) > thickness * renderTargetSize.y) {
        return 0;
    }
    return 1;
}

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
    RENDER_INFO_DESCRIPTOR(renderInfo);
    uint2 renderTargetSize = constants.xy;
    uint circleCount = constants.z;
    uint lineCount = constants.w;
    uint2 pixelIndex = renderTargetSize * vsOutput.texCoord;
    float v = 0;
    for (uint circleIndex = 0; circleIndex < circleCount; circleIndex++) {
        ShapeCircle c = shapeCircles[circleIndex];
        v += drawCircle(c.center, c.radius, pixelIndex, renderTargetSize, renderInfo.cameraProjViewMat);
    }
    for (uint lineIndex = 0; lineIndex < lineCount; lineIndex++) {
        ShapeLine l = shapeLines[lineIndex];
        v += drawLine(l.p0, l.p1, l.thickness, pixelIndex, renderTargetSize, renderInfo.cameraProjViewMat);
    }
    v = saturate(v);
    return float4(1, 1, 1, v * 0.8);
}
