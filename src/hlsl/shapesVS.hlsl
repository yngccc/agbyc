#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig "RootFlags(0), RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1)"

uint4 constants : register(b0);
StructuredBuffer<ShapeCircle> shapeCircles : register(t0);
StructuredBuffer<ShapeLine> shapeLines : register(t1);

float drawCircle(in float2 center, in float radius, in uint2 pixelIndex, in uint2 renderTextureSize) {
    float d = distance(float2(pixelIndex), float2(renderTextureSize) * center);
    if (d <= renderTextureSize.y * radius) {
        return 1;
    } else{
        return 0;
    }
}

float drawLine(in float2 p0, in float2 p1, in float thickness, in uint2 pixelIndex, in uint2 renderTextureSize) {
    p0 *= (float2) renderTextureSize;
    p1 *= (float2) renderTextureSize;
    float2 l = p1 - p0;
    float2 v = (float2) pixelIndex - p0;
    float2 vl = l * (dot(v, l) / dot(l, l));
    if (dot(vl, l) < 0) {
        return 0;
    }
    if (length(vl) > length(l)) {
        return 0;
    }
    if (length(vl - v) > thickness * renderTextureSize.y) {
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
    uint2 renderTargetSize = constants.xy;
    uint circleCount = constants.z;
    uint lineCount = constants.w;
    uint2 pixelIndex = renderTargetSize * vsOutput.texCoord;
    float v = 0;
    for (uint circleIndex = 0; circleIndex < circleCount; circleIndex++) {
        ShapeCircle c = shapeCircles[circleIndex];
        v += drawCircle(c.center, c.radius, pixelIndex, renderTargetSize);
    }
    for (uint lineIndex = 0; lineIndex < lineCount; lineIndex++) {
        ShapeLine l = shapeLines[lineIndex];
        v += drawLine(l.p0, l.p1, l.thickness, pixelIndex, renderTargetSize);
    }
    v = saturate(v);
    return float4(1, 1, 1, v * 0.8);
}
