#include "shared.hlsli"
#include "../structsHLSL.h"

#define rootSig "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), SRV(t0), SRV(t1), RootConstants(num32BitConstants=2, b0)"

StructuredBuffer<ShapeCircle> shapeCircles : register(t0);
StructuredBuffer<ShapeLine> shapeLines : register(t1);
uint2 counts : register(b0);

float drawCircle(in float2 center, in float radius, in uint2 pixelIndex, in uint2 renderTextureSize) {
    float d = distance(float2(pixelIndex), float2(renderTextureSize) * center);
    if (d <= renderTextureSize.y * radius) {
        return 1;
    } else {
        return 0;
    }
}

float drawLine(in float2 p0, in float2 p1, in float thickness, in uint2 pixelIndex, in uint2 renderTextureSize) {
    p0 *= (float2)renderTextureSize;
    p1 *= (float2)renderTextureSize;
    float2 l = p1 - p0;
    float2 v = (float2)pixelIndex - p0;
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

[RootSignature(rootSig)]
[numthreads(16, 16, 1)]
void computeShader(uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupID : SV_GroupID, uint2 groupThreadID : SV_GroupThreadID) {
    RENDER_TEXTURE_UAV_DESCRIPTOR(renderTexture);
    uint2 renderTextureSize;
    renderTexture.GetDimensions(renderTextureSize.x, renderTextureSize.y);
    if (dispatchThreadID.x >= renderTextureSize.x || dispatchThreadID.y >= renderTextureSize.y) {
        return;
    }
    float g = 0;
    //g += drawCircle(float2(0.5, 0.5), 0.1, dispatchThreadID, renderTextureSize);
    g += drawLine(float2(0.6, 0.3), float2(0.5, 0.50), 0.001, dispatchThreadID, renderTextureSize);
    renderTexture[dispatchThreadID].rgb += float3(0, g, 0);

    uint circleCount, circleStride;
    uint lineCount, lineStride;
    shapeCircles.GetDimensions(circleCount, circleStride);
    shapeLines.GetDimensions(lineCount, lineStride);
    
}
