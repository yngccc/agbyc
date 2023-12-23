#include "shared.hlsli"
#include "../sharedStructs.h"

#define rootSig "RootFlags(0), SRV(t0), SRV(t1), UAV(u0), RootConstants(num32BitConstants=1, b0)"

StructuredBuffer<float4x4> skinMatsBuffer : register(t0);
StructuredBuffer<Vertex> verticesBufferSrc : register(t1);
RWStructuredBuffer<Vertex> verticesBufferDst : register(u0);
uint verticeCount : register(b0);

[RootSignature(rootSig)]
[numthreads(128, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID) {
    uint vertexIndex = groupID.x * 128 + groupThreadID.x;
    if (vertexIndex < verticeCount) {
        Vertex vertex = verticesBufferSrc[vertexIndex];
        float4x4 jointMat = 
            skinMatsBuffer[vertex.joints[0]] * vertex.jointWeights[0] + 
            skinMatsBuffer[vertex.joints[1]] * vertex.jointWeights[1] + 
            skinMatsBuffer[vertex.joints[2]] * vertex.jointWeights[2] + 
            skinMatsBuffer[vertex.joints[3]] * vertex.jointWeights[3];
        vertex.position = mul(float4(vertex.position, 1), jointMat).xyz;
        vertex.normal = normalize(mul(vertex.normal, float3x3(jointMat[0].xyz, jointMat[1].xyz, jointMat[2].xyz)));
        verticesBufferDst[vertexIndex] = vertex;
    }
}