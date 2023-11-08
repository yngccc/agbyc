#include "shared.hlsli"
#include "../sceneStructs.h"

#define rootSig \
"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)"

StructuredBuffer<float4x4> skinMatsBuffer;
StructuredBuffer<Vertex> verticesBuffer;
RWStructuredBuffer<Vertex> verticesBufferWrite;

[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID) {
    uint vertexIndex = groupID.x * 32 + groupThreadID.x;
    Vertex vertex = verticesBuffer[vertexIndex];
    float4x4 jointMat = 
        skinMatsBuffer[vertex.joints[0]] * vertex.jointWeights[0] + 
        skinMatsBuffer[vertex.joints[1]] * vertex.jointWeights[1] + 
        skinMatsBuffer[vertex.joints[2]] * vertex.jointWeights[2] + 
        skinMatsBuffer[vertex.joints[3]] * vertex.jointWeights[3];
    vertex.position = mul(jointMat, float4(vertex.position, 0)).xyz;
    vertex.normal = mul(jointMat, float4(vertex.normal, 0)).xyz;
    vertex.normal = normalize(vertex.normal);
    verticesBufferWrite[vertexIndex] = vertex;
}