#include "common.h"

extern "C" {

__declspec(dllexport) 
void gameUpdateLive(ImGuiContext* imguiContext, Controller& controller, int2 mouseDeltaRaw, float mouseSensitivity, float controllerSensitivity, float frameTime, Player& player) {
    ImGui::SetCurrentContext(imguiContext);

    float3 moveDir = {0, 0, 0};
    bool jump = false;
    {
        float3 forwardDir = player.camera.lookAt - player.camera.position;
        forwardDir.y = 0;
        forwardDir = forwardDir.normalize();
        float3 sideDir = forwardDir.cross(float3(0, 1, 0));
        if (ImGui::IsKeyDown(ImGuiKey_W)) moveDir += forwardDir;
        if (ImGui::IsKeyDown(ImGuiKey_S)) moveDir += -forwardDir;
        if (ImGui::IsKeyDown(ImGuiKey_A)) moveDir += sideDir;
        if (ImGui::IsKeyDown(ImGuiKey_D)) moveDir += -sideDir;
        moveDir += forwardDir * controller.lsY;
        moveDir += sideDir * -controller.lsX;
        moveDir = moveDir.normalize();

        jump = ImGui::IsKeyPressed(ImGuiKey_Space);
        if (jump) { ImGui::DebugLog("jump\n"); }
    }

    if (moveDir == float3{0, 0, 0}) {
        player.state = PlayerStateIdle;
        player.movement = {0, 0, 0};
        player.modelInstance.animation = &player.modelInstance.model->animations[player.idleAnimationIndex];
    } else {
        if (!ImGui::IsKeyDown(ImGuiKey_LeftShift) && sqrtf(controller.lsX * controller.lsX + controller.lsY * controller.lsY) < 0.8f) {
            player.state = PlayerStateWalk;
            player.movement = moveDir * player.walkSpeed * (float)frameTime;
            player.modelInstance.animation = &player.modelInstance.model->animations[player.walkAnimationIndex];
        } else {
            player.state = PlayerStateRun;
            player.movement = moveDir * player.runSpeed * (float)frameTime;
            player.modelInstance.animation = &player.modelInstance.model->animations[player.runAnimationIndex];
        }
    }
}

}