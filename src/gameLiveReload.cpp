#include "common.h"

extern "C" {

__declspec(dllexport) int gameUpdateLiveReload(void* gameState) {
    printf("banana\n");
    return 0;
}

}