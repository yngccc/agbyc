#include <windows.h>
#include <filesystem>
#include <cassert>

#include "stb/stb_image_write.h"

std::filesystem::path exeDir = [] {
    wchar_t buf[512];
    DWORD n = GetModuleFileNameW(nullptr, buf, sizeof(buf) / sizeof(wchar_t));
    assert(n < (sizeof(buf) / sizeof(wchar_t)));
    std::filesystem::path path(buf);
    return path.parent_path();
}();

struct Pixel {
    uint8_t r, g, b, a;
};

static const int n = 1024;
static Pixel image[n * n];
static Pixel backgroundColor = Pixel{220, 220, 220, 255};
static Pixel lineColor = Pixel{90, 90, 90, 255};
static Pixel lineColor2 = Pixel{180, 180, 180, 255};
static int lineThickness = 8;
static int lineThicknessHalf = 4;

int main(int argc, char** argv) {
    assert(SetCurrentDirectoryW(exeDir.c_str()));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = backgroundColor;
        }
    }
    {
        int rowOffset = n / 4 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[(rowOffset + i) * n + j] = lineColor2;
            }
        }
        rowOffset = (n / 4) * 3 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[(rowOffset + i) * n + j] = lineColor2;
            }
        }
    }
    {
        int columnOffset = n / 4 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[j * n + columnOffset + i] = lineColor2;
            }
        }
        columnOffset = (n / 4) * 3 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[j * n + columnOffset + i] = lineColor2;
            }
        }
    }
    {
        int rowOffset = 0;
        for (int i = 0; i < lineThicknessHalf; i++) {
            for (int j = 0; j < n; j++) {
                image[(rowOffset + i) * n + j] = lineColor;
            }
        }
        rowOffset = n / 2 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[(rowOffset + i) * n + j] = lineColor;
            }
        }
        rowOffset = n - lineThicknessHalf;
        for (int i = 0; i < lineThicknessHalf; i++) {
            for (int j = 0; j < n; j++) {
                image[(rowOffset + i) * n + j] = lineColor;
            }
        }
    }
    {
        int columnOffset = 0;
        for (int i = 0; i < lineThicknessHalf; i++) {
            for (int j = 0; j < n; j++) {
                image[j * n + columnOffset + i] = lineColor;
            }
        }
        columnOffset = n / 2 - lineThicknessHalf;
        for (int i = 0; i < lineThickness; i++) {
            for (int j = 0; j < n; j++) {
                image[j * n + columnOffset + i] = lineColor;
            }
        }
        columnOffset = n - lineThicknessHalf;
        for (int i = 0; i < lineThicknessHalf; i++) {
            for (int j = 0; j < n; j++) {
                image[j * n + columnOffset + i] = lineColor;
            }
        }
    }

    int result = stbi_write_png("grid.png", n, n, 4, image, 0);
    assert(result);
}