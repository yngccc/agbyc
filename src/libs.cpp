#include "pch.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>

#include <ufbx/ufbx.c>

#include <stb/stb_ds.cpp>
#include <stb/stb_image.cpp>
#include <stb/stb_image_write.cpp>

#include <imgui/imgui.cpp>
#include <imgui/imgui_draw.cpp>
#include <imgui/imgui_widgets.cpp>
#include <imgui/imgui_tables.cpp>
#include <imgui/imgui_demo.cpp>
#include <imgui/imguizmo.cpp>

#include <d3d12ma/d3d12memalloc.cpp>

#define RYML_SINGLE_HDR_DEFINE_NOW
#include <rapidyaml/rapidyaml-0.8.0.hpp>

#define TRACY_ENABLE
#include <tracy/tracyclient.cpp>

#include <ozz/src_fused/ozz_base.cc>
#include <ozz/src_fused/ozz_options.cc>
#include <ozz/src_fused/ozz_animation.cc>
//#include <ozz/src_fused/ozz_animation_fbx.cc>
//#include <ozz/src_fused/ozz_animation_offline.cc>
//#include <ozz/src_fused/ozz_animation_tools.cc>
#include <ozz/src_fused/ozz_geometry.cc>
