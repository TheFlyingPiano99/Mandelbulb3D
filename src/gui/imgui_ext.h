#ifndef IMGUI_EXT_H
#define IMGUI_EXT_H

namespace ImGui {
	void VerticalSeparator();
	void BeginGroupPanel(const char* name, const ImVec2& size = ImVec2(0.0f, 0.0f));
	void EndGroupPanel();
	bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size = -1.0f);
	bool DragUInt32(const char* label, unsigned int* v, float v_speed = 1.0f, unsigned int v_min = 0, unsigned int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0);
}

namespace ImGuizmo
{
	bool Manipulate(OPERATION operation, MODE mode, float* matrix, float* deltaMatrix = nullptr, const float* snap = nullptr, const float* localBounds = nullptr, const float* boundsSnap = nullptr);
	void ResetGlobalID();
	void UniqueID();
}

#endif//IMGUI_EXT_H
