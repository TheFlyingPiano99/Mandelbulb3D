#include "scene/game_scene.h"
#include "gui/imgui_ext.h"

void GameScene::updateGui()
{
	ImGui::Begin("Outliner");

	ImGui::Text("Objects");
	for (int i = 0; auto& it : gameObjects) {
		ImGui::PushID(i);
		if (ImGui::RadioButton(it.name.c_str(), i == selectedObjectIndex)) selectedObjectIndex = i;
		if (i++ == selectedObjectIndex) {
			ImGuizmo::UniqueID();
			glm::mat4 m = it.transform.M();
			if (ImGuizmo::Manipulate(ImGuizmo::OPERATION::TRANSLATE, ImGuizmo::MODE::LOCAL, &m[0][0])) {
				it.transform.set(m);
			}
		}
		ImGui::PopID();
	}

	ImGui::Separator();

	ImGui::Text("Point lights");
	for (int i = 0; auto & it : pointLights) {
		ImGui::PushID(i);
		if (ImGui::RadioButton(it.name.c_str(), i == selectedPointLightIndex)) selectedPointLightIndex = i;
		if (i++ == selectedPointLightIndex) {
			ImGuizmo::UniqueID();
			glm::mat4 m = glm::translate(it.position);
			if (ImGuizmo::Manipulate(ImGuizmo::OPERATION::TRANSLATE, ImGuizmo::MODE::LOCAL, &m[0][0])) {
				glm::vec3 tr, rot, sc;
				ImGuizmo::DecomposeMatrixToComponents(&m[0][0], &tr[0], &rot[0], &sc[0]);
				it.position = tr;
			}
		}
		ImGui::PopID();
	}

	ImGui::End();
}