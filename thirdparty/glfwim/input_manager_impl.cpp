#include "input_manager.hpp"
#include "gui/gui_manager.h"
using namespace glfwim;

bool InputManager::isKeyboardCaptured()
{
    return theGUIManager.isKeyboardCaptured();
}

bool InputManager::isMouseCaptured()
{
    return theGUIManager.isMouseCaptured();
}