#include "input_manager.h"
#include <GLFW/glfw3.h>

namespace glfwim {
    PerFrameMonitorData::PerFrameMonitorData(const PerFrameMonitorData& that) {
        videoMode = std::make_unique<GLFWvidmode>(*that.videoMode);
        posX = that.posX;
        posY = that.posY;
        workAreaX = that.workAreaX;
        workAreaY = that.workAreaY;
        workAreaW = that.workAreaW;
        workAreaH = that.workAreaH;
        scaleX = that.scaleX;
        scaleY = that.scaleY;
    }

    PerFrameMonitorData& PerFrameMonitorData::operator=(const PerFrameMonitorData& that) {
        videoMode = std::make_unique<GLFWvidmode>(*that.videoMode);
        posX = that.posX;
        posY = that.posY;
        workAreaX = that.workAreaX;
        workAreaY = that.workAreaY;
        workAreaW = that.workAreaW;
        workAreaH = that.workAreaH;
        scaleX = that.scaleX;
        scaleY = that.scaleY;

        return *this;
    }

    void InputManager::initialize(GLFWwindow* window) {
        mainThreadId = std::this_thread::get_id();
        keyStates.resize(GLFW_KEY_LAST, -1);
        this->window = window;

        glfwSetWindowUserPointer(window, this);
	
		glfwSetDropCallback(window, [](auto window, int count, const char** paths) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            std::vector<std::string> vpaths;
            vpaths.reserve(count);
            for (int i = 0; i < count; ++i) {
                vpaths.push_back(paths[i]);
            }
            inputManager.pathDropEventQueue.emplace(std::move(vpaths));
        });

        glfwSetKeyCallback(window, [](auto window, int key, int scancode, int action, int mods) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.keyEventQueue.emplace(key, scancode, Modifier{mods}, Action{action});
        });

        glfwSetMouseButtonCallback(window, [](auto window, int button, int action, int mods) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.mouseButtonEventQueue.emplace(MouseButton{button}, Modifier{mods}, Action{action});
        });

        glfwSetScrollCallback(window, [](auto window, double x, double y) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.mouseScrollEventQueue.emplace(x, y); 
        });

        glfwSetCursorEnterCallback(window, [](auto window, int entered) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.cursorMovementEventQueue.emplace(CursorMovement{ !!entered });
        });

        glfwSetCursorPosCallback(window, [](auto window, double x, double y) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.cursorPositionEventQueue.emplace(x, y);
            inputManager.updateInputState();
        });

        glfwSetFramebufferSizeCallback(window, [](auto window, int x, int y) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.lastResizeTime.store(glfwGetTime(), std::memory_order::relaxed);
            inputManager.windowResizeEventQueue.emplace(x, y);
            inputManager.updateInputState();
        });

        glfwSetWindowPosCallback(window, [](auto window, int x, int y) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.windowMoveEventQueue.emplace(x, y); 
            inputManager.updateInputState();
        });

        glfwSetWindowFocusCallback(window, [](auto window, int focused) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.windowFocusEventQueue.emplace(focused);
            inputManager.updateInputState();
        });

        glfwSetWindowCloseCallback(window, [](auto window) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.windowCloseEventQueue.emplace();
            inputManager.updateInputState();
        });

        int count;
        GLFWmonitor** monitors = glfwGetMonitors(&count);        
        for (int i = 0; i < count; ++i) {
            glfwSetMonitorUserPointer(monitors[i], this);
        }

        glfwSetMonitorCallback([](auto monitor, int event) {
            InputManager* im = nullptr;
            int count;
            GLFWmonitor** monitors = glfwGetMonitors(&count);
            for (int i = 0; i < count; ++i) {
                im = (InputManager*)glfwGetMonitorUserPointer(monitors[i]);
                if (im != nullptr) break;
            }
            if (im != nullptr) {
                glfwSetMonitorUserPointer(monitor, im);
                im->monitorStateChangedEventQueue.emplace(monitor, event);
            }
        });

        glfwSetCharCallback(window, [](auto window, unsigned int codepoint) {
            auto& inputManager = *(InputManager*)glfwGetWindowUserPointer(window);
            inputManager.textEventQueue.emplace(codepoint);
        });

        currentKeyboardPriority = previousKeyboardPriority = currentMousePriority = previousMousePriority = defaultPriority = 0;
        registerWindow(window);
        globalInputState = new PerFrameGlobalInputData{};
        previousState = new PerFrameGlobalInputData{};
        fillInputState(previousState);
        fillInputState(globalInputState);
    }

    void InputManager::pollEvents() {
        glfwWaitEventsTimeout(0.1);
        elapsedTime();
        updateInputState();
    }

    void InputManager::runTasks() {
        std::packaged_task<void()> task;
        while (tasks.try_dequeue(task)) {
            task();
        }

        for (auto it : secondaryInputManagers) {
            it->runTasks();
        }
    }
    
    void InputManager::handleEvents() {
        // TODO: Mouse priority

        static const size_t MAX_EVENT_COUNT_PER_FRAME = 20;

        auto tempKeyHandlers = backupContainer(keyHandlers);
        auto tempUtf8KeyHandlers = backupContainer(utf8KeyHandlers);

        // priority decreased -> send artificial release to affected handlers
        if (previousKeyboardPriority > currentKeyboardPriority) {
            for (int i = 0; i < keyStates.size(); ++i) {
                int code = i + 1;
                auto& prio = keyStates[i];
                if (prio > currentKeyboardPriority) {
                    prio = currentKeyboardPriority;
                    for (auto& h : tempKeyHandlers) {
                        if (h.isEnabled() && currentKeyboardPriority < h.priority && previousKeyboardPriority >= h.priority)
                            h.handler(code, Modifier::None, Action::Release);
                    }

                    if (!tempUtf8KeyHandlers.empty()) {
                        const char* utf8key = glfwGetKeyName(code, glfwGetKeyScancode(code)); // key, scancode -> name
                        if (utf8key) {
                            for (auto& h : tempUtf8KeyHandlers) {
                                if (h.isEnabled() && currentKeyboardPriority < h.priority && previousKeyboardPriority >= h.priority)
                                    h.handler(utf8key, Modifier::None, Action::Release);
                            }
                        }
                    }
                }
            }
        }
        previousKeyboardPriority = currentKeyboardPriority;

        typename decltype(keyEventQueue)::value_type v;
        int i = 0;
        while (keyEventQueue.try_dequeue(v)) {
            for (auto& h : tempKeyHandlers) {
                int code = h.usesScancode() ? std::get<1>(v) : std::get<0>(v);
                bool s = std::get<3>(v) == Action::Press || (keyStates[std::get<0>(v) - 1] >= 0 && keyStates[std::get<0>(v) - 1] >= h.priority);
                if (s && h.isEnabled() && currentKeyboardPriority >= h.priority) 
                    h.handler(code, std::get<2>(v), std::get<3>(v));
            }
        
            if (!tempUtf8KeyHandlers.empty()) {
                const char* utf8key = glfwGetKeyName(std::get<0>(v), std::get<1>(v)); // key, scancode -> name
                if (utf8key) {
                    for (auto& h : tempUtf8KeyHandlers) {
                        bool s = std::get<3>(v) == Action::Press || (keyStates[std::get<0>(v) - 1] >= 0 && keyStates[std::get<0>(v) - 1] >= h.priority);
                        if (s && h.isEnabled() && currentKeyboardPriority >= h.priority) 
                            h.handler(utf8key, std::get<2>(v), std::get<3>(v));
                    }
                }
            }

            if (std::get<3>(v) == Action::Press) keyStates[std::get<0>(v) - 1] = currentKeyboardPriority;
            else if (std::get<3>(v) == Action::Release) keyStates[std::get<0>(v) - 1] = -1;

            i++;
        }

        restoreContainer(keyHandlers, tempKeyHandlers);
        restoreContainer(utf8KeyHandlers, tempUtf8KeyHandlers);

        std::function<void(void)> cursorHoldCallback;
        while (cursorHoldEventCallbackQueue.try_dequeue(cursorHoldCallback)) {
            cursorHoldCallback();
        }

        auto handle = [&](auto& queue, auto& handlers, int currentPriority, bool onlyLast) {
            auto tempHandlers = backupContainer(handlers);
            typename std::decay_t<decltype(queue)>::value_type v;
            int i = 0;
            while (queue.try_dequeue(v)) {
                i++;
                if (onlyLast) continue;
                for (auto& h : tempHandlers) {
                    if (h.isEnabled() && currentPriority >= h.priority) 
                        std::apply(h.handler, v);
                }
            }
            if (onlyLast && i > 0) {
                for (auto& h : tempHandlers) {
                    if (h.isEnabled() && currentPriority >= h.priority) 
                        std::apply(h.handler, v);
                }
            }
            restoreContainer(handlers, tempHandlers);
        };

        handle(mouseButtonEventQueue, mouseButtonHandlers, currentMousePriority, false);
        handle(mouseScrollEventQueue, mouseScrollHandlers, currentMousePriority, false);
        handle(cursorMovementEventQueue, cursorMovementHandlers, currentMousePriority, false);
        handle(cursorPositionEventQueue, cursorPositionHandlers, currentMousePriority, false);
        handle(windowResizeEventQueue, windowResizeHandlers, 0, true);
        handle(windowMoveEventQueue, windowMoveHandlers, 0, false);
        handle(windowFocusEventQueue, windowFocusHandlers, 0, false);
        handle(windowCloseEventQueue, windowCloseHandlers, 0, false);
        handle(pathDropEventQueue, pathDropHandlers, 0, false);
        handle(monitorStateChangedEventQueue, monitorStateChangedHandlers, 0, false);
        handle(textEventQueue, textHandlers, currentKeyboardPriority, false);

        for (auto it : secondaryInputManagers) {
            it->handleEvents();
        }
    }

    void InputManager::elapsedTime() {
        // TODO: cursorhold nal igy nem jo a backupcontainer es megoldas, mert ket szalbol van hasznalva
        auto tempHandlers = backupContainer(cursorHoldHandlers);

        double x, y;
        glfwGetCursorPos(window, &x, &y);
        for (auto& it : tempHandlers) {
            if (!it.isEnabled()) continue;

            if (it.handler.startTime < 0) {
                it.handler.startTime = glfwGetTime() * 1000.0;
                it.handler.x = x;
                it.handler.y = y;
                continue;
            }

            double dx = x - it.handler.x, dy = y - it.handler.y;
            double dv2 = dx * dx + dy * dy;
            if (dv2 <= it.handler.threshold2) {
                double elapsedTime = glfwGetTime() * 1000.0 - it.handler.startTime;
                if (elapsedTime >= it.handler.timeToTrigger) {
                    cursorHoldEventCallbackQueue.emplace(std::bind_front(it.handler.handler, it.handler.x, it.handler.y));
                }
            } else {
                it.handler.startTime = glfwGetTime() * 1000.0;
                it.handler.x = x;
                it.handler.y = y;
            }
        }

        restoreContainer(cursorHoldHandlers, tempHandlers);
    }

    void InputManager::fillInputState(PerFrameGlobalInputData* data) {
        *data = {};
        auto clipBoardStr = glfwGetClipboardString(window);
        data->clipBoardString = clipBoardStr == nullptr ? "" : clipBoardStr;
        for (int i = 0; i < 5; ++i) {
            data->mouseButton[i] = glfwGetMouseButton(window, i);
        }
        data->inputModeCursor = glfwGetInputMode(window, GLFW_CURSOR);
        int axes_c, buttons_c;
        const float* axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axes_c);
        for (int i = 0; i < axes_c; ++i) {
            data->joystickAxes.push_back(axes[i]);
        }
        const unsigned char* buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttons_c);
        for (int i = 0; i < buttons_c; ++i) {
            data->joystickButtons.push_back(buttons[i]);
        }
        data->gamepadStateErrorCode = glfwGetGamepadState(GLFW_JOYSTICK_1, data->gamepadState.get());
        int monitor_c;
        GLFWmonitor** monitors = glfwGetMonitors(&monitor_c);
        for (int i = 0; i < monitor_c; ++i) {
            PerFrameMonitorData mData = {};
            mData.videoMode = std::make_unique<GLFWvidmode>(*glfwGetVideoMode(monitors[i]));
            glfwGetMonitorPos(monitors[i], &mData.posX, &mData.posY);
            glfwGetMonitorWorkarea(monitors[i], &mData.workAreaX, &mData.workAreaY, &mData.workAreaW, &mData.workAreaH);
            glfwGetMonitorContentScale(monitors[i], &mData.scaleX, &mData.scaleY);
            data->monitors.push_back(std::move(mData));
        }
        glfwGetWindowSize(window, &data->w, &data->h);
        glfwGetFramebufferSize(window, &data->displayW, &data->displayH);

        for (auto w : secondaryWindows) {
            PerFramePerViewportData vData;
            vData.focused = glfwGetWindowAttrib(w, GLFW_FOCUSED);
            vData.hovered = glfwGetWindowAttrib(w, GLFW_HOVERED);
            vData.iconified = glfwGetWindowAttrib(w, GLFW_ICONIFIED);
            glfwGetCursorPos(w, &vData.cursorX, &vData.cursorY);
            for (int i = 0; i < 5; ++i) {
                vData.mouseButton[i] = glfwGetMouseButton(w, i);
            }
            glfwGetWindowPos(w, &vData.posX, &vData.posY);
            glfwGetWindowSize(w, &vData.width, &vData.height);
            data->viewportData.insert(std::make_pair(w, std::move(vData)));
        }

        data->timestamp = glfwGetTime();
    }

    void InputManager::updateInputState() {
        fillInputState(previousState);
        previousState = globalInputState.exchange(previousState, std::memory_order::release);
    }

    void InputManager::registerWindow(GLFWwindow* window) {
        secondaryWindows.push_back(window);
    }

    void InputManager::removeRegisteredWindow(GLFWwindow* window) {
        auto found = std::find(secondaryWindows.begin(), secondaryWindows.end(), window);
        assert(found != secondaryWindows.end());
        secondaryWindows.erase(found);
    }

    void InputManager::registerInputManager(InputManager* inputManager) {
        secondaryInputManagers.push_back(inputManager);
    }

    void InputManager::removeRegisteredInputManager(InputManager* inputManager) {
        auto found = std::find(secondaryInputManagers.begin(), secondaryInputManagers.end(), inputManager);
        assert(found != secondaryInputManagers.end());
        secondaryInputManagers.erase(found);
    }

    void InputManager::setCurrentKeyboardHandlerPriority(int priority) {
        currentKeyboardPriority = priority;
    }

    void InputManager::setCurrentMouseHandlerPriority(int priority) {
        currentMousePriority = priority;
    }

    void InputManager::setDefaultHandlerPriority(int priority) {
        defaultPriority = priority;
    }

    InputManager::CallbackHandler InputManager::registerKeyHandlerWithKey(std::function<void(int, Modifier, Action)> handler) {
        keyHandlers.emplace_back(std::move(handler), false, defaultPriority);
        return CallbackHandler{ this, CallbackType::Key, keyHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerKeyHandler(std::function<void(int, Modifier, Action)> handler) {
        keyHandlers.emplace_back(std::move(handler), true, defaultPriority);
        return CallbackHandler{ this, CallbackType::Key, keyHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerUtf8KeyHandler(std::function<void(const char*, Modifier, Action)> handler) {
        utf8KeyHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{this, CallbackType::Utf8Key, utf8KeyHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerMouseButtonHandler(std::function<void(MouseButton, Modifier, Action)> handler) {
        mouseButtonHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{this, CallbackType::MouseButton, mouseButtonHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerMouseScrollHandler(std::function<void(double, double)> handler) {
        mouseScrollHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{this, CallbackType::MouseScroll, mouseScrollHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerCursorMovementHandler(std::function<void(CursorMovement)> handler) {
        cursorMovementHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{this, CallbackType::CursorMovement, cursorMovementHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerCursorPositionHandler(std::function<void(double, double)> handler) {
        cursorPositionHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{this, CallbackType::CursorPosition, cursorPositionHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerCursorHoldHandler(double triggerTimeInMs, double threshold, std::function<void(double, double)> handler) {
        CursorHoldData data;
        data.handler = std::move(handler);
        data.threshold2 = threshold * threshold;
        data.timeToTrigger = triggerTimeInMs;
        data.x = data.y = 0;
        data.startTime = -1;
        cursorHoldHandlers.emplace_back(std::move(data), defaultPriority);
        return CallbackHandler{this, CallbackType::CursorHold, cursorHoldHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerWindowResizeHandler(std::function<void(int, int)> handler) {
        windowResizeHandlers.emplace_back(std::move(handler), 0);
        return CallbackHandler{this, CallbackType::WindowResize, windowResizeHandlers.size() - 1};
    }

    InputManager::CallbackHandler InputManager::registerWindowMoveHandler(std::function<void(int, int)> handler) {
        windowMoveHandlers.emplace_back(std::move(handler), 0);
        return CallbackHandler{ this, CallbackType::WindowMove, windowMoveHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerMonitorStateChangedHandler(std::function<void(GLFWmonitor*, int)> handler) {
        monitorStateChangedHandlers.emplace_back(std::move(handler), 0);
        return CallbackHandler{ this, CallbackType::MonitorStateChanged, monitorStateChangedHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerTextCallback(std::function<void(unsigned int)> handler) {
        textHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{ this, CallbackType::Text, textHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerWindowFocusHandler(std::function<void(bool)> handler) {
        windowFocusHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{ this, CallbackType::WindowFocus, windowFocusHandlers.size() - 1 };
    }

    InputManager::CallbackHandler InputManager::registerWindowCloseHandler(std::function<void()> handler) {
        windowCloseHandlers.emplace_back(std::move(handler), defaultPriority);
        return CallbackHandler{ this, CallbackType::WindowClose, windowCloseHandlers.size() - 1 };
    }
	
	InputManager::CallbackHandler InputManager::registerPathDropHandler_impl2(std::function<void(const std::vector<std::string>&)> handler) {
        pathDropHandlers.emplace_back(std::move(handler), 0);
        return CallbackHandler{this, CallbackType::PathDrop, pathDropHandlers.size() - 1};
    }

    void InputManager::setMouseMode(MouseMode mouseMode) {
        int mod;
        switch (mouseMode) {
        case MouseMode::Disabled: mod = GLFW_CURSOR_DISABLED; break;
        case MouseMode::Enabled: mod = GLFW_CURSOR_NORMAL; break;
        }
        executeOn([w = window, m = mod](){ glfwSetInputMode(w, GLFW_CURSOR, m); });
    }

    int InputManager::getSpaceScanCode()
    {
        return glfwGetKeyScancode(GLFW_KEY_SPACE);
    }

    int InputManager::getEnterScanCode()
    {
        return glfwGetKeyScancode(GLFW_KEY_ENTER);
    }

    int InputManager::getRightArrowScanCode()
    {
        return glfwGetKeyScancode(GLFW_KEY_RIGHT);
    }

    int InputManager::getLeftArrowScanCode()
    {
        return glfwGetKeyScancode(GLFW_KEY_LEFT);
    }

    void InputManager::CallbackHandler::enable_impl(bool enable)
    {
        switch (type)
        {
        default: assert(false); break;
        case CallbackType::Key:
            pInputManager->keyHandlers[index].setEnabled(enable);
            break;
        case CallbackType::Utf8Key:
            pInputManager->utf8KeyHandlers[index].setEnabled(enable);
            break;
        case CallbackType::MouseButton:
            pInputManager->mouseButtonHandlers[index].setEnabled(enable);
            break;
        case CallbackType::MouseScroll:
            pInputManager->mouseScrollHandlers[index].setEnabled(enable);
            break;
        case CallbackType::CursorMovement:
            pInputManager->cursorMovementHandlers[index].setEnabled(enable);
            break;
        case CallbackType::CursorPosition:
            pInputManager->cursorPositionHandlers[index].setEnabled(enable);
            break;
        case CallbackType::WindowResize:
            pInputManager->windowResizeHandlers[index].setEnabled(enable);
            break;
        case CallbackType::WindowMove:
            pInputManager->windowMoveHandlers[index].setEnabled(enable);
            break;
        case CallbackType::CursorHold:
            pInputManager->cursorHoldHandlers[index].setEnabled(enable);
            break;
		case CallbackType::PathDrop:
            pInputManager->pathDropHandlers[index].setEnabled(enable);
            break;
        case CallbackType::MonitorStateChanged:
            pInputManager->monitorStateChangedHandlers[index].setEnabled(enable);
            break;
        case CallbackType::Text:
            pInputManager->textHandlers[index].setEnabled(enable);
            break;
        case CallbackType::WindowFocus:
            pInputManager->windowFocusHandlers[index].setEnabled(enable);
            break;
        case CallbackType::WindowClose:
            pInputManager->windowCloseHandlers[index].setEnabled(enable);
            break;
        }
    }
}

glfwim::PerFrameGlobalInputData::PerFrameGlobalInputData()
    : gamepadState{new GLFWgamepadstate()}
{
}

glfwim::PerFrameGlobalInputData::PerFrameGlobalInputData(const PerFrameGlobalInputData& that)
    : PerFrameGlobalInputData()
{
    *this = that;
}

glfwim::PerFrameGlobalInputData& glfwim::PerFrameGlobalInputData::operator=(const PerFrameGlobalInputData& that)
{
    clipBoardString = that.clipBoardString;
    for (int i = 0; i < 5; ++i) {
        mouseButton[i] = that.mouseButton[i];
    }
    inputModeCursor = that.inputModeCursor;
    joystickAxes = that.joystickAxes;
    joystickButtons = that.joystickButtons;
    *gamepadState = *that.gamepadState;
    gamepadStateErrorCode = that.gamepadStateErrorCode;
    monitors = that.monitors;
    w = that.w;
    h = that.h;
    displayW = that.displayW;
    displayH = that.displayH;
    viewportData = that.viewportData;
    timestamp = that.timestamp;
    return *this;
}
