#ifndef INPUT_MANAGER_HPP
#define INPUT_MANAGER_HPP

#include <functional>
#include <vector>
#include <cstring>
#include <string>
#include <future>
#include <readerwriterqueue/readerwriterqueue.h>
#include <concurrentqueue/concurrentqueue.h>

struct GLFWwindow;
struct GLFWmonitor;
struct GLFWvidmode;
struct GLFWgamepadstate;

namespace glfwim {
    enum class Modifier {
        None = 0, Shift = 1, Control = 2, Alt = 4, Super = 8
    };

    enum class Action {
        Release = 0, Press = 1, Repeat = 2
    };

    enum class MouseButton {
        Left = 0, Right = 1, Middle = 2
    };

    enum class CursorMovement {
        Leave = 0, Enter = 1
    };

    enum class MouseMode {
        Disabled = 0, Enabled = 1
    };

    struct PerFrameMonitorData {
        std::unique_ptr<GLFWvidmode> videoMode;
        int posX, posY, workAreaX, workAreaY, workAreaW, workAreaH;
        float scaleX, scaleY;

        PerFrameMonitorData() = default;
        PerFrameMonitorData& operator=(const PerFrameMonitorData& that);
        PerFrameMonitorData(const PerFrameMonitorData& that);
    };

    struct PerFramePerViewportData {
        int focused, hovered, iconified;
        double cursorX, cursorY;
        int mouseButton[5];
        int posX, posY, width, height;
    };

    struct PerFrameGlobalInputData {
        PerFrameGlobalInputData();
        PerFrameGlobalInputData(const PerFrameGlobalInputData& that);
        PerFrameGlobalInputData& operator=(const PerFrameGlobalInputData& that);

        std::string clipBoardString;
        int mouseButton[5];
        int inputModeCursor;
        std::vector<float> joystickAxes;
        std::vector<unsigned char> joystickButtons;
        std::unique_ptr<GLFWgamepadstate> gamepadState;
        int gamepadStateErrorCode;
        std::vector<PerFrameMonitorData> monitors;
        int w, h, displayW, displayH;
        std::unordered_map<GLFWwindow*, PerFramePerViewportData> viewportData;
        double timestamp;
    };

    class InputManager {
    private:
        struct CursorHoldData {
            std::function<void(double, double)> handler;
            double x, y;
            double threshold2, timeToTrigger, startTime;
        };

    public:
        static InputManager& instance() {
            static InputManager instance;
            return instance;
        }
		InputManager() = default;
		InputManager(const InputManager&) = delete;
		InputManager(InputManager&&) = delete;
		InputManager& operator=(const InputManager&) = delete;
		InputManager& operator=(InputManager&&) = delete;
		
    public:
        void initialize(GLFWwindow* window);
        void pollEvents();
        void runTasks();
        void handleEvents();
        void fillInputState(PerFrameGlobalInputData* data);
        void setMouseMode(MouseMode mouseMode);
        void registerWindow(GLFWwindow* window);
        void removeRegisteredWindow(GLFWwindow* window);
        void registerInputManager(InputManager* inputManager);
        void removeRegisteredInputManager(InputManager* inputManager);
        void setCurrentKeyboardHandlerPriority(int priority);
        void setCurrentMouseHandlerPriority(int priority);
        void setDefaultHandlerPriority(int priority);

    public:
        enum class CallbackType { 
            Key, Utf8Key, MouseButton, MouseScroll, CursorMovement, CursorPosition, WindowResize, WindowMove, CursorHold, PathDrop, MonitorStateChanged, Text, WindowFocus, WindowClose
        };

        class CallbackHandler {
        public:
            CallbackHandler(InputManager* inputManager, CallbackType type, size_t index) : pInputManager{inputManager}, type{type}, index{index} {}

            void enable() { enable_impl(true); }
            void disable() { enable_impl(false); }

        private:
            void enable_impl(bool enable);

        private:
            InputManager* pInputManager;
            CallbackType type;
            size_t index;
        };

        friend class CallbackHandler;

    public:
        CallbackHandler registerKeyHandlerWithKey(std::function<void(int, Modifier, Action)> handler);

        CallbackHandler registerKeyHandler(std::function<void(int, Modifier, Action)> handler);

        template <typename H>
        CallbackHandler registerKeyHandler(int scancode, H handler) {
            return registerKeyHandler([h = std::move(handler), s = scancode](int scancode, Modifier modifier, Action action){
                if (scancode == s) h(modifier, action);
            });
        }

        template <typename H>
        CallbackHandler registerKeyHandler(int scancode, Modifier modifier, H handler) {
            return registerKeyHandler([h = std::move(handler), s = scancode, m = modifier](int scancode, Modifier modifier, Action action){
                if (scancode == s && modifier == m) h(action);
            });
        }

        template <typename H>
        CallbackHandler registerKeyHandler(int scancode, Modifier modifier, Action action, H handler) {
            return registerKeyHandler([h = std::move(handler), s = scancode, m = modifier, a = action](int scancode, Modifier modifier, Action action){
                if (scancode == s && modifier == m && action == a) h();
            });
        }

        CallbackHandler registerUtf8KeyHandler(std::function<void(const char*, Modifier, Action)> handler);

        template <typename H>
        CallbackHandler registerUtf8KeyHandler(const char* utf8code, H handler) {
            if (!strcmp(utf8code, " ")) {
                return registerKeyHandler(getSpaceScanCode(), handler);
            }
            else if (!strcmp(utf8code, "\n")) {
                return registerKeyHandler(getEnterScanCode(), handler);
            }
            else if (!strcmp(utf8code, "->")) {
                return registerKeyHandler(getRightArrowScanCode(), handler);
            }
            else if (!strcmp(utf8code, "<-")) {
                return registerKeyHandler(getLeftArrowScanCode(), handler);
            }
            else {
                return registerUtf8KeyHandler([h = std::move(handler), u = utf8code](const char* utf8code, Modifier modifier, Action action){
                    if (!strcmp(u, utf8code)) h(modifier, action);
                });
            }
        }

        template <typename H>
        CallbackHandler registerUtf8KeyHandler(const char* utf8code, Modifier modifier, H handler) {
            if (!strcmp(utf8code, " ")) {
                return registerKeyHandler(getSpaceScanCode(), modifier, handler);
            }
            else if (!strcmp(utf8code, "\n")) {
                return registerKeyHandler(getEnterScanCode(), modifier, handler);
            }
            else if (!strcmp(utf8code, "->")) {
                return registerKeyHandler(getRightArrowScanCode(), modifier, handler);
            }
            else if (!strcmp(utf8code, "<-")) {
                return registerKeyHandler(getLeftArrowScanCode(), modifier, handler);
            }
            else {
                return registerUtf8KeyHandler([h = std::move(handler), u = utf8code, m = modifier](const char* utf8code, Modifier modifier, Action action){
                    if (!strcmp(u, utf8code) && m == modifier) h(action);
                });
            }
        }

        template <typename H>
        CallbackHandler registerUtf8KeyHandler(const char* utf8code, Modifier modifier, Action action, H handler) {
            if (!strcmp(utf8code, " ")) {
                return registerKeyHandler(getSpaceScanCode(), modifier, action, handler);
            }
            else if (!strcmp(utf8code, "\n")) {
                return registerKeyHandler(getEnterScanCode(), modifier, action, handler);
            }
            else if (!strcmp(utf8code, "->")) {
                return registerKeyHandler(getRightArrowScanCode(), modifier, action, handler);
            }
            else if (!strcmp(utf8code, "<-")) {
                return registerKeyHandler(getLeftArrowScanCode(), modifier, action, handler);
            }
            else {
                return registerUtf8KeyHandler([h = std::move(handler), u = utf8code, m = modifier, a = action](const char* utf8code, Modifier modifier, Action action){
                    if (!strcmp(u, utf8code) && m == modifier && a == action) h();
                });
            }
        }

        CallbackHandler registerMouseButtonHandler(std::function<void(MouseButton, Modifier, Action)> handler);

        template <typename H>
        CallbackHandler registerMouseButtonHandler(MouseButton mouseButton, H handler) {
            return registerMouseButtonHandler([h = std::move(handler), mb = mouseButton](MouseButton mouseButton, Modifier modifier, Action action){
                if (mb == mouseButton) h(modifier, action);
            });
        }

        template <typename H>
        CallbackHandler registerMouseButtonHandler(MouseButton mouseButton, Modifier modifier, H handler) {
            return registerMouseButtonHandler([h = std::move(handler), mb = mouseButton, m = modifier](MouseButton mouseButton, Modifier modifier, Action action){
                if (mb == mouseButton && m == modifier) h(action);
            });
        }

        template <typename H>
        CallbackHandler registerMouseButtonHandler(MouseButton mouseButton, Modifier modifier, Action action, H handler) {
            return registerMouseButtonHandler([h = std::move(handler), mb = mouseButton, m = modifier, a = action](MouseButton mouseButton, Modifier modifier, Action action){
                if (mb == mouseButton && m == modifier && a == action) h();
            });
        }

        CallbackHandler registerMouseScrollHandler(std::function<void(double, double)> handler);

        CallbackHandler registerCursorMovementHandler(std::function<void(CursorMovement)> handler);

        template <typename H>
        CallbackHandler registerCursorMovementHandler(CursorMovement movement, H handler) {
            return registerCursorMovementHandler([h = std::move(handler), m = movement](CursorMovement movement){
                if (m == movement) h();
            });
        }

        CallbackHandler registerCursorPositionHandler(std::function<void(double, double)> handler);

        CallbackHandler registerCursorHoldHandler(double triggerTimeInMs, double threshold, std::function<void(double, double)> handler);

        CallbackHandler registerWindowResizeHandler(std::function<void(int, int)> handler);
        CallbackHandler registerWindowMoveHandler(std::function<void(int, int)> handler);
		
		template <typename Head, typename Second, typename... Args>
        CallbackHandler registerPathDropHandler(Head&& head, Second&& second, Args&&... args) {
            std::vector<std::string> filters; filters.reserve(sizeof...(args) + 1);
            return registerPathDropHandler_impl(std::move(filters), std::forward<Head>(head), std::forward<Second>(second), std::forward<Args>(args)...);
        }

        template <typename Handler>
        CallbackHandler registerPathDropHandler(Handler&& handler) {
            return registerPathDropHandler_impl(std::vector<std::string>{}, std::forward<Handler>(handler));
        }

        CallbackHandler registerMonitorStateChangedHandler(std::function<void(GLFWmonitor*, int)> handler);
        CallbackHandler registerTextCallback(std::function<void(unsigned int)> handler);
        CallbackHandler registerWindowFocusHandler(std::function<void(bool)> handler);
        CallbackHandler registerWindowCloseHandler(std::function<void()> handler);
		
    private:
        template <typename T, typename = void> struct helper : std::false_type {};
        template <typename T> struct helper<T, std::void_t<decltype(std::declval<T>()(std::declval<std::string>()))>> : std::true_type {};

        CallbackHandler registerPathDropHandler_impl2(std::function<void(const std::vector<std::string>&)> handler);

        template <typename Handler>
        CallbackHandler registerPathDropHandler_impl(std::vector<std::string>&& filters, Handler&& handler) {
            if constexpr (helper<Handler>::value) {
                return registerPathDropHandler_impl2([h = std::move(handler), f = std::move(filters)](const std::vector<std::string>& paths) {
                    for (auto& it : paths) {
                        std::string ext;
                        auto found = it.find_last_of(".");
                        if (found != std::string::npos) {
                            ext = it.substr(found + 1);
                        }
                        if (f.empty()) h(it);
                        else for (auto& fExt : f) {
                            if (ext == fExt) {
                                h(it);
                                break;
                            }
                        }
                    }
                });
            } else { // vector<string>
                return registerPathDropHandler_impl2([h = std::move(handler), f = std::move(filters)](const std::vector<std::string>& paths) {
                    std::vector<std::string> pps;
                    for (auto& it : paths) {
                        std::string ext;
                        auto found = it.find_last_of(".");
                        if (found != std::string::npos) {
                            ext = it.substr(found + 1);
                        }
                        if (f.empty()) pps.push_back(it);
                        else for (auto& fExt : f) {
                            if (ext == fExt) {
                                pps.push_back(it);
                                break;
                            }
                        }
                    }
                    h(pps);
                });
            }
        }

        template <typename... Tail>
        CallbackHandler registerPathDropHandler_impl(std::vector<std::string>&& filters, std::string&& head, Tail&&... tail) {
            filters.push_back(std::move(head));
            return registerPathDropHandler_impl(std::move(filters), std::forward<Tail>(tail)...);
        }

    public:
        template <typename T>
        std::future<void> executeOn(T task)
        {
            auto pt = std::packaged_task<void()>(std::move(task));
            auto future = pt.get_future();
            if (std::this_thread::get_id() == mainThreadId) {
                pt();
            } else {
                tasks.enqueue(std::move(pt));
            }
            return future;
        }

    private:
        bool isKeyboardCaptured();
        bool isMouseCaptured();

    private:
        void elapsedTime();
        void updateInputState();
        static int getSpaceScanCode();
        static int getEnterScanCode();
        static int getRightArrowScanCode();
        static int getLeftArrowScanCode();

        template <typename Container>
        static Container backupContainer(Container& container)
        {
            auto ret = std::move(container);
            container.clear();
            return ret;
        }

        template <typename Container>
        static void restoreContainer(Container& container, Container& backup)
        {
            std::swap(container, backup);
            if (!backup.empty()) {
                container.insert(std::end(container), std::make_move_iterator(std::begin(backup)), std::make_move_iterator(std::end(backup)));
            }
        }

        template <typename T>
        struct HandlerHolder {
            HandlerHolder(T handler, int priority) 
                : handler{std::move(handler)}
                , priority{ priority }
            {}
            HandlerHolder(HandlerHolder&& that) {
                handler = std::move(that.handler);
                priority = that.priority;
                enabled.store(that.enabled.load(std::memory_order::relaxed), std::memory_order::relaxed);
            }
            HandlerHolder& operator=(HandlerHolder that) {
                std::swap(handler, that.handler);
                std::swap(priority, that.priority);
                enabled.store(that.enabled.load(std::memory_order::relaxed), std::memory_order::relaxed);
                return *this; 
            }
            T handler;
            int priority;
            bool isEnabled() const { return enabled.load(std::memory_order::relaxed); }
            void setEnabled(bool enable) { enabled.store(enable, std::memory_order::relaxed); }
        private:
            std::atomic<bool> enabled = true;
        };

        template <typename T>
        struct KeyHandlerHolder : HandlerHolder<T> {
            KeyHandlerHolder(T handler, bool useScancode, int priority)
                : HandlerHolder<T>{std::move(handler), priority}
                , useScancode{useScancode}
            {}
            bool usesScancode() const { return useScancode; }
        private:
            bool useScancode;
        };

    public:
        std::atomic<PerFrameGlobalInputData*> globalInputState;
        std::atomic<double> lastResizeTime;

    private:
        GLFWwindow* window;
        std::thread::id mainThreadId;
        std::vector<int> keyStates;
        PerFrameGlobalInputData* previousState;
        std::vector<GLFWwindow*> secondaryWindows;
        std::vector<InputManager*> secondaryInputManagers;
        int defaultPriority, currentKeyboardPriority, currentMousePriority, previousKeyboardPriority, previousMousePriority;

    private:
        std::vector<KeyHandlerHolder<std::function<void(int, Modifier, Action)>>> keyHandlers;
        std::vector<HandlerHolder<std::function<void(const char*, Modifier, Action)>>> utf8KeyHandlers;
        std::vector<HandlerHolder<std::function<void(MouseButton, Modifier, Action)>>> mouseButtonHandlers;
        std::vector<HandlerHolder<std::function<void(double, double)>>> mouseScrollHandlers;
        std::vector<HandlerHolder<std::function<void(CursorMovement)>>> cursorMovementHandlers;
        std::vector<HandlerHolder<std::function<void(double, double)>>> cursorPositionHandlers;
        std::vector<HandlerHolder<std::function<void(int, int)>>> windowResizeHandlers;
        std::vector<HandlerHolder<std::function<void(int, int)>>> windowMoveHandlers;
        std::vector<HandlerHolder<std::function<void(GLFWmonitor*, int)>>> monitorStateChangedHandlers;
        std::vector<HandlerHolder<std::function<void(unsigned int)>>> textHandlers;
        std::vector<HandlerHolder<CursorHoldData>> cursorHoldHandlers;
		std::vector<HandlerHolder<std::function<void(const std::vector<std::string>& paths)>>> pathDropHandlers;
        std::vector<HandlerHolder<std::function<void(bool)>>> windowFocusHandlers;
        std::vector<HandlerHolder<std::function<void()>>> windowCloseHandlers;

        moodycamel::ReaderWriterQueue<std::tuple<int, int, Modifier, Action>> keyEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<MouseButton, Modifier, Action>> mouseButtonEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<double, double>> mouseScrollEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<CursorMovement>> cursorMovementEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<double, double>> cursorPositionEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<int, int>> windowResizeEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<int, int>> windowMoveEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<GLFWmonitor*, int>> monitorStateChangedEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<unsigned int>> textEventQueue;
        moodycamel::ReaderWriterQueue<std::function<void(void)>> cursorHoldEventCallbackQueue;
        moodycamel::ReaderWriterQueue<std::tuple<std::vector<std::string>>> pathDropEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<int>> windowFocusEventQueue;
        moodycamel::ReaderWriterQueue<std::tuple<>> windowCloseEventQueue;

    private:
        moodycamel::ConcurrentQueue<std::packaged_task<void()>> tasks;
    };
}
#endif
