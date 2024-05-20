#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <ostream>
#include <sstream>
#include <mutex>
#include <functional>
#include <thread>
#include <chrono>

#ifdef LOGGER_USING_FMT
#include <fmt/format.h>
#endif

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "./"
#endif

#ifndef LOGGER_IMPLEMENTATION

namespace ezpz {
#ifndef LOG_LEVELS
#define LOG_LEVELS(CHAR, NAME_STANDARD, NAME_UPPER, VALUE) \
    CHAR('F') NAME_STANDARD(Fatal) NAME_UPPER(FATAL) VALUE(1) \
    CHAR('E') NAME_STANDARD(Error) NAME_UPPER(ERROR) VALUE(2)\
    CHAR('W') NAME_STANDARD(Warning) NAME_UPPER(WARN) VALUE(4)\
    CHAR('I') NAME_STANDARD(Info) NAME_UPPER(INFO) VALUE(8)\
    CHAR('D') NAME_STANDARD(Debug) NAME_UPPER(DEBUG) VALUE(16)
#endif

#define NOOP(O)

#define MAKE_ENUM(E) E = 
#define MAKE_VALUE(V) V,

    enum class LogLevel : unsigned char {
        LOG_LEVELS(NOOP, NOOP, MAKE_ENUM, MAKE_VALUE)
        ALL = 255
    };

    LogLevel operator|(LogLevel lhs, LogLevel rhs);

#undef MAKE_ENUM
#undef MAKE_VALUE

namespace _impl {
    std::string toString(LogLevel logLevel);
    unsigned char fromChar(char c, std::string* pError = nullptr);
    bool HasLogLevelBit(unsigned char bits, LogLevel logLevel);
    std::ostream& operator<<(std::ostream& os, LogLevel logLevel);

    struct LogData {
       LogLevel level;
       int tick;
       std::string threadId;
       std::chrono::system_clock::time_point time;
       std::string message;
       int frameIndex;
    };

    class ILogOutput {
    public:
       virtual void Log(const LogData& data) = 0;
       virtual void SetVersion(const std::string& strVersion) = 0;
       virtual void SetStats(const std::string& strStats) = 0;
       virtual ~ILogOutput() = default;
    };

    struct OutputDescriptor {
        std::string pathFormat, lineFormat, contextFormat;
        bool bVersion, bStats, hasUserTag, append;
        unsigned char channels;

        void Reset()
        {
            contextFormat = "{{Indent}}{{Message}}";
            pathFormat = "communication.log";
            lineFormat = "{{Message}}";
            bVersion = true;
            bStats = true;
            hasUserTag = false;
            append = false;
            channels = (unsigned char)LogLevel::ALL;
        }
    };

    class LogOutputStream : public ILogOutput {
    public:
        LogOutputStream(OutputDescriptor desc, std::ostream& os) : os{os}, desc{desc} {}
        ~LogOutputStream() = default;
        void Log(const LogData& data) override;
        void SetVersion(const std::string& strVersion) override {
            if (desc.bVersion) os << "[VERSION]\n" << strVersion << std::endl << "[\\VERSION]\n"; 
        }
        void SetStats(const std::string& strStats) override {
            if (desc.bStats) os << "[STATS]\n" << strStats << std::endl << "[\\STATS]\n"; 
        }

    private:
        OutputDescriptor desc;
        std::ostream& os;
        volatile int lastFiredFrameIndex = -1;
        volatile int currentTabLevel = 0;
    };

    class LogOutputFile : public LogOutputStream {
    public:
        LogOutputFile(OutputDescriptor desc);

    private:
        std::ofstream file;
    };

    class LogOutputMemory : public ILogOutput {
    public:
        ~LogOutputMemory() = default;
        void Log(const LogData& data) override {
            logs.push_back(data);
        }

        void SetVersion(const std::string& strVersion) override {
            this->strVersion = strVersion; 
        }

        void SetStats(const std::string& strStats) override {
            this->strStats = strStats; 
        }

        void WriteDataToOutput(ILogOutput& output) const {
            if (!strVersion.empty()) output.SetVersion(strVersion);
            for (auto& it : logs) {
                output.Log(it);
            }
            if (!strStats.empty()) output.SetStats(strStats);
        }

    private:
    private:
        std::vector<LogData> logs;
        std::string strStats, strVersion;
    };

    inline void log_impl(std::stringstream& ss) {}

    template <typename T>
    inline void log_impl(std::stringstream& ss, T&& val) {
        ss << val;
    }

    template <typename T, typename... Tail>
    inline void log_impl(std::stringstream& ss, T&& val, Tail&&... args) {
        ss << val;
        return log_impl(ss, args...);
    }
}

	class Logger {
	private:
		Logger() = default;
		~Logger() = default;
		Logger(Logger const&) = delete;
		Logger& operator=(Logger const&) = delete;
	public:
		static Logger& Instance() {
			static Logger instance;
			return instance;
		}

    private:
        struct LogFrame {
            std::string message;
            std::chrono::system_clock::time_point time;
            int tick;
            int prev;
        };

        struct LogContext {
            std::vector<LogFrame> frames;
            inline static thread_local int actFrameIndex = -1;
        };

        friend class _impl::LogOutputStream;

	private:
		std::unique_ptr<_impl::LogOutputMemory> memoryLog;
        std::vector<std::unique_ptr<_impl::ILogOutput>> outputLogs;
        std::vector<_impl::OutputDescriptor> outputDescriptors;
        LogContext logContext;
		int tick = -1;
        mutable std::mutex mutex;

        bool isInit() const { return memoryLog.get(); }

	public:
        std::string Init(const std::string& path);

        struct [[nodiscard]] LogContextFrameHandler {
            LogContextFrameHandler(Logger& l) : l{ l }, index{ l.logContext.actFrameIndex } {}
            static LogContextFrameHandler makeReference(const Logger& l) { LogContextFrameHandler h{ const_cast<Logger&>(l)}; h.popped = true; return h; }
            ~LogContextFrameHandler() { Pop(); }
            void Pop() { if (!popped) { l.PopContext(); popped = true; } }
            void SetAsRootFrame() const { l.logContext.actFrameIndex = index; }
        private:
            Logger& l;
            bool popped = false;
            int index;
        };

        class LoggerWrapper {
            LoggerWrapper(Logger& logger, LogLevel logLevel) : logger{logger}, defaultLogLevel{logLevel} {}
        public:
            template <typename... Args> LogContextFrameHandler PushContext(Args&&... args) {
                return logger.PushContext(std::forward<Args>(args)...);
            }
            LogContextFrameHandler GetCurrentContext() const { return logger.GetCurrentContext(); }
            template <typename... Args> void Log(LogLevel logLevel, Args&&... args) { logger.Log(logLevel | defaultLogLevel, args...); }
#define TEMPLATE_NAME(N) template <typename... Args> void Log##N(Args&&... args)
#define TEMPLATE_LEVEL(L) { logger.Log(LogLevel::L | defaultLogLevel, std::forward<Args>(args)...); }

            LOG_LEVELS(NOOP, TEMPLATE_NAME, TEMPLATE_LEVEL, NOOP)

#undef TEMPLATE_NAME
#undef TEMPLATE_LEVEL
        private:
            friend class Logger;

            Logger& logger;
            LogLevel defaultLogLevel;
        };

        LoggerWrapper CreateWrapper(LogLevel logLevel) {
            return LoggerWrapper{*this, logLevel};
        }
        
        void SetUsertag(const std::string& usertag, const std::string& replacement) {
#ifdef USE_EZPZLOGGER
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif // EZPZLOGGER_USE_LOCK
            SetUsertag_Impl(usertag, replacement);
#endif // USE_EZPZLOGGER
        }

        template <typename... Args> LogContextFrameHandler PushContext(Args&&... args) {
#ifdef USE_EZPZLOGGER
            LogFrame lf;
            lf.time = std::chrono::system_clock::now();
            lf.tick = tick;
            lf.prev = logContext.actFrameIndex;

#ifndef LOGGER_USING_FMT
            std::stringstream ss;
            _impl::log_impl(ss, args...);
            lf.message = ss.str();
#else // LOGGER_USING_FMT
            lf.message = fmt::format(args...);
#endif // LOGGER_USING_FMT

#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{ mutex };
#endif // EZPZLOGGER_USE_LOCK
            logContext.frames.push_back(lf);
            logContext.actFrameIndex = logContext.frames.size() - 1;
#endif // USE_EZPZLOGGER
            return LogContextFrameHandler{ *this };
        }

        LogContextFrameHandler GetCurrentContext() const {
            return LogContextFrameHandler::makeReference(*this);
        }

        std::function<std::string(LogLevel)> LogLevelToString = _impl::toString;
        template <typename... Args> void Log(LogLevel logLevel, Args&&... args) {
#ifdef USE_EZPZLOGGER
#ifndef LOGGER_USING_FMT
            std::stringstream ss;
            _impl::log_impl(ss, args...);

#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif // EZPZLOGGER_USE_LOCK
            Log_Impl(logLevel, ss.str());
#else // LOGGER_USING_FMT
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif // EZPZLOGGER_USE_LOCK
            Log_Impl(logLevel, fmt::format(args...));
#endif // LOGGER_USING_FMT
#endif // USE_EZPZLOGGER
        }

#define TEMPLATE_NAME(N) template <typename... Args> void Log##N(Args&&... args)
#define TEMPLATE_LEVEL(L) { Log(LogLevel::L, args...); }

        LOG_LEVELS(NOOP, TEMPLATE_NAME, TEMPLATE_LEVEL, NOOP)

#undef TEMPLATE_NAME
#undef TEMPLATE_LEVEL

        template <typename T> void SetVersion(const T& version) {
#ifdef USE_EZPZLOGGER
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif
            auto str = version.toString();
            memoryLog->SetVersion(str);
            for (auto& it : outputLogs) {
                it->SetVersion(str);
            }
#endif
        }

        template <typename T> void SetStats(const T& stats) {
#ifdef USE_EZPZLOGGER
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif
            auto str = stats.toString();
            memoryLog->SetStats(str);
            for (auto& it : outputLogs) {
                it->SetStats(str);
            }
#endif
        }

		void StartNextTick();

        void SetToString(std::function<std::string(LogLevel)> formatter) {
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif
            LogLevelToString = std::move(formatter);
        }

    private:
        void PopContext() {
#ifdef USE_EZPZLOGGER
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{ mutex };
#endif // EZPZLOGGER_USE_LOCK
            logContext.actFrameIndex = logContext.frames[logContext.actFrameIndex].prev;
#endif // USE_EZPZLOGGER
        }

        void AddOutputChannel(std::unique_ptr<_impl::ILogOutput> pOutput) {
#ifdef USE_EZPZLOGGER
#ifdef EZPZLOGGER_USE_LOCK
            std::lock_guard<std::mutex> _l{mutex};
#endif
            AddOutputChannel_Impl(std::move(pOutput));
#endif
        }

		void Log_Impl(LogLevel logLevel, std::string const& str);
        void AddOutputChannel_Impl(std::unique_ptr<_impl::ILogOutput> pOutput);
        void SetUsertag_Impl(const std::string& usertag, const std::string& replacement);
	};

	inline Logger& theLogger = Logger::Instance();
}

#endif // LOGGER_IMPLEMENTATION

#ifdef LOGGER_IMPLEMENTATION
#include <iostream>
#include <filesystem>
#include <thread>

#ifdef _CRT_SECURE_NO_WARNINGS
    #define HAD_CRT_SUPRESSION
#else
    #define _CRT_SECURE_NO_WARNINGS
#endif

namespace fs = std::filesystem;
using namespace std::chrono;

namespace ezpz {
namespace _impl {
    std::string replaceString(std::string subject, const std::string& search, const std::string& replace) {
        size_t pos = 0;
        while ((pos = subject.find(search, pos)) != std::string::npos) {
             subject.replace(pos, search.length(), replace);
             pos += replace.length();
        }
        return subject;
    }

    // Based on: https://stackoverflow.com/questions/15957805/extract-year-month-day-etc-from-stdchronotime-point-in-c
    std::string replaceTime(std::string subject, system_clock::time_point tp) {
        typedef duration<int, std::ratio_multiply<hours::period, std::ratio<24> >::type> days;
        system_clock::duration d = tp.time_since_epoch();
        d -= duration_cast<days>(d);
        d -= duration_cast<hours>(d);
        d -= duration_cast<minutes>(d);
        d -= duration_cast<seconds>(d);

        milliseconds ms = duration_cast<milliseconds>(d);

        time_t tt = system_clock::to_time_t(tp);
        auto t = localtime(&tt);

        auto year = std::to_string(t->tm_year + 1900);
        auto month = std::to_string(t->tm_mon + 1);
        auto day = std::to_string(t->tm_mday);
        auto hour = std::to_string(t->tm_hour);
        auto minute = std::to_string(t->tm_min);
        auto second = std::to_string(t->tm_sec);
        auto msstr = std::to_string(ms.count());

        subject = replaceString(subject, "{{Year}}", year);
        subject = replaceString(subject, "{{Month}}", month);
        subject = replaceString(subject, "{{Day}}", day);
        subject = replaceString(subject, "{{Hour}}", hour);
        subject = replaceString(subject, "{{Minute}}", minute);
        subject = replaceString(subject, "{{Second}}", second);
        subject = replaceString(subject, "{{Ms}}", msstr);

        return subject;
    }

    std::vector<std::string> split(const std::string& str, char delim)
    {
        std::stringstream ss(str);
        std::string tmp;
        std::vector<std::string> tokens;
        while (getline(ss, tmp, delim))
        {
            tokens.push_back(tmp);
        }
        return tokens;
    }

    // trim from start (in place)
    void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    }

    // trim from end (in place)
    void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

    // trim from both ends (in place)
    void trim(std::string &s) {
        ltrim(s);
        rtrim(s);
    }
}

    std::string Logger::Init(const std::string& path)
	{
#ifdef USE_EZPZLOGGER
        using namespace _impl;
        memoryLog = std::make_unique<_impl::LogOutputMemory>(); 

        std::ifstream file{path};

        if (!file.is_open()) {
            return "Logger config file {" + path + "} not found!";
        }

        auto t = system_clock::now();

        _impl::OutputDescriptor desc;
        desc.Reset();
        bool descValid = false;

        auto finishDesc = [&](){
            if (descValid) {
                if (desc.hasUserTag) {
                    outputDescriptors.push_back(desc);
                } else {
                    if (desc.pathFormat == "cerr") {
                        outputLogs.push_back(std::make_unique<_impl::LogOutputStream>(desc, std::cerr));
                    } else if (desc.pathFormat == "cout") {
                        outputLogs.push_back(std::make_unique<_impl::LogOutputStream>(desc, std::cout));
                    } else {
                        outputLogs.push_back(std::make_unique<_impl::LogOutputFile>(desc));
                    }
                }

                desc.Reset();
            }
            descValid = true;
        };
       
        std::stringstream error;
        int lineCount = 0;
        std::string line = "";
        while (std::getline(file, line)) {
            lineCount++;
            trim(line);
            if (line.empty() || line[0] == '#') {
                continue;
            } else if (line.front() == '[') { // new output
                finishDesc();

                if (line.back() != ']') {
                    error << "Line " << lineCount << ": closing ] is missing.\n";
                    continue;
                }

                line = line.substr(1, line.size() - 2); // remove [ ]
                line = replaceTime(line, t);
                line = replaceString(line, "{{ProjectRoot}}", PROJECT_SOURCE_DIR);

                desc.pathFormat = line;
                desc.hasUserTag = desc.pathFormat.find("{$", 0) != std::string::npos;
            } else if (line.find('=', 0) == std::string::npos) {
                error << "Line " << lineCount << ": separator (=) is missing.\n";
                continue;
            } else {
                auto splitted = split(line, '=');
                if (splitted.size() != 2) {
                    error << "Line " << lineCount << ": multiple separators (=) are found.\n";
                    continue;
                }
                trim(splitted[0]);
                trim(splitted[1]);
                if (splitted[0] == "channels") {
                    desc.channels = 0;
                    for (auto c : splitted[1]) {
                        std::string err;
                        auto res = _impl::fromChar(c, &err);
                        if (err.empty()) {
                            desc.channels |= res;
                        } else {
                            error << "Line " << lineCount << ": '" << c << "' is ignored. " << err << "\n";
                        }
                    }
                } else if (splitted[0] == "stats") {
                    if (splitted[1] == "true") {
                        desc.bStats = true;
                    } else if (splitted[1] == "false") {
                        desc.bStats = false;
                    } else {
                        error << "Line " << lineCount << ": value must be 'true' or 'false'. Defaulting to 'true'.\n";
                        continue;
                    }
                } else if (splitted[0] == "version") {
                    if (splitted[1] == "true") {
                        desc.bVersion = true;
                    } else if (splitted[1] == "false") {
                        desc.bVersion = false;
                    } else {
                        error << "Line " << lineCount << ": value must be 'true' or 'false'. Defaulting to 'true'.\n";
                        continue;
                    }
                } else if (splitted[0] == "format") {
                    desc.lineFormat = splitted[1];
                    desc.hasUserTag |= desc.lineFormat.find("{$", 0) != std::string::npos;
                } else if (splitted[0] == "context") {
                    desc.contextFormat = splitted[1];
                    desc.hasUserTag |= desc.lineFormat.find("{$", 0) != std::string::npos;
                } else if (splitted[0] == "append") {
                    if (splitted[1] == "true") {
                        desc.append = true;
                    } else if (splitted[1] == "false") {
                        desc.append = false;
                    } else {
                        error << "Line " << lineCount << ": value must be 'true' or 'false'. Defaulting to 'false'.\n";
                        continue;
                    }
                } else {
                    error << "Line " << lineCount << ": key is not recognized.\n";
                    continue;
                }
            }
        }
        finishDesc();
        return error.str();
#else
        return "";
#endif
	}

    void _impl::LogOutputStream::Log(const LogData& data) {
        if (_impl::HasLogLevelBit(desc.channels, data.level)) {
            // Write correct context
            std::vector<int> frameIndexesToWrite, backtrackedFrames;
            auto findFrameInBacktrack = [&](auto index) {
                for (int i = 0; i < backtrackedFrames.size(); ++i) {
                    if (backtrackedFrames[i] == index) {
                        return i;
                    }
                }
                return -1;
            };

            // try going from last written back to data
            int actFrameIndex = lastFiredFrameIndex;
            int actTabLevel = currentTabLevel;
            while (actFrameIndex != data.frameIndex && actFrameIndex != -1) {
                backtrackedFrames.push_back(actFrameIndex); // keep track of frames
                actFrameIndex = theLogger.logContext.frames[actFrameIndex].prev;
                actTabLevel--;
            }

            // if not found try again in reverse: from data back to last written
            if (actFrameIndex == -1) {
                actTabLevel = currentTabLevel;
                actFrameIndex = data.frameIndex;
                int backTrack = 0;
                while ((backTrack = findFrameInBacktrack(actFrameIndex)) < 0 && actFrameIndex != -1) {
                    frameIndexesToWrite.push_back(actFrameIndex);
                    actFrameIndex = theLogger.logContext.frames[actFrameIndex].prev;
                }
                if (actFrameIndex != -1) {
                    actTabLevel -= backTrack;
                }
            }

            // if still not found then lets go from the begininng
            if (actFrameIndex == -1) actTabLevel = 0;
            currentTabLevel = actTabLevel;

            for (int i = frameIndexesToWrite.size() - 1; i >= 0; --i) {
                auto& frame = theLogger.logContext.frames[frameIndexesToWrite[i]];
                auto out = desc.contextFormat;
                out = replaceTime(out, frame.time);
                out = replaceString(out, "{{Tick}}", std::to_string(frame.tick));
                out = replaceString(out, "{{Message}}", frame.message);
                std::string tab; tab.resize(currentTabLevel * 4, ' ');
                out = replaceString(out, "{{Indent}}", std::move(tab));
                if (!out.empty()) {
                    os << out << std::endl;
                    currentTabLevel = currentTabLevel + 1;
                }
            }
            lastFiredFrameIndex = data.frameIndex;

            // Write log
            auto out = desc.lineFormat;
            out = replaceTime(out, data.time);
            out = replaceString(out, "{{Tick}}", std::to_string(data.tick));
            out = replaceString(out, "{{Message}}", data.message);
            std::string tab; tab.resize(currentTabLevel * 4, ' ');
            out = replaceString(out, "{{Indent}}", std::move(tab));
            out = replaceString(out, "{{Level}}", theLogger.LogLevelToString(data.level));
            out = replaceString(out, "{{ThreadId}}", data.threadId);
            //out = replaceString(out, "{{Context}}", data.context);
            if (!out.empty()) {
                os << out << std::endl;
            }
        }
    }
    
    _impl::LogOutputFile::LogOutputFile(OutputDescriptor desc)
        : LogOutputStream{desc, file}
    {
        try {
            auto path = fs::path{desc.pathFormat};
            if (path.has_parent_path()) {
                fs::create_directories(path.remove_filename());
            }

            auto filename = fs::path{ desc.pathFormat }.filename().string();
            std::string invalidChars = R"(<>:"/\|?*)";
            for (auto invalidChar : invalidChars) {
                if (filename.find(invalidChar) != std::string::npos) {
                    std::stringstream ss;
                    ss << "Filename contains invalid character: '" << invalidChar << "'";
                    throw std::runtime_error(ss.str());
                }
            }

            file.open(desc.pathFormat, desc.append ? std::ifstream::app : std::ifstream::trunc);
            if (!file.is_open()) {
                throw std::runtime_error("Cant open the file, dont know why...");
            }
        } catch(std::exception& e) {
            std::cerr << "Could not create file: " << desc.pathFormat << std::endl << "Exception: " << e.what() << std::endl;
        }
    }

	void Logger::Log_Impl(LogLevel logLevel, std::string const& str)
	{
        if (!isInit()) return;
        _impl::LogData data;
        data.level = logLevel;
        std::stringstream tmp; tmp << std::this_thread::get_id();
        data.threadId = tmp.str();
        data.tick = tick;
        data.message = str;
        data.time = system_clock::now();
        data.frameIndex = logContext.actFrameIndex;

        memoryLog->Log(data);
        for (auto& it : outputLogs) {
            it->Log(data);
        }
	}

    void Logger::SetUsertag_Impl(const std::string& tag, const std::string& rep)
    {
        if (!isInit()) return;
        using namespace _impl;
        std::vector<_impl::OutputDescriptor> tmp;
        for (auto& it : outputDescriptors) {
            it.pathFormat = replaceString(it.pathFormat, "{$" + tag + "$}", rep);
            it.lineFormat = replaceString(it.lineFormat, "{$" + tag + "$}", rep);
            it.contextFormat = replaceString(it.contextFormat, "{$" + tag + "$}", rep);

            it.hasUserTag = false;
            it.hasUserTag |= it.pathFormat.find("{$", 0) != std::string::npos;
            it.hasUserTag |= it.lineFormat.find("{$", 0) != std::string::npos;
            it.hasUserTag |= it.contextFormat.find("{$", 0) != std::string::npos;
            
            if (!it.hasUserTag) {
                AddOutputChannel_Impl(std::make_unique<_impl::LogOutputFile>(it));
            } else {
                tmp.push_back(it);
            }
        }
        outputDescriptors.clear();
        outputDescriptors = std::move(tmp);
    }

    void Logger::AddOutputChannel_Impl(std::unique_ptr<_impl::ILogOutput> pOutput)
    {
        if (!isInit()) return;
        memoryLog->WriteDataToOutput(*pOutput);
        outputLogs.push_back(std::move(pOutput));
    }

	void Logger::StartNextTick()
	{
		++tick;
	}

	std::ostream& _impl::operator<<(std::ostream& os, LogLevel logLevel)
	{
        os << toString(logLevel);
        return os;
	}

    bool _impl::HasLogLevelBit(unsigned char bits, LogLevel logLevel) {
        // Csak akkor illeszkedik, ha az összes bit-re figyel 
        return (bits & (unsigned char)logLevel) == (unsigned char)logLevel;
    }
    
    LogLevel operator|(LogLevel lhs, LogLevel rhs) {
        return LogLevel((unsigned char)lhs | (unsigned char)rhs);
    }

#define MAKE_CASE(C) case LogLevel::C: if (first) first = false; else ret += "|"; ret += #C; break;
    std::string _impl::toString(LogLevel logLevel)
    {
        std::string ret = "";
        auto ll = (unsigned char)logLevel;
        bool first = true;
        for (int i = 0; i < 8; ++i) {
            auto bit = ll & (1 << i);
            switch ((LogLevel)bit)
            {
                LOG_LEVELS(NOOP, NOOP, MAKE_CASE, NOOP)
            default:
                continue;
            }
        }
        return ret;
    }
#undef MAKE_CASE

#define MAKE_CHAR_CASE(C) case C:
#define MAKE_LEVEL_RETURN(L) return (unsigned char)LogLevel::L;
    unsigned char _impl::fromChar(char c, std::string* pError)
    {
        switch (c) {
            LOG_LEVELS(MAKE_CHAR_CASE, NOOP, MAKE_LEVEL_RETURN, NOOP)
        default:
            if (pError) *pError = "Valid characters are {W|I|F|E|D}.";
            return 0;
        }
    }
#undef MAKE_CHAR_CASE
#undef MAKE_LEVEL_RETURN

#undef NOOP
}

#ifndef HAD_CRT_SUPRESSION
    #undef _CRT_SECURE_NO_WARNINGS
#endif

#endif
