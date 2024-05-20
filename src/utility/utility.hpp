#ifndef VULKAN_INTRO_UTILITY_HPP
#define VULKAN_INTRO_UTILITY_HPP

class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES* operator&();
    ~WindowsSecurityAttributes();
};

inline void hash_combine([[maybe_unused]] std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine(seed, rest...);
}

inline void hash_combine_raw([[maybe_unused]] std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine_raw(std::size_t& seed, const T& v, Rest... rest) {
    seed ^= v + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine(seed, rest...);
}

void* mAlignedAlloc(size_t size, size_t alignment);
void mAlignedFree(void* ptr);

size_t getAlignedSize(size_t size, size_t alignment);

template <typename T>
class Indexer {
    void* ptr;
    unsigned int stride;
public:
    static Indexer make(void* ptr, unsigned int stride) {
        if (stride == 0) stride = sizeof(T);
        Indexer i;
        i.ptr = ptr;
        i.stride = stride;
        return i;
    }
    T& operator[](size_t index) {
        return *std::launder(reinterpret_cast<T*>((char*)ptr + stride * index));
    }

    const T& operator[](size_t index) const {
        return *std::launder(reinterpret_cast<T*>((char*)ptr + stride * index));
    }

    auto begin() { return &(*this)[0]; }

    using value_t = T;
};

template <typename F>
struct ConvertingIndexer {
    void* ptr;
    unsigned int stride;
    const F& convert;

    using value_t = typename std::invoke_result<F, void*>::type;

    ConvertingIndexer(void* ptr, unsigned int stride, const F& converter) : ptr{ptr}, stride{stride}, convert{converter} { assert (stride > 0); }

    auto operator[](size_t index) const {
        return convert((char*)ptr + stride * index);
    }
};

template <typename F>
auto makeConvertingIndexer(void* ptr, unsigned int stride, const F& converter) {
    return ConvertingIndexer<F>{ptr, stride, converter};
}

class DummyIndexer {
public:
    size_t operator[](size_t index) const {
        return index;
    }
};

template <typename Value, typename Index>
struct IndexedIndexer {
    Value val;
    Index index;

    typename Value::value_t& operator[](size_t idx) {
        return val[(size_t)index[idx]];
    }

    typename Value::value_t operator[](size_t idx) const {
        return val[(size_t)index[idx]];
    }
};

template <typename Value, typename Index>
auto makeIndexedIndexer(Value&& values, Index&& index) {
    return IndexedIndexer<typename std::remove_reference<Value>::type, typename std::remove_reference<Index>::type>{std::forward<Value>(values), std::forward<Index>(index)};
}

template <typename F, typename T>
void copy(const F& from, size_t indexCount, T& to) {
    for (size_t i = 0; i < indexCount; ++i) {
        to[i] = from[i];
    }
}

template <class T, class S, class C>
S& Container(std::priority_queue<T, S, C>& q) {
    struct HackedQueue : private std::priority_queue<T, S, C> {
        static S& Container(std::priority_queue<T, S, C>& q) {
            return q.*&HackedQueue::c;
        }
    };
    return HackedQueue::Container(q);
}

std::string readFile(const char* path);
void writeFile(const char* path, const std::string& content);

// trim from start (in place)
static inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string& s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

template <typename T>
struct reversion_wrapper { T& iterable; };

template <typename T>
auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

template <typename T>
auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

template <typename T>
reversion_wrapper<T> reverse(T&& iterable) { return { iterable }; }


#endif//VULKAN_INTRO_UTILITY_HPP
