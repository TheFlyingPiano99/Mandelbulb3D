#include "utility.hpp"
#include <AclAPI.h>

WindowsSecurityAttributes::WindowsSecurityAttributes() {
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
        1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));

    PSID* ppSID =
        (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor,
        SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
        SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
        0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions =
        STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

    SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

    m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
    m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
    return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID* ppSID =
        (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(m_winPSecurityDescriptor);
}

void* mAlignedAlloc(size_t size, size_t alignment)
{
#ifdef __unix__
    return aligned_alloc(alignment, size);
#elif defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#endif
}

void mAlignedFree(void* ptr)
{
#ifdef __unix__
    return free(ptr);
#elif defined(_WIN32) || defined(_WIN64)
    return _aligned_free(ptr);
#endif
}

size_t getAlignedSize(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

std::string readFile(const char* path)
{
    std::ostringstream buf;
    std::ifstream input(path);
    if (!input.is_open()) {
        logger.LogError("Could not open file for reading: ", path);
        return "";
    }
    buf << input.rdbuf();
    return buf.str();
}

void writeFile(const char* path, const std::string& content)
{
    std::ofstream o{ path, std::ofstream::out | std::ofstream::trunc };
    if (!o.is_open()) {
        logger.LogError("Could not open file for writing: ", path);
        return;
    }
    o << content;
}


