// Native installer operations. No shell, service, autostart or resident process.
// Process shutdown is restricted to known executables in the installation paths.
#define _WIN32_WINNT 0x0601
#include <windows.h>
#include <tlhelp32.h>
#include <restartmanager.h>
#include <initguid.h>
#include <netfw.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef IDLETOKEN_FIREWALL_PREFIX
#define IDLETOKEN_FIREWALL_PREFIX L"IdleToken"
#endif

static const wchar_t *const executables[] = {
    L"idletoken-client.exe", L"idletoken-coord.exe", L"idletoken-worker.exe",
    L"idletoken-platform-agent.exe", L"idletoken-server.exe", L"idletoken-rpc-server.exe",
};

static void check(HRESULT status, const char *operation) {
    if (FAILED(status)) {
        char message[192];
        std::snprintf(message, sizeof(message), "%s failed (HRESULT 0x%08lx)", operation, (unsigned long)status);
        throw std::runtime_error(message);
    }
}

static std::wstring full_path(const std::wstring &path) {
    std::vector<wchar_t> buffer(32768);
    DWORD length = GetFullPathNameW(path.c_str(), (DWORD)buffer.size(), buffer.data(), nullptr);
    if (!length || length >= buffer.size()) throw std::runtime_error("Invalid installation path");
    std::wstring result(buffer.data(), length);
    HANDLE file = CreateFileW(result.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        length = GetFinalPathNameByHandleW(file, buffer.data(), (DWORD)buffer.size(), FILE_NAME_NORMALIZED);
        CloseHandle(file);
        if (!length || length >= buffer.size()) throw std::runtime_error("Cannot resolve installation path");
        result.assign(buffer.data(), length);
        if (result.compare(0, 8, L"\\\\?\\UNC\\") == 0) result = L"\\\\" + result.substr(8);
        else if (result.compare(0, 4, L"\\\\?\\") == 0) result.erase(0, 4);
    }
    while (result.size() > 3 && result.back() == L'\\') result.pop_back();
    return result;
}

static bool same_path(const std::wstring &left, const std::wstring &right) {
    return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_EQUAL;
}

struct process {
    HANDLE handle;
    RM_UNIQUE_PROCESS identity;
};

static std::vector<process> owned_processes(const std::vector<std::wstring> &directories) {
    std::vector<std::wstring> paths;
    for (const auto &directory : directories)
        for (const auto *name : executables) paths.push_back(full_path(directory + L"\\" + name));
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot enumerate running applications");
    std::vector<process> found;
    PROCESSENTRY32W entry = {};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Process32FirstW(snapshot, &entry); more; more = Process32NextW(snapshot, &entry)) {
        bool known = false;
        for (const auto *name : executables) if (same_path(entry.szExeFile, name)) known = true;
        if (!known) continue;
        HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE | PROCESS_TERMINATE,
                                    FALSE, entry.th32ProcessID);
        if (!handle) continue; // The installer will fail if a locked target cannot be replaced.
        std::vector<wchar_t> image(32768);
        DWORD length = (DWORD)image.size();
        bool owned = false;
        if (QueryFullProcessImageNameW(handle, 0, image.data(), &length)) {
            const auto actual = full_path(std::wstring(image.data(), length));
            for (const auto &path : paths) if (same_path(actual, path)) owned = true;
        }
        FILETIME created, exited, kernel, user;
        if (owned && GetProcessTimes(handle, &created, &exited, &kernel, &user)) {
            found.push_back({handle, {entry.th32ProcessID, created}});
        } else CloseHandle(handle);
    }
    CloseHandle(snapshot);
    return found;
}

static void stop(const std::vector<std::wstring> &directories) {
    auto found = owned_processes(directories);
    if (found.empty()) return;
    // Restart Manager provides the normal Windows application shutdown protocol.
    // Register PID + creation time, not names, so unrelated installations survive.
    DWORD session = 0;
    WCHAR key[CCH_RM_SESSION_KEY + 1] = {};
    if (RmStartSession(&session, 0, key) == ERROR_SUCCESS) {
        std::vector<RM_UNIQUE_PROCESS> identities;
        for (const auto &item : found) identities.push_back(item.identity);
        if (RmRegisterResources(session, 0, nullptr, (UINT)identities.size(), identities.data(), 0, nullptr) == ERROR_SUCCESS) {
            DWORD result = RmShutdown(session, 0, nullptr);
            std::printf("Installer: application shutdown result %lu\n", (unsigned long)result);
        }
        RmEndSession(session);
    }
    for (const auto &item : found) CloseHandle(item.handle);
    // A supervisor can still be exiting when its child is inspected. Re-enumerate
    // exact paths and retain process handles while stopping, avoiding PID reuse.
    const ULONGLONG deadline = GetTickCount64() + 15000;
    unsigned empty_checks = 0;
    do {
        found = owned_processes(directories);
        if (found.empty()) {
            if (++empty_checks == 2) return;
        } else {
            empty_checks = 0;
            for (const auto &item : found) {
                if (WaitForSingleObject(item.handle, 0) == WAIT_TIMEOUT) {
                    std::printf("Installer: stopping remaining IdleToken process %lu\n",
                                (unsigned long)item.identity.dwProcessId);
                    TerminateProcess(item.handle, 0);
                    WaitForSingleObject(item.handle, 1000);
                }
                CloseHandle(item.handle);
            }
        }
        Sleep(250);
    } while (GetTickCount64() < deadline);
    throw std::runtime_error("IdleToken processes did not stop; close the application and retry");
}

template<class T> struct com_ptr {
    T *value = nullptr;
    ~com_ptr() { if (value) value->Release(); }
    T *operator->() const { return value; }
};
struct text {
    BSTR value;
    explicit text(const std::wstring &s) : value(SysAllocString(s.c_str())) {
        if (!value) throw std::bad_alloc();
    }
    ~text() { SysFreeString(value); }
    operator BSTR() const { return value; }
};
struct firewall_rule {
    const wchar_t *name;
    const wchar_t *executable;
    long protocol;
    const wchar_t *remote;
};
static const firewall_rule rules[] = {
    {L"coordinator", L"idletoken-coord.exe", NET_FW_IP_PROTOCOL_TCP, L"LocalSubnet"},
    {L"coordinator discovery", L"idletoken-coord.exe", NET_FW_IP_PROTOCOL_UDP, L"LocalSubnet"},
    {L"worker", L"idletoken-worker.exe", NET_FW_IP_PROTOCOL_UDP, L"LocalSubnet"},
    {L"compute node", L"idletoken-rpc-server.exe", NET_FW_IP_PROTOCOL_TCP, L"LocalSubnet"},
    // Pairing can also use a private overlay; tensor traffic remains LAN-only.
    {L"app", L"idletoken-client.exe", NET_FW_IP_PROTOCOL_TCP, L"*"},
    {L"app discovery", L"idletoken-client.exe", NET_FW_IP_PROTOCOL_UDP, L"*"},
};

static void firewall(const std::wstring &directory, bool install) {
    check(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED), "COM initialization");
    struct cleanup { ~cleanup() { CoUninitialize(); } } cleanup;
    com_ptr<INetFwPolicy2> policy;
    check(CoCreateInstance(CLSID_NetFwPolicy2, nullptr, CLSCTX_INPROC_SERVER,
        IID_INetFwPolicy2, (void **)&policy.value), "Windows Firewall policy");
    com_ptr<INetFwRules> collection;
    check(policy->get_Rules(&collection.value), "Windows Firewall rules");
    for (const auto &definition : rules) {
        const text name(std::wstring(IDLETOKEN_FIREWALL_PREFIX) + L" " + definition.name);
        const text application(full_path(directory + L"\\" + definition.executable));
        if (!install) {
            com_ptr<INetFwRule> old;
            if (SUCCEEDED(collection->Item(name, &old.value))) {
                BSTR old_path = nullptr;
                check(old->get_ApplicationName(&old_path), "Read firewall application path");
                bool owned = old_path && same_path(full_path(old_path), (BSTR)application);
                SysFreeString(old_path);
                if (owned) check(collection->Remove(name), "Remove IdleToken firewall rule");
            }
            continue;
        }
        if (GetFileAttributesW(application) == INVALID_FILE_ATTRIBUTES)
            throw std::runtime_error("Cannot create a firewall rule for a missing application");
        com_ptr<INetFwRule> rule;
        check(CoCreateInstance(CLSID_NetFwRule, nullptr, CLSCTX_INPROC_SERVER,
            IID_INetFwRule, (void **)&rule.value), "Windows Firewall rule");
        check(rule->put_Name(name), "Set firewall rule name");
        check(rule->put_Description(text(L"IdleToken cluster connections on private networks.")), "Set rule description");
        check(rule->put_Grouping(text(IDLETOKEN_FIREWALL_PREFIX)), "Set rule group");
        check(rule->put_ApplicationName(application), "Set rule application");
        check(rule->put_Protocol(definition.protocol), "Set rule protocol");
        check(rule->put_RemoteAddresses(text(definition.remote)), "Set rule network scope");
        check(rule->put_Direction(NET_FW_RULE_DIR_IN), "Set rule direction");
        check(rule->put_Profiles(NET_FW_PROFILE2_PRIVATE | NET_FW_PROFILE2_DOMAIN), "Set rule profiles");
        check(rule->put_Action(NET_FW_ACTION_ALLOW), "Set rule action");
        check(rule->put_Enabled(VARIANT_TRUE), "Enable rule");
        // Remove the earlier broad rule before replacing its definition.
        com_ptr<INetFwRule> old;
        if (SUCCEEDED(collection->Item(name, &old.value)))
            check(collection->Remove(name), "Replace IdleToken firewall rule");
        check(collection->Add(rule.value), "Add IdleToken firewall rule");
    }
}

int wmain(int argc, wchar_t **argv) {
    try {
        if (argc < 3) throw std::runtime_error("Usage: idletoken-installer-helper --stop|--firewall-install|--firewall-remove <installation-directory> [legacy-directory]");
        const std::wstring command(argv[1]);
        if (command == L"--stop") {
            std::vector<std::wstring> directories;
            for (int i = 2; i < argc; ++i) directories.push_back(full_path(argv[i]));
            stop(directories);
        } else if (argc == 3 && (command == L"--firewall-install" || command == L"--firewall-remove")) {
            firewall(full_path(argv[2]), command == L"--firewall-install");
        } else throw std::runtime_error("Unknown installer operation");
        std::puts("INSTALLER_HELPER_OK");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INSTALLER_HELPER_FAIL: %s\n", error.what());
        return 1;
    }
}
