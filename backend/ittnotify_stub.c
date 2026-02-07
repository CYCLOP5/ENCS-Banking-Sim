// Minimal stubs for Intel ITT JIT profiling symbols that some PyTorch builds
// expect at dynamic link time. These are safe no-ops.
//
// Build:
//   gcc -shared -fPIC -O2 -o libittnotify_stub.so ittnotify_stub.c

#include <stdint.h>

__attribute__((visibility("default")))
unsigned int iJIT_GetNewMethodID(void) {
    return 0;
}

__attribute__((visibility("default")))
int iJIT_IsProfilingActive(void) {
    return 0;
}

__attribute__((visibility("default")))
int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}
