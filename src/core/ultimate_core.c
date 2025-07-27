#include "ultimate_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#endif

/* Global system state */
static ultimate_state_t g_system_state = ULTIMATE_STATE_UNINITIALIZED;
static ultimate_init_config_t g_system_config;
static uint32_t g_system_start_time = 0;
static bool g_critical_section_active = false;

#ifdef _WIN32
static CRITICAL_SECTION g_critical_section;
static LARGE_INTEGER g_performance_frequency;
static LARGE_INTEGER g_performance_counter_start;
#else
static pthread_mutex_t g_critical_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct timeval g_start_time;
#endif

/* Core API Implementation */

ultimate_error_t ultimate_init(const ultimate_init_config_t* config) {
    if (g_system_state != ULTIMATE_STATE_UNINITIALIZED) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    if (!config) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    g_system_state = ULTIMATE_STATE_INITIALIZING;
    
    // Copy configuration
    memcpy(&g_system_config, config, sizeof(ultimate_init_config_t));
    
    // Initialize platform-specific components
#ifdef _WIN32
    InitializeCriticalSection(&g_critical_section);
    
    // Initialize high-resolution timer
    if (!QueryPerformanceFrequency(&g_performance_frequency)) {
        g_system_state = ULTIMATE_STATE_ERROR;
        return ULTIMATE_ERROR_HARDWARE_FAILURE;
    }
    
    QueryPerformanceCounter(&g_performance_counter_start);
#else
    gettimeofday(&g_start_time, NULL);
#endif
    
    // Initialize memory system
    ultimate_error_t error = ultimate_memory_init(NULL, 0);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        g_system_state = ULTIMATE_STATE_ERROR;
        return error;
    }
    
    g_system_start_time = ultimate_get_time_ms();
    g_system_state = ULTIMATE_STATE_READY;
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_start(void) {
    if (g_system_state != ULTIMATE_STATE_READY) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    g_system_state = ULTIMATE_STATE_RUNNING;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_stop(void) {
    if (g_system_state != ULTIMATE_STATE_RUNNING) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    g_system_state = ULTIMATE_STATE_READY;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_shutdown(void) {
    if (g_system_state == ULTIMATE_STATE_UNINITIALIZED) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    // Cleanup memory system
    ultimate_memory_deinit();
    
    // Cleanup platform-specific components
#ifdef _WIN32
    DeleteCriticalSection(&g_critical_section);
#endif
    
    g_system_state = ULTIMATE_STATE_UNINITIALIZED;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_state_t ultimate_get_state(void) {
    return g_system_state;
}

uint32_t ultimate_get_version(void) {
    return (ULTIMATE_VERSION_MAJOR << 24) | 
           (ULTIMATE_VERSION_MINOR << 16) | 
           (ULTIMATE_VERSION_PATCH << 8);
}

const char* ultimate_get_version_string(void) {
    return ULTIMATE_VERSION_STRING;
}

uint32_t ultimate_get_tick_count(void) {
#ifdef _WIN32
    return GetTickCount();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)((tv.tv_sec - g_start_time.tv_sec) * 1000 + 
                      (tv.tv_usec - g_start_time.tv_usec) / 1000);
#endif
}

uint32_t ultimate_get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER current_counter;
    QueryPerformanceCounter(&current_counter);
    
    return (uint32_t)(((current_counter.QuadPart - g_performance_counter_start.QuadPart) * 1000) / 
                      g_performance_frequency.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)((tv.tv_sec - g_start_time.tv_sec) * 1000 + 
                      (tv.tv_usec - g_start_time.tv_usec) / 1000);
#endif
}

void ultimate_delay_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

void ultimate_delay_us(uint32_t us) {
#ifdef _WIN32
    // Windows doesn't have microsecond precision sleep, use high-resolution timer
    LARGE_INTEGER start, current, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    uint64_t target_ticks = (uint64_t)us * frequency.QuadPart / 1000000;
    
    do {
        QueryPerformanceCounter(&current);
    } while ((current.QuadPart - start.QuadPart) < target_ticks);
#else
    usleep(us);
#endif
}

void ultimate_enter_critical(void) {
#ifdef _WIN32
    EnterCriticalSection(&g_critical_section);
#else
    pthread_mutex_lock(&g_critical_mutex);
#endif
    g_critical_section_active = true;
}

void ultimate_exit_critical(void) {
    g_critical_section_active = false;
#ifdef _WIN32
    LeaveCriticalSection(&g_critical_section);
#else
    pthread_mutex_unlock(&g_critical_mutex);
#endif
}

void ultimate_system_reset(void) {
    // Perform soft reset
    if (g_system_state == ULTIMATE_STATE_RUNNING) {
        ultimate_stop();
    }
    
    if (g_system_state == ULTIMATE_STATE_READY) {
        ultimate_start();
    }
}

void ultimate_system_recovery(void) {
    // Attempt to recover from error state
    if (g_system_state == ULTIMATE_STATE_ERROR) {
        g_system_state = ULTIMATE_STATE_READY;
    }
}