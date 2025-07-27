#include "ultimate_memory.h"
#include "ultimate_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

/* Memory management structures */
typedef struct memory_block {
    size_t size;
    bool is_free;
    struct memory_block* next;
    struct memory_block* prev;
    uint32_t magic;
} memory_block_t;

typedef struct {
    void* heap_start;
    size_t heap_size;
    size_t free_size;
    size_t min_free_size;
    memory_block_t* free_list;
    bool initialized;
    uint32_t alloc_count;
    uint32_t free_count;
} heap_manager_t;

/* Global heap manager */
static heap_manager_t g_heap_manager = {0};

/* Memory pool structure */
typedef struct {
    void* pool_start;
    size_t block_size;
    size_t block_count;
    uint8_t* free_bitmap;
    size_t free_blocks;
    bool initialized;
} memory_pool_t;

/* Constants */
#define MEMORY_BLOCK_MAGIC 0xDEADBEEF
#define MAX_MEMORY_POOLS 16

static memory_pool_t g_memory_pools[MAX_MEMORY_POOLS];
static size_t g_pool_count = 0;

/* Memory debugging */
#ifdef DEBUG
typedef struct alloc_info {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    const char* function;
    struct alloc_info* next;
} alloc_info_t;

static alloc_info_t* g_alloc_list = NULL;
static size_t g_total_allocated = 0;
static bool g_leak_check_enabled = false;
#endif

/* Memory API Implementation */

ultimate_error_t ultimate_memory_init(void* heap_start, size_t heap_size) {
    if (g_heap_manager.initialized) {
        return ULTIMATE_ERROR_ALREADY_INITIALIZED;
    }
    
    // Use default heap size if not specified
    if (heap_size == 0) {
        heap_size = ULTIMATE_HEAP_SIZE;
    }
    
    // Allocate heap if not provided
    if (heap_start == NULL) {
#ifdef _WIN32
        heap_start = VirtualAlloc(NULL, heap_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!heap_start) {
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
#else
        heap_start = mmap(NULL, heap_size, PROT_READ | PROT_WRITE, 
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (heap_start == MAP_FAILED) {
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
#endif
    }
    
    // Initialize heap manager
    g_heap_manager.heap_start = heap_start;
    g_heap_manager.heap_size = heap_size;
    g_heap_manager.free_size = heap_size - sizeof(memory_block_t);
    g_heap_manager.min_free_size = g_heap_manager.free_size;
    g_heap_manager.initialized = true;
    g_heap_manager.alloc_count = 0;
    g_heap_manager.free_count = 0;
    
    // Initialize first free block
    memory_block_t* first_block = (memory_block_t*)heap_start;
    first_block->size = heap_size - sizeof(memory_block_t);
    first_block->is_free = true;
    first_block->next = NULL;
    first_block->prev = NULL;
    first_block->magic = MEMORY_BLOCK_MAGIC;
    
    g_heap_manager.free_list = first_block;
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_memory_deinit(void) {
    if (!g_heap_manager.initialized) {
        return ULTIMATE_ERROR_NOT_INITIALIZED;
    }
    
#ifdef _WIN32
    VirtualFree(g_heap_manager.heap_start, 0, MEM_RELEASE);
#else
    munmap(g_heap_manager.heap_start, g_heap_manager.heap_size);
#endif
    
    memset(&g_heap_manager, 0, sizeof(heap_manager_t));
    
    return ULTIMATE_ERROR_SUCCESS;
}

void* ultimate_malloc(size_t size) {
    if (!g_heap_manager.initialized || size == 0) {
        return NULL;
    }
    
    // Align size
    size = ULTIMATE_MEMORY_ALIGN(size);
    
    // Find suitable free block
    memory_block_t* current = g_heap_manager.free_list;
    while (current) {
        if (current->is_free && current->size >= size) {
            // Split block if necessary
            if (current->size > size + sizeof(memory_block_t) + ULTIMATE_MEMORY_ALIGNMENT) {
                memory_block_t* new_block = (memory_block_t*)((uint8_t*)current + sizeof(memory_block_t) + size);
                new_block->size = current->size - size - sizeof(memory_block_t);
                new_block->is_free = true;
                new_block->next = current->next;
                new_block->prev = current;
                new_block->magic = MEMORY_BLOCK_MAGIC;
                
                if (current->next) {
                    current->next->prev = new_block;
                }
                current->next = new_block;
                current->size = size;
            }
            
            current->is_free = false;
            g_heap_manager.free_size -= current->size;
            g_heap_manager.alloc_count++;
            
            if (g_heap_manager.free_size < g_heap_manager.min_free_size) {
                g_heap_manager.min_free_size = g_heap_manager.free_size;
            }
            
            return (uint8_t*)current + sizeof(memory_block_t);
        }
        current = current->next;
    }
    
    return NULL; // Out of memory
}

void* ultimate_calloc(size_t count, size_t size) {
    size_t total_size = count * size;
    void* ptr = ultimate_malloc(total_size);
    if (ptr) {
        memset(ptr, 0, total_size);
    }
    return ptr;
}

void* ultimate_realloc(void* ptr, size_t new_size) {
    if (!ptr) {
        return ultimate_malloc(new_size);
    }
    
    if (new_size == 0) {
        ultimate_free(ptr);
        return NULL;
    }
    
    memory_block_t* block = (memory_block_t*)((uint8_t*)ptr - sizeof(memory_block_t));
    if (block->magic != MEMORY_BLOCK_MAGIC || block->is_free) {
        return NULL; // Invalid pointer
    }
    
    if (block->size >= new_size) {
        return ptr; // Current block is large enough
    }
    
    // Allocate new block and copy data
    void* new_ptr = ultimate_malloc(new_size);
    if (new_ptr) {
        memcpy(new_ptr, ptr, block->size < new_size ? block->size : new_size);
        ultimate_free(ptr);
    }
    
    return new_ptr;
}

void ultimate_free(void* ptr) {
    if (!ptr || !g_heap_manager.initialized) {
        return;
    }
    
    memory_block_t* block = (memory_block_t*)((uint8_t*)ptr - sizeof(memory_block_t));
    if (block->magic != MEMORY_BLOCK_MAGIC || block->is_free) {
        return; // Invalid pointer or already freed
    }
    
    block->is_free = true;
    g_heap_manager.free_size += block->size;
    g_heap_manager.free_count++;
    
    // Coalesce with adjacent free blocks
    if (block->next && block->next->is_free) {
        block->size += block->next->size + sizeof(memory_block_t);
        if (block->next->next) {
            block->next->next->prev = block;
        }
        block->next = block->next->next;
    }
    
    if (block->prev && block->prev->is_free) {
        block->prev->size += block->size + sizeof(memory_block_t);
        if (block->next) {
            block->next->prev = block->prev;
        }
        block->prev->next = block->next;
    }
}

void* ultimate_aligned_malloc(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL; // Alignment must be power of 2
    }
    
    size_t total_size = size + alignment + sizeof(void*);
    void* raw_ptr = ultimate_malloc(total_size);
    if (!raw_ptr) {
        return NULL;
    }
    
    uintptr_t aligned_addr = ((uintptr_t)raw_ptr + sizeof(void*) + alignment - 1) & ~(alignment - 1);
    void** aligned_ptr = (void**)aligned_addr;
    aligned_ptr[-1] = raw_ptr;
    
    return (void*)aligned_addr;
}

void ultimate_aligned_free(void* ptr) {
    if (!ptr) {
        return;
    }
    
    void** aligned_ptr = (void**)ptr;
    ultimate_free(aligned_ptr[-1]);
}

/* Memory utility functions */
void* ultimate_memcpy(void* dest, const void* src, size_t count) {
    return memcpy(dest, src, count);
}

void* ultimate_memset(void* dest, int value, size_t count) {
    return memset(dest, value, count);
}

int ultimate_memcmp(const void* ptr1, const void* ptr2, size_t count) {
    return memcmp(ptr1, ptr2, count);
}

void* ultimate_memmove(void* dest, const void* src, size_t count) {
    return memmove(dest, src, count);
}

/* Memory validation functions */
bool ultimate_memory_is_valid_ptr(const void* ptr) {
    if (!ptr || !g_heap_manager.initialized) {
        return false;
    }
    
    uintptr_t heap_start = (uintptr_t)g_heap_manager.heap_start;
    uintptr_t heap_end = heap_start + g_heap_manager.heap_size;
    uintptr_t ptr_addr = (uintptr_t)ptr;
    
    return (ptr_addr >= heap_start && ptr_addr < heap_end);
}

bool ultimate_memory_is_heap_ptr(const void* ptr) {
    return ultimate_memory_is_valid_ptr(ptr);
}

/* Memory statistics */
ultimate_error_t ultimate_memory_get_heap_stats(ultimate_heap_stats_t* stats) {
    if (!stats || !g_heap_manager.initialized) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    stats->total_size = g_heap_manager.heap_size;
    stats->free_size = g_heap_manager.free_size;
    stats->used_size = g_heap_manager.heap_size - g_heap_manager.free_size;
    stats->min_free_size = g_heap_manager.min_free_size;
    stats->alloc_count = g_heap_manager.alloc_count;
    stats->free_count = g_heap_manager.free_count;
    stats->fragmentation_percent = (float)(stats->used_size) / stats->total_size * 100.0f;
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_memory_check_integrity(void) {
    if (!g_heap_manager.initialized) {
        return ULTIMATE_ERROR_NOT_INITIALIZED;
    }
    
    memory_block_t* current = (memory_block_t*)g_heap_manager.heap_start;
    size_t total_size = 0;
    
    while ((uint8_t*)current < (uint8_t*)g_heap_manager.heap_start + g_heap_manager.heap_size) {
        if (current->magic != MEMORY_BLOCK_MAGIC) {
            return ULTIMATE_ERROR_MEMORY_CORRUPTION;
        }
        
        total_size += current->size + sizeof(memory_block_t);
        
        if (current->next) {
            current = current->next;
        } else {
            break;
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Memory pool functions */
ultimate_error_t ultimate_pool_create(const ultimate_pool_config_t* config,
                                     ultimate_pool_handle_t* pool) {
    if (!config || !pool || g_pool_count >= MAX_MEMORY_POOLS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    memory_pool_t* new_pool = &g_memory_pools[g_pool_count];
    
    size_t total_size = config->block_size * config->block_count;
    new_pool->pool_start = ultimate_malloc(total_size);
    if (!new_pool->pool_start) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    new_pool->block_size = config->block_size;
    new_pool->block_count = config->block_count;
    new_pool->free_blocks = config->block_count;
    
    // Initialize free bitmap
    size_t bitmap_size = (config->block_count + 7) / 8;
    new_pool->free_bitmap = (uint8_t*)ultimate_malloc(bitmap_size);
    if (!new_pool->free_bitmap) {
        ultimate_free(new_pool->pool_start);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    memset(new_pool->free_bitmap, 0, bitmap_size);
    new_pool->initialized = true;
    
    *pool = (ultimate_pool_handle_t)(g_pool_count + 1);
    g_pool_count++;
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_pool_delete(ultimate_pool_handle_t pool) {
    size_t pool_index = (size_t)pool - 1;
    if (pool_index >= g_pool_count || !g_memory_pools[pool_index].initialized) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    memory_pool_t* pool_ptr = &g_memory_pools[pool_index];
    ultimate_free(pool_ptr->pool_start);
    ultimate_free(pool_ptr->free_bitmap);
    memset(pool_ptr, 0, sizeof(memory_pool_t));
    
    return ULTIMATE_ERROR_SUCCESS;
}

void* ultimate_pool_alloc(ultimate_pool_handle_t pool) {
    size_t pool_index = (size_t)pool - 1;
    if (pool_index >= g_pool_count || !g_memory_pools[pool_index].initialized) {
        return NULL;
    }
    
    memory_pool_t* pool_ptr = &g_memory_pools[pool_index];
    if (pool_ptr->free_blocks == 0) {
        return NULL;
    }
    
    // Find first free block
    for (size_t i = 0; i < pool_ptr->block_count; i++) {
        size_t byte_index = i / 8;
        size_t bit_index = i % 8;
        
        if (!(pool_ptr->free_bitmap[byte_index] & (1 << bit_index))) {
            // Mark as allocated
            pool_ptr->free_bitmap[byte_index] |= (1 << bit_index);
            pool_ptr->free_blocks--;
            
            return (uint8_t*)pool_ptr->pool_start + (i * pool_ptr->block_size);
        }
    }
    
    return NULL;
}

ultimate_error_t ultimate_pool_free(ultimate_pool_handle_t pool, void* ptr) {
    size_t pool_index = (size_t)pool - 1;
    if (pool_index >= g_pool_count || !g_memory_pools[pool_index].initialized || !ptr) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    memory_pool_t* pool_ptr = &g_memory_pools[pool_index];
    
    // Calculate block index
    uintptr_t offset = (uintptr_t)ptr - (uintptr_t)pool_ptr->pool_start;
    if (offset % pool_ptr->block_size != 0 || offset >= pool_ptr->block_count * pool_ptr->block_size) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t block_index = offset / pool_ptr->block_size;
    size_t byte_index = block_index / 8;
    size_t bit_index = block_index % 8;
    
    // Check if already free
    if (!(pool_ptr->free_bitmap[byte_index] & (1 << bit_index))) {
        return ULTIMATE_ERROR_INVALID_PARAMETER; // Double free
    }
    
    // Mark as free
    pool_ptr->free_bitmap[byte_index] &= ~(1 << bit_index);
    pool_ptr->free_blocks++;
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Fast memory operations */
void ultimate_memory_copy_fast(void* dest, const void* src, size_t count) {
    // Use optimized memory copy if available
#ifdef _WIN32
    CopyMemory(dest, src, count);
#else
    memcpy(dest, src, count);
#endif
}

void ultimate_memory_set_fast(void* dest, int value, size_t count) {
    memset(dest, value, count);
}

/* Memory barriers */
void ultimate_memory_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __sync_synchronize();
#endif
}

void ultimate_memory_read_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __asm__ __volatile__("lfence" ::: "memory");
#endif
}

void ultimate_memory_write_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __asm__ __volatile__("sfence" ::: "memory");
#endif
}

/* System information */
uint32_t ultimate_system_get_free_heap_size(void) {
    return (uint32_t)g_heap_manager.free_size;
}

uint32_t ultimate_system_get_min_free_heap_size(void) {
    return (uint32_t)g_heap_manager.min_free_size;
}