#include "graphics/ultimate_image.h"
#include "ultimate_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Global configuration */
static int32_t g_thread_count = 1;
static bool g_gpu_acceleration = false;
static uint64_t g_operations_count = 0;
static uint64_t g_total_time_ms = 0;

/* Helper macros */
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define PI 3.14159265358979323846

/* Image format information */
static const struct {
    ultimate_image_format_t format;
    uint32_t channels;
    uint32_t depth;
    const char* name;
} format_info[] = {
    {ULTIMATE_IMAGE_FORMAT_RGB24, 3, 8, "RGB24"},
    {ULTIMATE_IMAGE_FORMAT_RGBA32, 4, 8, "RGBA32"},
    {ULTIMATE_IMAGE_FORMAT_BGR24, 3, 8, "BGR24"},
    {ULTIMATE_IMAGE_FORMAT_BGRA32, 4, 8, "BGRA32"},
    {ULTIMATE_IMAGE_FORMAT_GRAY8, 1, 8, "GRAY8"},
    {ULTIMATE_IMAGE_FORMAT_GRAY16, 1, 16, "GRAY16"},
    {ULTIMATE_IMAGE_FORMAT_RGB48, 3, 16, "RGB48"},
    {ULTIMATE_IMAGE_FORMAT_RGBA64, 4, 16, "RGBA64"},
    {ULTIMATE_IMAGE_FORMAT_YUV420, 3, 8, "YUV420"},
    {ULTIMATE_IMAGE_FORMAT_YUV422, 3, 8, "YUV422"},
    {ULTIMATE_IMAGE_FORMAT_YUV444, 3, 8, "YUV444"},
    {ULTIMATE_IMAGE_FORMAT_HSV, 3, 8, "HSV"},
    {ULTIMATE_IMAGE_FORMAT_LAB, 3, 8, "LAB"},
    {ULTIMATE_IMAGE_FORMAT_XYZ, 3, 8, "XYZ"}
};

/* Helper functions */
static uint32_t get_format_channels(ultimate_image_format_t format) {
    for (size_t i = 0; i < sizeof(format_info) / sizeof(format_info[0]); i++) {
        if (format_info[i].format == format) {
            return format_info[i].channels;
        }
    }
    return 0;
}

static uint32_t get_format_depth(ultimate_image_format_t format) {
    for (size_t i = 0; i < sizeof(format_info) / sizeof(format_info[0]); i++) {
        if (format_info[i].format == format) {
            return format_info[i].depth;
        }
    }
    return 0;
}

static size_t get_pixel_size(ultimate_image_format_t format) {
    uint32_t channels = get_format_channels(format);
    uint32_t depth = get_format_depth(format);
    return (channels * depth + 7) / 8;
}

/* Core Image Operations */

ultimate_error_t ultimate_image_create(ultimate_image_t** image, 
                                      uint32_t width, uint32_t height, 
                                      ultimate_image_format_t format) {
    if (!image || width == 0 || height == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_image_t* img = (ultimate_image_t*)ultimate_malloc(sizeof(ultimate_image_t));
    if (!img) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    img->width = width;
    img->height = height;
    img->channels = get_format_channels(format);
    img->depth = get_format_depth(format);
    img->format = format;
    
    if (img->channels == 0 || img->depth == 0) {
        ultimate_free(img);
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t pixel_size = get_pixel_size(format);
    img->stride = width * pixel_size;
    img->data_size = img->stride * height;
    
    img->data = (uint8_t*)ultimate_malloc(img->data_size);
    if (!img->data) {
        ultimate_free(img);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    memset(img->data, 0, img->data_size);
    img->metadata = NULL;
    
    *image = img;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_destroy(ultimate_image_t* image) {
    if (!image) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (image->data) {
        ultimate_free(image->data);
    }
    
    if (image->metadata) {
        ultimate_free(image->metadata);
    }
    
    ultimate_free(image);
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_clone(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    memcpy((*dst)->data, src->data, src->data_size);
    return ULTIMATE_ERROR_SUCCESS;
}

/* Image I/O - Basic implementations */
ultimate_error_t ultimate_image_load(const char* filename, ultimate_image_t** image) {
    if (!filename || !image) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return ULTIMATE_ERROR_FILE_NOT_FOUND;
    }
    
    // Simple BMP loader for demonstration
    uint8_t header[54];
    if (fread(header, 1, 54, file) != 54) {
        fclose(file);
        return ULTIMATE_ERROR_INVALID_FORMAT;
    }
    
    // Check BMP signature
    if (header[0] != 'B' || header[1] != 'M') {
        fclose(file);
        return ULTIMATE_ERROR_INVALID_FORMAT;
    }
    
    uint32_t width = *(uint32_t*)&header[18];
    uint32_t height = *(uint32_t*)&header[22];
    uint16_t bpp = *(uint16_t*)&header[28];
    
    ultimate_image_format_t format;
    if (bpp == 24) {
        format = ULTIMATE_IMAGE_FORMAT_BGR24;
    } else if (bpp == 32) {
        format = ULTIMATE_IMAGE_FORMAT_BGRA32;
    } else {
        fclose(file);
        return ULTIMATE_ERROR_INVALID_FORMAT;
    }
    
    ultimate_error_t error = ultimate_image_create(image, width, height, format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        fclose(file);
        return error;
    }
    
    // Read pixel data
    fseek(file, *(uint32_t*)&header[10], SEEK_SET);
    size_t bytes_read = fread((*image)->data, 1, (*image)->data_size, file);
    
    fclose(file);
    
    if (bytes_read != (*image)->data_size) {
        ultimate_image_destroy(*image);
        *image = NULL;
        return ULTIMATE_ERROR_IO_ERROR;
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_save(const ultimate_image_t* image, const char* filename, 
                                    ultimate_image_type_t type, int quality) {
    if (!image || !filename) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    // Simple BMP saver for demonstration
    if (type == ULTIMATE_IMAGE_TYPE_BMP) {
        FILE* file = fopen(filename, "wb");
        if (!file) {
            return ULTIMATE_ERROR_IO_ERROR;
        }
        
        uint32_t file_size = 54 + image->data_size;
        uint8_t header[54] = {0};
        
        // BMP Header
        header[0] = 'B';
        header[1] = 'M';
        *(uint32_t*)&header[2] = file_size;
        *(uint32_t*)&header[10] = 54;
        *(uint32_t*)&header[14] = 40;
        *(uint32_t*)&header[18] = image->width;
        *(uint32_t*)&header[22] = image->height;
        *(uint16_t*)&header[26] = 1;
        *(uint16_t*)&header[28] = image->channels * image->depth;
        *(uint32_t*)&header[34] = image->data_size;
        
        fwrite(header, 1, 54, file);
        fwrite(image->data, 1, image->data_size, file);
        
        fclose(file);
        return ULTIMATE_ERROR_SUCCESS;
    }
    
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

/* Basic Image Operations */
ultimate_error_t ultimate_image_resize(const ultimate_image_t* src, ultimate_image_t** dst,
                                      uint32_t new_width, uint32_t new_height,
                                      ultimate_interpolation_t interpolation) {
    if (!src || !dst || new_width == 0 || new_height == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, new_width, new_height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    float x_ratio = (float)src->width / new_width;
    float y_ratio = (float)src->height / new_height;
    size_t pixel_size = get_pixel_size(src->format);
    
    for (uint32_t y = 0; y < new_height; y++) {
        for (uint32_t x = 0; x < new_width; x++) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;
            
            uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + (x * pixel_size);
            
            if (interpolation == ULTIMATE_INTERPOLATION_NEAREST) {
                uint32_t nearest_x = (uint32_t)(src_x + 0.5f);
                uint32_t nearest_y = (uint32_t)(src_y + 0.5f);
                
                nearest_x = CLAMP(nearest_x, 0, src->width - 1);
                nearest_y = CLAMP(nearest_y, 0, src->height - 1);
                
                uint8_t* src_pixel = src->data + (nearest_y * src->stride) + (nearest_x * pixel_size);
                memcpy(dst_pixel, src_pixel, pixel_size);
            } else {
                // Bilinear interpolation
                uint32_t x1 = (uint32_t)src_x;
                uint32_t y1 = (uint32_t)src_y;
                uint32_t x2 = MIN(x1 + 1, src->width - 1);
                uint32_t y2 = MIN(y1 + 1, src->height - 1);
                
                float dx = src_x - x1;
                float dy = src_y - y1;
                
                for (uint32_t c = 0; c < pixel_size; c++) {
                    float p11 = src->data[(y1 * src->stride) + (x1 * pixel_size) + c];
                    float p12 = src->data[(y1 * src->stride) + (x2 * pixel_size) + c];
                    float p21 = src->data[(y2 * src->stride) + (x1 * pixel_size) + c];
                    float p22 = src->data[(y2 * src->stride) + (x2 * pixel_size) + c];
                    
                    float interpolated = p11 * (1 - dx) * (1 - dy) +
                                       p12 * dx * (1 - dy) +
                                       p21 * (1 - dx) * dy +
                                       p22 * dx * dy;
                    
                    dst_pixel[c] = (uint8_t)CLAMP(interpolated, 0, 255);
                }
            }
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_crop(const ultimate_image_t* src, ultimate_image_t** dst,
                                    const ultimate_rect_t* rect) {
    if (!src || !dst || !rect) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (rect->x + rect->width > src->width || rect->y + rect->height > src->height) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, rect->width, rect->height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    size_t pixel_size = get_pixel_size(src->format);
    size_t row_bytes = rect->width * pixel_size;
    
    for (uint32_t y = 0; y < rect->height; y++) {
        uint8_t* src_row = src->data + ((rect->y + y) * src->stride) + (rect->x * pixel_size);
        uint8_t* dst_row = (*dst)->data + (y * (*dst)->stride);
        memcpy(dst_row, src_row, row_bytes);
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_rotate(const ultimate_image_t* src, ultimate_image_t** dst,
                                      float angle, ultimate_interpolation_t interpolation) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    float rad = angle * PI / 180.0f;
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);
    
    // Calculate new dimensions
    float corners_x[] = {0, src->width, src->width, 0};
    float corners_y[] = {0, 0, src->height, src->height};
    
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    
    for (int i = 0; i < 4; i++) {
        float new_x = corners_x[i] * cos_a - corners_y[i] * sin_a;
        float new_y = corners_x[i] * sin_a + corners_y[i] * cos_a;
        
        min_x = MIN(min_x, new_x);
        max_x = MAX(max_x, new_x);
        min_y = MIN(min_y, new_y);
        max_y = MAX(max_y, new_y);
    }
    
    uint32_t new_width = (uint32_t)(max_x - min_x + 0.5f);
    uint32_t new_height = (uint32_t)(max_y - min_y + 0.5f);
    
    ultimate_error_t error = ultimate_image_create(dst, new_width, new_height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    float center_x = src->width / 2.0f;
    float center_y = src->height / 2.0f;
    float new_center_x = new_width / 2.0f;
    float new_center_y = new_height / 2.0f;
    
    size_t pixel_size = get_pixel_size(src->format);
    
    for (uint32_t y = 0; y < new_height; y++) {
        for (uint32_t x = 0; x < new_width; x++) {
            // Reverse rotation to find source pixel
            float dx = x - new_center_x;
            float dy = y - new_center_y;
            
            float src_x = dx * cos_a + dy * sin_a + center_x;
            float src_y = -dx * sin_a + dy * cos_a + center_y;
            
            uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + (x * pixel_size);
            
            if (src_x >= 0 && src_x < src->width && src_y >= 0 && src_y < src->height) {
                if (interpolation == ULTIMATE_INTERPOLATION_NEAREST) {
                    uint32_t nearest_x = (uint32_t)(src_x + 0.5f);
                    uint32_t nearest_y = (uint32_t)(src_y + 0.5f);
                    
                    uint8_t* src_pixel = src->data + (nearest_y * src->stride) + (nearest_x * pixel_size);
                    memcpy(dst_pixel, src_pixel, pixel_size);
                } else {
                    // Bilinear interpolation (similar to resize)
                    uint32_t x1 = (uint32_t)src_x;
                    uint32_t y1 = (uint32_t)src_y;
                    uint32_t x2 = MIN(x1 + 1, src->width - 1);
                    uint32_t y2 = MIN(y1 + 1, src->height - 1);
                    
                    float dx = src_x - x1;
                    float dy = src_y - y1;
                    
                    for (uint32_t c = 0; c < pixel_size; c++) {
                        float p11 = src->data[(y1 * src->stride) + (x1 * pixel_size) + c];
                        float p12 = src->data[(y1 * src->stride) + (x2 * pixel_size) + c];
                        float p21 = src->data[(y2 * src->stride) + (x1 * pixel_size) + c];
                        float p22 = src->data[(y2 * src->stride) + (x2 * pixel_size) + c];
                        
                        float interpolated = p11 * (1 - dx) * (1 - dy) +
                                           p12 * dx * (1 - dy) +
                                           p21 * (1 - dx) * dy +
                                           p22 * dx * dy;
                        
                        dst_pixel[c] = (uint8_t)CLAMP(interpolated, 0, 255);
                    }
                }
            } else {
                // Fill with black for pixels outside source
                memset(dst_pixel, 0, pixel_size);
            }
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_flip_horizontal(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    size_t pixel_size = get_pixel_size(src->format);
    
    for (uint32_t y = 0; y < src->height; y++) {
        for (uint32_t x = 0; x < src->width; x++) {
            uint8_t* src_pixel = src->data + (y * src->stride) + (x * pixel_size);
            uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + ((src->width - 1 - x) * pixel_size);
            memcpy(dst_pixel, src_pixel, pixel_size);
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_flip_vertical(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    for (uint32_t y = 0; y < src->height; y++) {
        uint8_t* src_row = src->data + (y * src->stride);
        uint8_t* dst_row = (*dst)->data + ((src->height - 1 - y) * (*dst)->stride);
        memcpy(dst_row, src_row, src->stride);
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Color Space Conversions */
ultimate_error_t ultimate_image_rgb_to_gray(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (src->format != ULTIMATE_IMAGE_FORMAT_RGB24 && src->format != ULTIMATE_IMAGE_FORMAT_RGBA32) {
        return ULTIMATE_ERROR_INVALID_FORMAT;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, ULTIMATE_IMAGE_FORMAT_GRAY8);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    size_t src_pixel_size = get_pixel_size(src->format);
    
    for (uint32_t y = 0; y < src->height; y++) {
        for (uint32_t x = 0; x < src->width; x++) {
            uint8_t* src_pixel = src->data + (y * src->stride) + (x * src_pixel_size);
            uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + x;
            
            // Luminance formula: 0.299*R + 0.587*G + 0.114*B
            float gray = 0.299f * src_pixel[0] + 0.587f * src_pixel[1] + 0.114f * src_pixel[2];
            *dst_pixel = (uint8_t)CLAMP(gray, 0, 255);
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_rgb_to_hsv(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst || src->format != ULTIMATE_IMAGE_FORMAT_RGB24) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, ULTIMATE_IMAGE_FORMAT_HSV);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    for (uint32_t y = 0; y < src->height; y++) {
        for (uint32_t x = 0; x < src->width; x++) {
            uint8_t* src_pixel = src->data + (y * src->stride) + (x * 3);
            uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + (x * 3);
            
            float r = src_pixel[0] / 255.0f;
            float g = src_pixel[1] / 255.0f;
            float b = src_pixel[2] / 255.0f;
            
            float max_val = MAX(MAX(r, g), b);
            float min_val = MIN(MIN(r, g), b);
            float delta = max_val - min_val;
            
            float h = 0, s = 0, v = max_val;
            
            if (delta > 0) {
                s = delta / max_val;
                
                if (max_val == r) {
                    h = 60.0f * fmod((g - b) / delta, 6.0f);
                } else if (max_val == g) {
                    h = 60.0f * ((b - r) / delta + 2.0f);
                } else {
                    h = 60.0f * ((r - g) / delta + 4.0f);
                }
                
                if (h < 0) h += 360.0f;
            }
            
            dst_pixel[0] = (uint8_t)(h * 255.0f / 360.0f);
            dst_pixel[1] = (uint8_t)(s * 255.0f);
            dst_pixel[2] = (uint8_t)(v * 255.0f);
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Image Enhancement */
ultimate_error_t ultimate_image_adjust_brightness(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 float brightness) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_clone(src, dst);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    int32_t brightness_offset = (int32_t)(brightness * 255.0f);
    size_t pixel_size = get_pixel_size(src->format);
    
    for (size_t i = 0; i < (*dst)->data_size; i++) {
        int32_t new_value = (*dst)->data[i] + brightness_offset;
        (*dst)->data[i] = (uint8_t)CLAMP(new_value, 0, 255);
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_adjust_contrast(const ultimate_image_t* src, ultimate_image_t** dst,
                                               float contrast) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_clone(src, dst);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    float factor = (259.0f * (contrast * 255.0f + 255.0f)) / (255.0f * (259.0f - contrast * 255.0f));
    
    for (size_t i = 0; i < (*dst)->data_size; i++) {
        float new_value = factor * ((*dst)->data[i] - 128.0f) + 128.0f;
        (*dst)->data[i] = (uint8_t)CLAMP(new_value, 0, 255);
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_adjust_gamma(const ultimate_image_t* src, ultimate_image_t** dst,
                                            float gamma) {
    if (!src || !dst || gamma <= 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_error_t error = ultimate_image_clone(src, dst);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    // Build gamma correction lookup table
    uint8_t gamma_table[256];
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float corrected = powf(normalized, 1.0f / gamma);
        gamma_table[i] = (uint8_t)(corrected * 255.0f);
    }
    
    for (size_t i = 0; i < (*dst)->data_size; i++) {
        (*dst)->data[i] = gamma_table[(*dst)->data[i]];
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Filtering Operations */
ultimate_error_t ultimate_image_blur_gaussian(const ultimate_image_t* src, ultimate_image_t** dst,
                                             float sigma_x, float sigma_y) {
    if (!src || !dst || sigma_x <= 0 || sigma_y <= 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    // Create Gaussian kernel
    int32_t kernel_size_x = (int32_t)(6 * sigma_x + 1);
    int32_t kernel_size_y = (int32_t)(6 * sigma_y + 1);
    if (kernel_size_x % 2 == 0) kernel_size_x++;
    if (kernel_size_y % 2 == 0) kernel_size_y++;
    
    ultimate_filter_kernel_t* kernel;
    ultimate_error_t error = ultimate_filter_kernel_create_gaussian(&kernel, kernel_size_x, sigma_x);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    error = ultimate_image_apply_kernel(src, dst, kernel, ULTIMATE_EDGE_CLAMP);
    ultimate_filter_kernel_destroy(kernel);
    
    return error;
}

ultimate_error_t ultimate_image_apply_kernel(const ultimate_image_t* src, ultimate_image_t** dst,
                                            const ultimate_filter_kernel_t* kernel,
                                            ultimate_edge_mode_t edge_mode) {
    if (!src || !dst || !kernel) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_error_t error = ultimate_image_create(dst, src->width, src->height, src->format);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        return error;
    }
    
    size_t pixel_size = get_pixel_size(src->format);
    int32_t half_width = kernel->width / 2;
    int32_t half_height = kernel->height / 2;
    
    for (uint32_t y = 0; y < src->height; y++) {
        for (uint32_t x = 0; x < src->width; x++) {
            for (uint32_t c = 0; c < pixel_size; c++) {
                float sum = 0.0f;
                
                for (int32_t ky = -half_height; ky <= half_height; ky++) {
                    for (int32_t kx = -half_width; kx <= half_width; kx++) {
                        int32_t src_x = (int32_t)x + kx;
                        int32_t src_y = (int32_t)y + ky;
                        
                        // Handle edge cases
                        if (edge_mode == ULTIMATE_EDGE_CLAMP) {
                            src_x = CLAMP(src_x, 0, (int32_t)src->width - 1);
                            src_y = CLAMP(src_y, 0, (int32_t)src->height - 1);
                        } else if (edge_mode == ULTIMATE_EDGE_WRAP) {
                            src_x = src_x % (int32_t)src->width;
                            src_y = src_y % (int32_t)src->height;
                            if (src_x < 0) src_x += src->width;
                            if (src_y < 0) src_y += src->height;
                        } else if (src_x < 0 || src_x >= (int32_t)src->width || 
                                  src_y < 0 || src_y >= (int32_t)src->height) {
                            continue; // Skip out-of-bounds pixels
                        }
                        
                        uint8_t pixel_value = src->data[(src_y * src->stride) + (src_x * pixel_size) + c];
                        float kernel_value = kernel->data[(ky + half_height) * kernel->width + (kx + half_width)];
                        sum += pixel_value * kernel_value;
                    }
                }
                
                sum = sum / kernel->divisor + kernel->offset;
                uint8_t* dst_pixel = (*dst)->data + (y * (*dst)->stride) + (x * pixel_size) + c;
                *dst_pixel = (uint8_t)CLAMP(sum, 0, 255);
            }
        }
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Kernel Creation Helpers */
ultimate_error_t ultimate_filter_kernel_create_gaussian(ultimate_filter_kernel_t** kernel,
                                                       int32_t size, float sigma) {
    if (!kernel || size <= 0 || sigma <= 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (size % 2 == 0) size++; // Ensure odd size
    
    ultimate_filter_kernel_t* k = (ultimate_filter_kernel_t*)ultimate_malloc(sizeof(ultimate_filter_kernel_t));
    if (!k) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    k->width = size;
    k->height = size;
    k->data = (float*)ultimate_malloc(size * size * sizeof(float));
    if (!k->data) {
        ultimate_free(k);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    int32_t center = size / 2;
    float sum = 0.0f;
    float sigma_sq_2 = 2.0f * sigma * sigma;
    
    for (int32_t y = 0; y < size; y++) {
        for (int32_t x = 0; x < size; x++) {
            int32_t dx = x - center;
            int32_t dy = y - center;
            float distance_sq = dx * dx + dy * dy;
            float value = expf(-distance_sq / sigma_sq_2);
            
            k->data[y * size + x] = value;
            sum += value;
        }
    }
    
    k->divisor = sum;
    k->offset = 0.0f;
    
    *kernel = k;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_filter_kernel_create_sobel_x(ultimate_filter_kernel_t** kernel) {
    if (!kernel) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_filter_kernel_t* k = (ultimate_filter_kernel_t*)ultimate_malloc(sizeof(ultimate_filter_kernel_t));
    if (!k) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    k->width = 3;
    k->height = 3;
    k->data = (float*)ultimate_malloc(9 * sizeof(float));
    if (!k->data) {
        ultimate_free(k);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    float sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    memcpy(k->data, sobel_x, 9 * sizeof(float));
    k->divisor = 1.0f;
    k->offset = 128.0f; // Offset for signed values
    
    *kernel = k;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_filter_kernel_create_sobel_y(ultimate_filter_kernel_t** kernel) {
    if (!kernel) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    ultimate_filter_kernel_t* k = (ultimate_filter_kernel_t*)ultimate_malloc(sizeof(ultimate_filter_kernel_t));
    if (!k) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    k->width = 3;
    k->height = 3;
    k->data = (float*)ultimate_malloc(9 * sizeof(float));
    if (!k->data) {
        ultimate_free(k);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    float sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    memcpy(k->data, sobel_y, 9 * sizeof(float));
    k->divisor = 1.0f;
    k->offset = 128.0f;
    
    *kernel = k;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_filter_kernel_destroy(ultimate_filter_kernel_t* kernel) {
    if (!kernel) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (kernel->data) {
        ultimate_free(kernel->data);
    }
    ultimate_free(kernel);
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Edge Detection */
ultimate_error_t ultimate_image_edge_sobel(const ultimate_image_t* src, ultimate_image_t** dst) {
    if (!src || !dst) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale first if needed
    ultimate_image_t* gray_src = NULL;
    if (src->format != ULTIMATE_IMAGE_FORMAT_GRAY8) {
        ultimate_error_t error = ultimate_image_rgb_to_gray(src, &gray_src);
        if (error != ULTIMATE_ERROR_SUCCESS) {
            return error;
        }
        src = gray_src;
    }
    
    ultimate_filter_kernel_t* kernel_x;
    ultimate_filter_kernel_t* kernel_y;
    ultimate_image_t* grad_x = NULL;
    ultimate_image_t* grad_y = NULL;
    
    ultimate_error_t error = ultimate_filter_kernel_create_sobel_x(&kernel_x);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        goto cleanup;
    }
    
    error = ultimate_filter_kernel_create_sobel_y(&kernel_y);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        goto cleanup;
    }
    
    error = ultimate_image_apply_kernel(src, &grad_x, kernel_x, ULTIMATE_EDGE_CLAMP);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        goto cleanup;
    }
    
    error = ultimate_image_apply_kernel(src, &grad_y, kernel_y, ULTIMATE_EDGE_CLAMP);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        goto cleanup;
    }
    
    // Combine gradients
    error = ultimate_image_create(dst, src->width, src->height, ULTIMATE_IMAGE_FORMAT_GRAY8);
    if (error != ULTIMATE_ERROR_SUCCESS) {
        goto cleanup;
    }
    
    for (uint32_t y = 0; y < src->height; y++) {
        for (uint32_t x = 0; x < src->width; x++) {
            size_t idx = y * src->stride + x;
            float gx = grad_x->data[idx] - 128.0f; // Remove offset
            float gy = grad_y->data[idx] - 128.0f;
            float magnitude = sqrtf(gx * gx + gy * gy);
            (*dst)->data[idx] = (uint8_t)CLAMP(magnitude, 0, 255);
        }
    }
    
cleanup:
    if (gray_src) ultimate_image_destroy(gray_src);
    if (grad_x) ultimate_image_destroy(grad_x);
    if (grad_y) ultimate_image_destroy(grad_y);
    if (kernel_x) ultimate_filter_kernel_destroy(kernel_x);
    if (kernel_y) ultimate_filter_kernel_destroy(kernel_y);
    
    return error;
}

/* Utility Functions */
ultimate_error_t ultimate_image_get_pixel(const ultimate_image_t* image, uint32_t x, uint32_t y,
                                         uint8_t* pixel) {
    if (!image || !pixel || x >= image->width || y >= image->height) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t pixel_size = get_pixel_size(image->format);
    uint8_t* src_pixel = image->data + (y * image->stride) + (x * pixel_size);
    memcpy(pixel, src_pixel, pixel_size);
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_set_pixel(ultimate_image_t* image, uint32_t x, uint32_t y,
                                         const uint8_t* pixel) {
    if (!image || !pixel || x >= image->width || y >= image->height) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t pixel_size = get_pixel_size(image->format);
    uint8_t* dst_pixel = image->data + (y * image->stride) + (x * pixel_size);
    memcpy(dst_pixel, pixel, pixel_size);
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_get_info(const ultimate_image_t* image, uint32_t* width,
                                        uint32_t* height, ultimate_image_format_t* format) {
    if (!image) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (width) *width = image->width;
    if (height) *height = image->height;
    if (format) *format = image->format;
    
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_validate(const ultimate_image_t* image) {
    if (!image) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (image->width == 0 || image->height == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (!image->data) {
        return ULTIMATE_ERROR_NULL_POINTER;
    }
    
    if (image->channels == 0 || image->depth == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t expected_stride = image->width * get_pixel_size(image->format);
    if (image->stride < expected_stride) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    size_t expected_size = image->stride * image->height;
    if (image->data_size < expected_size) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Performance and Threading */
ultimate_error_t ultimate_image_set_thread_count(int32_t thread_count) {
    if (thread_count <= 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    g_thread_count = thread_count;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_enable_gpu_acceleration(bool enabled) {
    g_gpu_acceleration = enabled;
    return ULTIMATE_ERROR_SUCCESS;
}

ultimate_error_t ultimate_image_get_processing_stats(uint64_t* operations_count,
                                                    uint64_t* total_time_ms,
                                                    float* avg_fps) {
    if (operations_count) *operations_count = g_operations_count;
    if (total_time_ms) *total_time_ms = g_total_time_ms;
    if (avg_fps && g_total_time_ms > 0) {
        *avg_fps = (float)g_operations_count * 1000.0f / g_total_time_ms;
    }
    
    return ULTIMATE_ERROR_SUCCESS;
}

/* Placeholder implementations for advanced functions */
ultimate_error_t ultimate_image_super_resolution(const ultimate_image_t* src, ultimate_image_t** dst,
                                                float scale_factor, const char* method) {
    // For now, use simple bicubic upscaling
    uint32_t new_width = (uint32_t)(src->width * scale_factor);
    uint32_t new_height = (uint32_t)(src->height * scale_factor);
    return ultimate_image_resize(src, dst, new_width, new_height, ULTIMATE_INTERPOLATION_CUBIC);
}

ultimate_error_t ultimate_image_inpaint(const ultimate_image_t* src, const ultimate_image_t* mask,
                                       ultimate_image_t** dst, const char* method) {
    // Simple inpainting - just copy source for now
    return ultimate_image_clone(src, dst);
}

ultimate_error_t ultimate_image_denoise_gaussian(const ultimate_image_t* src, ultimate_image_t** dst,
                                                float sigma) {
    return ultimate_image_blur_gaussian(src, dst, sigma, sigma);
}

/* Additional placeholder implementations for completeness */
ultimate_error_t ultimate_image_load_from_memory(const uint8_t* data, size_t size, 
                                                ultimate_image_t** image) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_save_to_memory(const ultimate_image_t* image, 
                                              ultimate_image_type_t type, int quality,
                                              uint8_t** data, size_t* size) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_convert_format(const ultimate_image_t* src, ultimate_image_t** dst,
                                              ultimate_image_format_t target_format) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_hsv_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_rgb_to_lab(const ultimate_image_t* src, ultimate_image_t** dst) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_lab_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_rgb_to_yuv(const ultimate_image_t* src, ultimate_image_t** dst) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_yuv_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_adjust_saturation(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 float saturation) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_adjust_hue(const ultimate_image_t* src, ultimate_image_t** dst,
                                          float hue) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}

ultimate_error_t ultimate_image_apply_adjustments(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 const ultimate_image_adjustments_t* adjustments) {
    return ULTIMATE_ERROR_NOT_IMPLEMENTED;
}