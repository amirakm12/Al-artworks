#ifndef ULTIMATE_IMAGE_H
#define ULTIMATE_IMAGE_H

/**
 * @file ultimate_image.h
 * @brief ULTIMATE Image Processing System
 * @version 1.0.0
 * @date 2024
 * 
 * Comprehensive image processing functionality including:
 * - Basic image operations (load, save, resize, crop)
 * - Color space conversions
 * - Filtering and enhancement
 * - Geometric transformations
 * - Advanced processing (denoising, super-resolution, etc.)
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Image format definitions */
typedef enum {
    ULTIMATE_IMAGE_FORMAT_UNKNOWN = 0,
    ULTIMATE_IMAGE_FORMAT_RGB24,
    ULTIMATE_IMAGE_FORMAT_RGBA32,
    ULTIMATE_IMAGE_FORMAT_BGR24,
    ULTIMATE_IMAGE_FORMAT_BGRA32,
    ULTIMATE_IMAGE_FORMAT_GRAY8,
    ULTIMATE_IMAGE_FORMAT_GRAY16,
    ULTIMATE_IMAGE_FORMAT_RGB48,
    ULTIMATE_IMAGE_FORMAT_RGBA64,
    ULTIMATE_IMAGE_FORMAT_YUV420,
    ULTIMATE_IMAGE_FORMAT_YUV422,
    ULTIMATE_IMAGE_FORMAT_YUV444,
    ULTIMATE_IMAGE_FORMAT_HSV,
    ULTIMATE_IMAGE_FORMAT_LAB,
    ULTIMATE_IMAGE_FORMAT_XYZ
} ultimate_image_format_t;

typedef enum {
    ULTIMATE_IMAGE_TYPE_JPEG = 0,
    ULTIMATE_IMAGE_TYPE_PNG,
    ULTIMATE_IMAGE_TYPE_BMP,
    ULTIMATE_IMAGE_TYPE_TIFF,
    ULTIMATE_IMAGE_TYPE_GIF,
    ULTIMATE_IMAGE_TYPE_WEBP,
    ULTIMATE_IMAGE_TYPE_RAW,
    ULTIMATE_IMAGE_TYPE_HDR,
    ULTIMATE_IMAGE_TYPE_EXR
} ultimate_image_type_t;

/* Image structure */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t depth;                    // Bits per channel
    ultimate_image_format_t format;
    size_t stride;                     // Bytes per row
    size_t data_size;
    uint8_t* data;
    void* metadata;                    // EXIF, color profiles, etc.
} ultimate_image_t;

/* Image processing parameters */
typedef struct {
    float brightness;
    float contrast;
    float saturation;
    float hue;
    float gamma;
    float exposure;
    float highlights;
    float shadows;
    float whites;
    float blacks;
    float clarity;
    float vibrance;
} ultimate_image_adjustments_t;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} ultimate_rect_t;

typedef struct {
    float x;
    float y;
} ultimate_point_t;

/* Filter kernels */
typedef struct {
    int32_t width;
    int32_t height;
    float* data;
    float divisor;
    float offset;
} ultimate_filter_kernel_t;

/* Interpolation methods */
typedef enum {
    ULTIMATE_INTERPOLATION_NEAREST = 0,
    ULTIMATE_INTERPOLATION_LINEAR,
    ULTIMATE_INTERPOLATION_CUBIC,
    ULTIMATE_INTERPOLATION_LANCZOS,
    ULTIMATE_INTERPOLATION_MITCHELL,
    ULTIMATE_INTERPOLATION_CATMULL_ROM
} ultimate_interpolation_t;

/* Edge handling modes */
typedef enum {
    ULTIMATE_EDGE_CLAMP = 0,
    ULTIMATE_EDGE_WRAP,
    ULTIMATE_EDGE_MIRROR,
    ULTIMATE_EDGE_CONSTANT
} ultimate_edge_mode_t;

/* Noise types */
typedef enum {
    ULTIMATE_NOISE_GAUSSIAN = 0,
    ULTIMATE_NOISE_UNIFORM,
    ULTIMATE_NOISE_SALT_PEPPER,
    ULTIMATE_NOISE_POISSON,
    ULTIMATE_NOISE_SPECKLE
} ultimate_noise_type_t;

/* Core Image Operations */

/* Image creation and destruction */
ultimate_error_t ultimate_image_create(ultimate_image_t** image, 
                                      uint32_t width, uint32_t height, 
                                      ultimate_image_format_t format);
ultimate_error_t ultimate_image_destroy(ultimate_image_t* image);
ultimate_error_t ultimate_image_clone(const ultimate_image_t* src, ultimate_image_t** dst);

/* Image I/O */
ultimate_error_t ultimate_image_load(const char* filename, ultimate_image_t** image);
ultimate_error_t ultimate_image_save(const ultimate_image_t* image, const char* filename, 
                                    ultimate_image_type_t type, int quality);
ultimate_error_t ultimate_image_load_from_memory(const uint8_t* data, size_t size, 
                                                ultimate_image_t** image);
ultimate_error_t ultimate_image_save_to_memory(const ultimate_image_t* image, 
                                              ultimate_image_type_t type, int quality,
                                              uint8_t** data, size_t* size);

/* Basic Image Operations */
ultimate_error_t ultimate_image_resize(const ultimate_image_t* src, ultimate_image_t** dst,
                                      uint32_t new_width, uint32_t new_height,
                                      ultimate_interpolation_t interpolation);
ultimate_error_t ultimate_image_crop(const ultimate_image_t* src, ultimate_image_t** dst,
                                    const ultimate_rect_t* rect);
ultimate_error_t ultimate_image_rotate(const ultimate_image_t* src, ultimate_image_t** dst,
                                      float angle, ultimate_interpolation_t interpolation);
ultimate_error_t ultimate_image_flip_horizontal(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_flip_vertical(const ultimate_image_t* src, ultimate_image_t** dst);

/* Color Space Conversions */
ultimate_error_t ultimate_image_convert_format(const ultimate_image_t* src, ultimate_image_t** dst,
                                              ultimate_image_format_t target_format);
ultimate_error_t ultimate_image_rgb_to_gray(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_rgb_to_hsv(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_hsv_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_rgb_to_lab(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_lab_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_rgb_to_yuv(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_yuv_to_rgb(const ultimate_image_t* src, ultimate_image_t** dst);

/* Image Enhancement and Adjustment */
ultimate_error_t ultimate_image_adjust_brightness(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 float brightness);
ultimate_error_t ultimate_image_adjust_contrast(const ultimate_image_t* src, ultimate_image_t** dst,
                                               float contrast);
ultimate_error_t ultimate_image_adjust_gamma(const ultimate_image_t* src, ultimate_image_t** dst,
                                            float gamma);
ultimate_error_t ultimate_image_adjust_saturation(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 float saturation);
ultimate_error_t ultimate_image_adjust_hue(const ultimate_image_t* src, ultimate_image_t** dst,
                                          float hue);
ultimate_error_t ultimate_image_apply_adjustments(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 const ultimate_image_adjustments_t* adjustments);

/* Filtering Operations */
ultimate_error_t ultimate_image_apply_kernel(const ultimate_image_t* src, ultimate_image_t** dst,
                                            const ultimate_filter_kernel_t* kernel,
                                            ultimate_edge_mode_t edge_mode);
ultimate_error_t ultimate_image_blur_gaussian(const ultimate_image_t* src, ultimate_image_t** dst,
                                             float sigma_x, float sigma_y);
ultimate_error_t ultimate_image_blur_box(const ultimate_image_t* src, ultimate_image_t** dst,
                                        int32_t kernel_size);
ultimate_error_t ultimate_image_blur_motion(const ultimate_image_t* src, ultimate_image_t** dst,
                                           float angle, float distance);
ultimate_error_t ultimate_image_sharpen(const ultimate_image_t* src, ultimate_image_t** dst,
                                       float strength);
ultimate_error_t ultimate_image_unsharp_mask(const ultimate_image_t* src, ultimate_image_t** dst,
                                            float radius, float strength, float threshold);

/* Edge Detection */
ultimate_error_t ultimate_image_edge_sobel(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_edge_prewitt(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_edge_roberts(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_edge_canny(const ultimate_image_t* src, ultimate_image_t** dst,
                                          float low_threshold, float high_threshold);
ultimate_error_t ultimate_image_edge_laplacian(const ultimate_image_t* src, ultimate_image_t** dst);

/* Morphological Operations */
ultimate_error_t ultimate_image_erode(const ultimate_image_t* src, ultimate_image_t** dst,
                                     const ultimate_filter_kernel_t* kernel);
ultimate_error_t ultimate_image_dilate(const ultimate_image_t* src, ultimate_image_t** dst,
                                      const ultimate_filter_kernel_t* kernel);
ultimate_error_t ultimate_image_opening(const ultimate_image_t* src, ultimate_image_t** dst,
                                       const ultimate_filter_kernel_t* kernel);
ultimate_error_t ultimate_image_closing(const ultimate_image_t* src, ultimate_image_t** dst,
                                       const ultimate_filter_kernel_t* kernel);

/* Noise Operations */
ultimate_error_t ultimate_image_add_noise(const ultimate_image_t* src, ultimate_image_t** dst,
                                         ultimate_noise_type_t noise_type, float intensity);
ultimate_error_t ultimate_image_denoise_gaussian(const ultimate_image_t* src, ultimate_image_t** dst,
                                                float sigma);
ultimate_error_t ultimate_image_denoise_bilateral(const ultimate_image_t* src, ultimate_image_t** dst,
                                                 float spatial_sigma, float color_sigma);
ultimate_error_t ultimate_image_denoise_nlmeans(const ultimate_image_t* src, ultimate_image_t** dst,
                                               float h, int32_t template_size, int32_t search_size);
ultimate_error_t ultimate_image_denoise_wiener(const ultimate_image_t* src, ultimate_image_t** dst,
                                              float noise_variance);

/* Advanced Processing */
ultimate_error_t ultimate_image_super_resolution(const ultimate_image_t* src, ultimate_image_t** dst,
                                                float scale_factor, const char* method);
ultimate_error_t ultimate_image_inpaint(const ultimate_image_t* src, const ultimate_image_t* mask,
                                       ultimate_image_t** dst, const char* method);
ultimate_error_t ultimate_image_segment_watershed(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_segment_kmeans(const ultimate_image_t* src, ultimate_image_t** dst,
                                              int32_t k, int32_t max_iterations);
ultimate_error_t ultimate_image_segment_mean_shift(const ultimate_image_t* src, ultimate_image_t** dst,
                                                  float spatial_radius, float color_radius);

/* Feature Detection */
ultimate_error_t ultimate_image_detect_corners_harris(const ultimate_image_t* src,
                                                     ultimate_point_t** corners, size_t* count,
                                                     float threshold, float k);
ultimate_error_t ultimate_image_detect_corners_shi_tomasi(const ultimate_image_t* src,
                                                         ultimate_point_t** corners, size_t* count,
                                                         float threshold, int32_t max_corners);
ultimate_error_t ultimate_image_detect_blobs(const ultimate_image_t* src,
                                            ultimate_point_t** blobs, float** sizes, size_t* count);

/* Histogram Operations */
ultimate_error_t ultimate_image_histogram_calculate(const ultimate_image_t* src,
                                                   uint32_t** histogram, int32_t bins);
ultimate_error_t ultimate_image_histogram_equalize(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_histogram_match(const ultimate_image_t* src, 
                                               const ultimate_image_t* reference,
                                               ultimate_image_t** dst);

/* Geometric Transformations */
ultimate_error_t ultimate_image_transform_affine(const ultimate_image_t* src, ultimate_image_t** dst,
                                               const float* transform_matrix,
                                               ultimate_interpolation_t interpolation);
ultimate_error_t ultimate_image_transform_perspective(const ultimate_image_t* src, ultimate_image_t** dst,
                                                    const float* transform_matrix,
                                                    ultimate_interpolation_t interpolation);
ultimate_error_t ultimate_image_warp_polar(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_warp_cylindrical(const ultimate_image_t* src, ultimate_image_t** dst);
ultimate_error_t ultimate_image_warp_spherical(const ultimate_image_t* src, ultimate_image_t** dst);

/* Blending and Compositing */
ultimate_error_t ultimate_image_blend_alpha(const ultimate_image_t* src1, const ultimate_image_t* src2,
                                           ultimate_image_t** dst, float alpha);
ultimate_error_t ultimate_image_blend_add(const ultimate_image_t* src1, const ultimate_image_t* src2,
                                         ultimate_image_t** dst);
ultimate_error_t ultimate_image_blend_multiply(const ultimate_image_t* src1, const ultimate_image_t* src2,
                                              ultimate_image_t** dst);
ultimate_error_t ultimate_image_blend_screen(const ultimate_image_t* src1, const ultimate_image_t* src2,
                                            ultimate_image_t** dst);
ultimate_error_t ultimate_image_blend_overlay(const ultimate_image_t* src1, const ultimate_image_t* src2,
                                             ultimate_image_t** dst);

/* Utility Functions */
ultimate_error_t ultimate_image_get_pixel(const ultimate_image_t* image, uint32_t x, uint32_t y,
                                         uint8_t* pixel);
ultimate_error_t ultimate_image_set_pixel(ultimate_image_t* image, uint32_t x, uint32_t y,
                                         const uint8_t* pixel);
ultimate_error_t ultimate_image_get_info(const ultimate_image_t* image, uint32_t* width,
                                        uint32_t* height, ultimate_image_format_t* format);
ultimate_error_t ultimate_image_validate(const ultimate_image_t* image);

/* Kernel Creation Helpers */
ultimate_error_t ultimate_filter_kernel_create_gaussian(ultimate_filter_kernel_t** kernel,
                                                       int32_t size, float sigma);
ultimate_error_t ultimate_filter_kernel_create_sobel_x(ultimate_filter_kernel_t** kernel);
ultimate_error_t ultimate_filter_kernel_create_sobel_y(ultimate_filter_kernel_t** kernel);
ultimate_error_t ultimate_filter_kernel_create_laplacian(ultimate_filter_kernel_t** kernel);
ultimate_error_t ultimate_filter_kernel_create_sharpen(ultimate_filter_kernel_t** kernel);
ultimate_error_t ultimate_filter_kernel_destroy(ultimate_filter_kernel_t* kernel);

/* Performance and Threading */
ultimate_error_t ultimate_image_set_thread_count(int32_t thread_count);
ultimate_error_t ultimate_image_enable_gpu_acceleration(bool enabled);
ultimate_error_t ultimate_image_get_processing_stats(uint64_t* operations_count,
                                                    uint64_t* total_time_ms,
                                                    float* avg_fps);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_IMAGE_H */