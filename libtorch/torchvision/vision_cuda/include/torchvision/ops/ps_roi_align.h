#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API std::tuple<at::Tensor, at::Tensor> ps_roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);

VISION_API std::tuple<at::Tensor, at::Tensor> ps_roi_align_symint(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio);

namespace detail {

at::Tensor _ps_roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

at::Tensor _ps_roi_align_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    c10::SymInt batch_size,
    c10::SymInt channels,
    c10::SymInt height,
    c10::SymInt width);

} // namespace detail

} // namespace ops
} // namespace vision
