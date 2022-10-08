#include "decode_jpeg.h"
#include "common_jpeg.h"

namespace vision {
namespace image {

#if !JPEG_FOUND
torch::Tensor decode_jpeg(const torch::Tensor& data, ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_jpeg: torchvision not compiled with libjpeg support");
}
#else

using namespace detail;

namespace {

struct torch_jpeg_mgr {
  struct jpeg_source_mgr pub;
  const JOCTET* data;
  size_t len;
};

static void torch_jpeg_init_source(j_decompress_ptr cinfo) {}

static boolean torch_jpeg_fill_input_buffer(j_decompress_ptr cinfo) {
  // No more data.  Probably an incomplete image;  Raise exception.
  torch_jpeg_error_ptr myerr = (torch_jpeg_error_ptr)cinfo->err;
  strcpy(myerr->jpegLastErrorMsg, "Image is incomplete or truncated");
  longjmp(myerr->setjmp_buffer, 1);
}

static void torch_jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
  torch_jpeg_mgr* src = (torch_jpeg_mgr*)cinfo->src;
  if (src->pub.bytes_in_buffer < (size_t)num_bytes) {
    // Skipping over all of remaining data;  output EOI.
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    // Skipping over only some of the remaining data.
    src->pub.next_input_byte += num_bytes;
    src->pub.bytes_in_buffer -= num_bytes;
  }
}

static void torch_jpeg_term_source(j_decompress_ptr cinfo) {}

static void torch_jpeg_set_source_mgr(
    j_decompress_ptr cinfo,
    const unsigned char* data,
    size_t len) {
  torch_jpeg_mgr* src;
  if (cinfo->src == 0) { // if this is first time;  allocate memory
    cinfo->src = (struct jpeg_source_mgr*)(*cinfo->mem->alloc_small)(
        (j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(torch_jpeg_mgr));
  }
  src = (torch_jpeg_mgr*)cinfo->src;
  src->pub.init_source = torch_jpeg_init_source;
  src->pub.fill_input_buffer = torch_jpeg_fill_input_buffer;
  src->pub.skip_input_data = torch_jpeg_skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart; // default
  src->pub.term_source = torch_jpeg_term_source;
  // fill the buffers
  src->data = (const JOCTET*)data;
  src->len = len;
  src->pub.bytes_in_buffer = len;
  src->pub.next_input_byte = src->data;
}

} // namespace

torch::Tensor decode_jpeg(const torch::Tensor& data, ImageReadMode mode) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cpu.decode_jpeg.decode_jpeg");
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  struct jpeg_decompress_struct cinfo;
  struct torch_jpeg_error_mgr jerr;

  auto datap = data.data_ptr<uint8_t>();
  // Setup decompression structure
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = torch_jpeg_error_exit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object.
     */
    jpeg_destroy_decompress(&cinfo);
    TORCH_CHECK(false, jerr.jpegLastErrorMsg);
  }

  jpeg_create_decompress(&cinfo);
  torch_jpeg_set_source_mgr(&cinfo, datap, data.numel());

  // read info from header.
  jpeg_read_header(&cinfo, TRUE);

  int channels = cinfo.num_components;

  if (mode != IMAGE_READ_MODE_UNCHANGED) {
    switch (mode) {
      case IMAGE_READ_MODE_GRAY:
        if (cinfo.jpeg_color_space != JCS_GRAYSCALE) {
          cinfo.out_color_space = JCS_GRAYSCALE;
          channels = 1;
        }
        break;
      case IMAGE_READ_MODE_RGB:
        if (cinfo.jpeg_color_space != JCS_RGB) {
          cinfo.out_color_space = JCS_RGB;
          channels = 3;
        }
        break;
      /*
       * Libjpeg does not support converting from CMYK to grayscale etc. There
       * is a way to do this but it involves converting it manually to RGB:
       * https://github.com/tensorflow/tensorflow/blob/86871065265b04e0db8ca360c046421efb2bdeb4/tensorflow/core/lib/jpeg/jpeg_mem.cc#L284-L313
       */
      default:
        jpeg_destroy_decompress(&cinfo);
        TORCH_CHECK(false, "The provided mode is not supported for JPEG files");
    }

    jpeg_calc_output_dimensions(&cinfo);
  }

  jpeg_start_decompress(&cinfo);

  int height = cinfo.output_height;
  int width = cinfo.output_width;

  int stride = width * channels;
  auto tensor =
      torch::empty({int64_t(height), int64_t(width), channels}, torch::kU8);
  auto ptr = tensor.data_ptr<uint8_t>();
  while (cinfo.output_scanline < cinfo.output_height) {
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    jpeg_read_scanlines(&cinfo, &ptr, 1);
    ptr += stride;
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return tensor.permute({2, 0, 1});
}

#endif

} // namespace image
} // namespace vision
