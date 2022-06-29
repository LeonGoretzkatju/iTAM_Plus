#ifndef DEPTH_SEGMENTATION_COMMON_H_
#define DEPTH_SEGMENTATION_COMMON_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <vector>
#include <set>

#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz/vizcore.hpp>


struct Segment {
  std::vector<cv::Vec3f> points;
  std::vector<cv::Vec3f> normals;
  std::vector<cv::Vec3f> original_colors;
  std::set<size_t> label;
  std::set<size_t> instance_label;
  std::set<size_t> semantic_label;
};

const static std::string kDebugWindowName = "DebugImages";
constexpr bool kUseTracker = true;

enum class SurfaceNormalEstimationMethod {
  kFals = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS,
  kLinemod = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,
  kSri = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI,
  kDepthWindowFilter = 3,
};

struct SurfaceNormalParams {
  SurfaceNormalParams() {
//    CHECK_EQ(window_size % 2u, 1u);
//    CHECK_GT(window_size, 1u);
    if (method != SurfaceNormalEstimationMethod::kDepthWindowFilter) {
      //CHECK_LT(window_size, 8u);
      //std::cout<<"s"<<std::endl;
    }
  }
  size_t window_size = 13u;
  SurfaceNormalEstimationMethod method =
      SurfaceNormalEstimationMethod::kDepthWindowFilter;
  bool display = false;
  double distance_factor_threshold = 0.05;
};

struct MaxDistanceMapParams {
  MaxDistanceMapParams() {
      //CHECK_EQ(window_size % 2u, 1u);
  }
  bool use_max_distance = true;
  size_t window_size = 1u;
  bool display = false;
  bool exclude_nan_as_max_distance = false;
  bool ignore_nan_coordinates = false;  // TODO(ff): This probably doesn't make
                                        // a lot of sense -> consider removing
                                        // it.
  bool use_threshold = true;
  double noise_thresholding_factor = 10.0;
  double sensor_noise_param_1st_order = 0.0012;  // From Nguyen et al. (2012)
  double sensor_noise_param_2nd_order = 0.0019;  // From Nguyen et al. (2012)
  double sensor_noise_param_3rd_order = 0.0001;  // From Nguyen et al. (2012)
  double sensor_min_distance = 0.02;
};

struct DepthDiscontinuityMapParams {
  DepthDiscontinuityMapParams() {
     // CHECK_EQ(kernel_size % 2u, 1u);
  }
  bool use_discontinuity = true;
  size_t kernel_size = 3u;
  double discontinuity_ratio = 0.01;
  bool display = false;
};

struct MinConvexityMapParams {
  MinConvexityMapParams() {
      //CHECK_EQ(window_size % 2u, 1u);
  }
  bool use_min_convexity = true;
  size_t morphological_opening_size = 1u;
  size_t window_size = 5u;
  size_t step_size = 1u;
  bool display = false;
  bool use_morphological_opening = true;
  bool use_threshold = true;
  double threshold = 0.97;
  double mask_threshold = -0.0005;
};

struct FinalEdgeMapParams {
  size_t morphological_opening_size = 1u;
  size_t morphological_closing_size = 1u;
  bool use_morphological_opening = true;
  bool use_morphological_closing = true;
  bool display = false;
};

enum class LabelMapMethod {
  kFloodFill = 0,
  kContour = 1,
};

struct LabelMapParams {
  LabelMapMethod method = LabelMapMethod::kContour;
  size_t min_size = 500u;
  bool use_inpaint = false;
  size_t inpaint_method = 0u;
  bool display = true;
};

struct SemanticInstanceSegmentationParams {
  bool enable = false;
  float overlap_threshold = 0.8f;
};

struct IsNan {
  template <class T>
  bool operator()(T const& p) const {
    return std::isnan(p);
  }
};

struct IsNotNan {
  template <class T>
  bool operator()(T const& p) const {
    return !std::isnan(p);
  }
};

struct Params {
  bool dilate_depth_image = false;
  size_t dilation_size = 1u;
  FinalEdgeMapParams final_edge;
  LabelMapParams label;
  DepthDiscontinuityMapParams depth_discontinuity;
  MaxDistanceMapParams max_distance;
  MinConvexityMapParams min_convexity;
  SurfaceNormalParams normals;
  SemanticInstanceSegmentationParams semantic_instance_segmentation;
  bool visualize_segmented_scene = true;
};


void computeCovariance(const cv::Mat& neighborhood, const cv::Vec3f& mean,
                       const size_t neighborhood_size, cv::Mat* covariance) ;

size_t findNeighborhood(const cv::Mat& depth_map, const size_t window_size,
                        const float max_distance, const size_t x,
                        const size_t y, cv::Mat* neighborhood,
                        cv::Vec3f* mean) ;

// \brief Compute point normals of a depth image.
//
// Compute the point normals by looking at a neighborhood around each pixel.
// We're taking a standard squared kernel, where we discard points that are too
// far away from the center point (by evaluating the Euclidean distance).
//
void cComputeOwnNormals(const SurfaceNormalParams& params,
                       const cv::Mat& depth_map, cv::Mat* normals);
  // namespace depth_segmentation

#endif  // DEPTH_SEGMENTATION_COMMON_H_
