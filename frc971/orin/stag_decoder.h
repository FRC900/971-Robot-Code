#ifndef STAG_DECODER_INC__
#define STAG_DECODER_INC__

#include <stddef.h>  // for size_t

#include <array>   // for array
#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

#include <opencv2/core/mat.hpp>    // for Mat
#include <opencv2/core/types.hpp>  // for Point2d

#include "frc971/orin/camera_matrix.h"
#include "frc971/orin/confidence_filter.h"      // for ConfidenceFilter
#include "frc971/orin/decoded_tag.h"            // for DecodedTag
#include "frc971/orin/decoder_engine.h"         // for DecoderEngine
#include "frc971/orin/decoder_softmax.h"        // for DecoderSoftmax
#include "frc971/orin/grid_prior.h"             // for GridPrior
#include "frc971/orin/stage2_corners.h"         // for Stage2Corners
#include "frc971/orin/stage2_keypoint_trust.h"  // for Stage2KeypointTrust
#include "frc971/orin/suppress_and_average_keypoints.h"  // for SuppressAndAverageKeypoints
#include "gpu_apriltag/span.hpp"                         // for span

template <typename T> struct GpuImage;
struct GridPriorValue;
class Stage2Keypoint;
class Stage2KeypointGroup;
class Timings;
template <size_t GRID_SIZE> struct PointsAndIDs;

template <class MARKER_DICT, size_t GRID_SIZE>
class STagDecoder
{
public:
 STagDecoder(const MARKER_DICT &markerDict,
             const frc971::apriltag::CameraMatrix &cameraMatrix,
             const frc971::apriltag::DistCoeffs &distCoeffs,
             Timings &timing);
 STagDecoder(const STagDecoder &other) = delete;
 STagDecoder(STagDecoder &&other) noexcept = delete;

 STagDecoder &operator=(const STagDecoder &other) = delete;
 STagDecoder &operator=(STagDecoder &&other) noexcept = delete;

 void initEngine(const std::string &modelPath,
                 const std::string &onnxModelFilename);
 
 // Decode tags
 // grayImageDevice - the full input image, converted to grayscale and copied to device
 // rois - vector of corners of tags. Tag corners are in clockwise order from tr->tl->bl->br
 // Returns 2-entry array per tag. Each array entry is 1 pass of the decode (the initial
 // followed by the one after refining corners)
 std::vector<std::array<DecodedTag<GRID_SIZE>, 2>> detectTags(
     const GpuImage<uint8_t> &grayImageDevice,
     const std::vector<std::array<cv::Point2d, 4>> &rois);

 virtual ~STagDecoder() = default;

 void setMinGridMatchRatio(const double minGridMatchRatio);
 double getMinGridMatchRatio(void) const;
 ushort2 getModelSize(void) const;
 cudaStream_t getCudaStream(void);

private:
    void runInference(std::vector<std::vector<Stage2KeypointGroup>> &stage2KeypointGroupss,
                      std::vector<std::array<float2, 4>> &stage2Corners,
                      const GpuImage<uint8_t> &detectInputs,
                      const tcb::span<const std::array<cv::Point2d, 4>> &rois);
    void fillEmptyIds(PointsAndIDs<GRID_SIZE + 2> &orderedFineGridPointsIds,
                      const tcb::span<const Stage2KeypointGroup> &fineGridPointsWithIdsCandidates) const;

    const MARKER_DICT &m_markerDict;
    cv::Mat m_cameraMatrix;
    cv::Mat m_distCoeffs;
    Timings &m_timing;
    std::unique_ptr<DecoderEngine> m_decodeEngine;
    GridPrior<0, true, true> m_stage2CornerPrior;
    GridPrior<0, true, true> m_stage2GridPrior;
    DecoderSoftmax m_stage2DecoderSoftmax;

    // Buffer formatted in the shape expected by the base engine runInference call
    // Each vector size will be 1 since there's just one input to runInference
    // call, and the code takes care of extracting potentially multiple tag 
    // detection regions from this image and stuffs them into the model input.
    std::vector<std::vector<GpuImage<uint8_t>>> m_engineInputs;

    ConfidenceFilter<Stage2Keypoint, const tcb::span<const GridPriorValue> &, Stage2Predicate> m_confidenceFilter{1024};
    Stage2KeypointTrust m_keypointTrust;
    SuppressAndAverageKeypoints<Stage2Keypoint, Stage2KeypointGroup> m_keypointGrouper;

    Stage2Corners m_corners;

    // TODO - not sure how configurable this needs to be
    static constexpr size_t m_maxBatchSize = 4;
    double m_minGridMatchRatio = 0.4;
};

#include "frc971/orin/marker_dict.h"
template <size_t GRID_SIZE>
using ArucoSTagDecoder = STagDecoder<ArucoMarkerDict<GRID_SIZE>, GRID_SIZE>;
#endif
