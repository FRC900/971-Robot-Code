#ifndef SUPPRESS_AND_AVERAGE_KEYPOINTS_H__
#define SUPPRESS_AND_AVERAGE_KEYPOINTS_H__

#include <vector>
#include "gpu_apriltag/span.hpp"

template <class INPUT, class OUTPUT>
class SuppressAndAverageKeypoints
{
public:
    SuppressAndAverageKeypoints();

    SuppressAndAverageKeypoints(const SuppressAndAverageKeypoints &other) = delete;
    SuppressAndAverageKeypoints(SuppressAndAverageKeypoints &&other) noexcept = delete;

    SuppressAndAverageKeypoints &operator=(const SuppressAndAverageKeypoints &other) = delete;
    SuppressAndAverageKeypoints &operator=(SuppressAndAverageKeypoints &&other) noexcept = delete;

    virtual ~SuppressAndAverageKeypoints();
    void compute(const tcb::span<const INPUT> &input,
                 const float sigma,
                 const float min_cos,
                 cudaStream_t cudaStream);
    const tcb::span<const OUTPUT> getOutput();

private:
    INPUT    *m_hInput{nullptr};
    std::vector<OUTPUT> m_output;
    bool     *m_dGroupMatrix{nullptr};
    bool     *m_hGroupMatrix{nullptr};
    uint32_t  m_allocatedInputSize{0};
    uint32_t  m_thisInputSize{0};

    cudaEvent_t m_hostInputReadyEvent;
    cudaEvent_t m_outputReadyEvent;
    cudaStream_t m_hostMemcpyStream;
};

#endif
