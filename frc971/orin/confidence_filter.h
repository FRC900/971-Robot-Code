#ifndef CONFIDENCE_FILTER_INC__
#define CONFIDENCE_FILTER_INC__

#include "cuda_runtime.h"
#include "gpu_apriltag/span.hpp"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// This takes the raw outputs of inference, filters out values
// with confidence that is too low, and saves the remaining
// results in a more friendly structure (compared to flat arrays
// of floats coming out of the model)
// Part of the "saves the remaining results" is converting from
// model-relative normalized coordinates into input-image relative
// pixel coordinates.  This is done using the scale and shift values
// in the GRID_DIM object particular to each type of filtering.

template <class OUTPUT, class GRID_DIM, class PREDICATE>
class ConfidenceFilter
{
public:
    explicit ConfidenceFilter();

    ConfidenceFilter(const ConfidenceFilter &other) = delete;
    ConfidenceFilter(ConfidenceFilter &&other) noexcept = default;

    ConfidenceFilter &operator=(const ConfidenceFilter &other) = delete;
    ConfidenceFilter &operator=(ConfidenceFilter &&other) noexcept = delete;

    virtual ~ConfidenceFilter();

    ConfidenceFilter *withConfidence(const float confidence);

    // Takes array of input T's, where T has a .confidence field.
    // Returns a compacted array of Ts with the T's with confidence 
    // higher than the function arg confidence
    void detect(const std::array<const float *, 3> &inputs,
                GRID_DIM gridDims,
                const float centerVariance,
                const float scaleVariance,
                cudaStream_t cudaStream, 
                const bool forceCudaGraphRegen);

    tcb::span<const OUTPUT> getOutput();

private:
    bool reallocBuffers(const size_t length, cudaStream_t cudaStream);
    OUTPUT *m_dOutput{nullptr};
    cudaEvent_t m_outputReadyEvent;
    cudaGraph_t m_cudaGraph;
    cudaGraphExec_t m_cudaGraphInstance;
    uint32_t *m_hOutputCountPtr{nullptr};
    size_t m_count{0};
    size_t m_blockSize{1024};

    // Scratch space needed for cuCompactor. Reallocated only when needed because
    // of changes in image size (i.e. hopefully never for an input video stream)
    uint32_t *m_dBlocksCount{};
    uint32_t *m_dBlocksOffset{};

    // functor used to test confidence of predictions
    PREDICATE m_predicate{0};

    bool m_needNewCudaGraph{true};
};

// Predicate to filter on the best of 2 non-background scores
// TODO : does this need to have m_confidence copied from host to device when changed?
class DecoderPredicate
{
public:
    explicit DecoderPredicate(const float confidence)
        : m_confidence{confidence}
    {}
    __device__ bool operator()(const float *f, const int index, const int length) const;
    __device__ const char *getName() const { return "Stage2"; }
    float m_confidence{0.6f};
};

#endif