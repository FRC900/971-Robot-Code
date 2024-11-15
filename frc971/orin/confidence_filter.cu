#include <iostream>
#include "frc971/orin/confidence_filter.h"
#include "frc971/orin/cuCompactor.cuh"
#include "frc971/orin/cuda_utils.h"

template <class OUTPUT, class GRID_DIM, class PREDICATE>
ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::ConfidenceFilter(void)
{
    // Output size is calculated as cumulative size up to the last block plus the
    // number of outputs from the final block.
    cudaSafeCall(cudaMallocHost(&m_hOutputCountPtr, sizeof(*m_hOutputCountPtr)));
    cudaSafeCall(cudaEventCreate(&m_outputReadyEvent));
}

template <class OUTPUT, class GRID_DIM, class PREDICATE>
ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::~ConfidenceFilter(void)
{
    std::cout << "ConfidenceFilter::~ConfidenceFilter" << std::endl;
    cudaSafeCall(cudaFree(m_dOutput));
    cudaSafeCall(cudaFreeHost(m_hOutputCountPtr));
    cudaSafeCall(cudaFree(m_dBlocksCount));
    cudaSafeCall(cudaFree(m_dBlocksOffset));

    cudaSafeCall(cudaEventDestroy(m_outputReadyEvent));
}

// Set min confidence for a valid element
template <class OUTPUT, class GRID_DIM, class PREDICATE>
ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE> *ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::withConfidence(const float confidence)
{

    m_predicate.m_confidence = confidence;
    m_needNewCudaGraph = true;

    return this;
}
template <class OUTPUT, class GRID_DIM, class PREDICATE>
bool ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::reallocBuffers(const size_t count, cudaStream_t cudaStream)
{
    if (count > m_count)
    {
        cudaSafeCall(cudaFreeAsync(m_dOutput, cudaStream));
        cudaSafeCall(cudaFreeAsync(m_dBlocksCount, cudaStream));
        cudaSafeCall(cudaFreeAsync(m_dBlocksOffset, cudaStream));
        cudaSafeCall(cudaMallocAsync(&m_dOutput, sizeof(OUTPUT) * count, cudaStream));

        const uint32_t numBlocks = iDivUp(count, m_blockSize);
        cudaSafeCall(cudaMallocAsync(&m_dBlocksCount, sizeof(*m_dBlocksCount) * numBlocks, cudaStream));
        // Add 1 to hold the overall total valid count
        cudaSafeCall(cudaMallocAsync(&m_dBlocksOffset, sizeof(*m_dBlocksOffset) * (numBlocks + 1), cudaStream));

        m_count = count;
        return true;
    }
    return false;
}

template <class OUTPUT, class GRID_DIM, class PREDICATE>
void ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::detect(const std::array<const float *, 3> &inputs,
                                                           const GRID_DIM gridDims,
                                                           const float centerVariance,
                                                           const float sizeVariance,
                                                           cudaStream_t cudaStream,
                                                           const bool forceCudaGraphRegen)
{
    const bool buffersResized = reallocBuffers(gridDims.size(), cudaStream);
    if (buffersResized || forceCudaGraphRegen || m_needNewCudaGraph)
    {
        std::cout << "Keypoint detector : generating CUDA graph buffersResize = " <<  buffersResized << " forceCudaGraphRegen = " << forceCudaGraphRegen << std::endl;
        cudaSafeCall(cudaStreamSynchronize(cudaStream));
        cudaSafeCall(cudaStreamBeginCapture(cudaStream, cudaStreamCaptureModeGlobal));
        cuCompactor::compact(inputs,
                             m_dOutput,
                             m_hOutputCountPtr,
                             gridDims.data(),
                             centerVariance,
                             sizeVariance,
                             m_count,
                             m_predicate,
                             m_blockSize,
                             m_dBlocksCount,
                             m_dBlocksOffset,
                             cudaStream);
        cudaSafeCall(cudaStreamEndCapture(cudaStream, &m_cudaGraph));

        cudaSafeCall(cudaGraphInstantiate(&m_cudaGraphInstance, m_cudaGraph, NULL, NULL, 0));
        cudaSafeCall(cudaStreamSynchronize(cudaStream));
        m_needNewCudaGraph = false;
    }
    cudaSafeCall(cudaGraphLaunch(m_cudaGraphInstance, cudaStream));
    cudaSafeCall(cudaEventRecord(m_outputReadyEvent, cudaStream));
}

// Have to wait until a queued async D2H memcpy into m_hOutputCountPtr finishes
// before allowing host to get the number of output which passed filtering
template <class OUTPUT, class GRID_DIM, class PREDICATE>
tcb::span<const OUTPUT> ConfidenceFilter<OUTPUT, GRID_DIM, PREDICATE>::getOutput()
{
    cudaSafeCall(cudaEventSynchronize(m_outputReadyEvent));
    return tcb::span<const OUTPUT>(m_dOutput, m_hOutputCountPtr[0]);
}


// Predicate to filter on the max of two confidences
__device__ bool DecoderPredicate::operator()(const float *f, const int index, const int length) const
{
    return max(f[index], f[index + length]) > m_confidence;
}

#include "frc971/orin/grid_prior_value.h"
#include "frc971/orin/stage2_keypoint.h"
template class ConfidenceFilter<Stage2Keypoint, const tcb::span<const GridPriorValue> &, DecoderPredicate>;