// Merge similar sets of INPUTs into a single output, where the 
// output's values are the confidence-weighted combination of
// the various input fields
//
// Kind of like non-max suppression, but also better(?) since it
// merges detections into an averaged midpoint rather than just
// rejecting the non-max inference results

#include <cstdio>
#include <iostream>
#include <vector>
#include "frc971/orin/cuda_utils.h"
#include "frc971/orin/suppress_and_average_keypoints.h"

template <class INPUT>
__global__ void computeGroupMembership(bool *groupMatrix,
                                       const INPUT *input,
                                       const uint32_t inputCount,
                                       const float sigma,
                                       const float min_cos)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= inputCount) || (y >= inputCount))
    {
        return;
    }
    // Only need one diagonal of the matrix calculated
    // It is symmetrical, but also the other diagonal isn't ever referenced
    if (y < x)
    {
        return;
    }

    bool result = true;
    //printf ("x = %d, y = %d, index = %d\n ", x, y, x * inputCount + y);
    if (x != y)
    {
        result = input[x].check(input[y], x, y, sigma, min_cos);
    }

    //printf ("x = %d, y = %d, index = %d, result = %d, sigma = %f, min_cos = %f\n", x, y, x * inputCount + y, result, sigma, min_cos);
    groupMatrix[x * inputCount + y] = result;
}

template <class INPUT, class OUTPUT>
SuppressAndAverageKeypoints<INPUT, OUTPUT>::SuppressAndAverageKeypoints()
{
    cudaSafeCall(cudaEventCreate(&m_hostInputReadyEvent));
    cudaSafeCall(cudaEventCreate(&m_outputReadyEvent));
    cudaSafeCall(cudaStreamCreate(&m_hostMemcpyStream));
}

template <class INPUT, class OUTPUT>
SuppressAndAverageKeypoints<INPUT, OUTPUT>::~SuppressAndAverageKeypoints()
{
    cudaSafeCall(cudaFreeHost(m_hInput));
    cudaSafeCall(cudaFree(m_dGroupMatrix));
    cudaSafeCall(cudaFreeHost(m_hGroupMatrix));
    cudaSafeCall(cudaEventDestroy(m_hostInputReadyEvent));
    cudaSafeCall(cudaEventDestroy(m_outputReadyEvent));
    cudaSafeCall(cudaStreamDestroy(m_hostMemcpyStream));
}

template <class INPUT, class OUTPUT>
void SuppressAndAverageKeypoints<INPUT, OUTPUT>::compute(const tcb::span<const INPUT> &input,
                                                         const float sigma,
                                                         const float min_cos,
                                                         cudaStream_t cudaStream)
{
    constexpr int32_t blockSize = 8;

    // Reallocate buffers if needed to fit the new input size
    if (input.size() > m_allocatedInputSize)
    {
        cudaSafeCall(cudaFreeHost(m_hInput));
        cudaSafeCall(cudaMallocHost(&m_hInput, sizeof(INPUT) * input.size()));

        // Allocate an nxn grid to mark same/not-same group from input i to input j.
        cudaSafeCall(cudaFreeAsync(m_dGroupMatrix, cudaStream));
        cudaSafeCall(cudaMallocAsync(&m_dGroupMatrix, sizeof(bool) * input.size() * input.size(), cudaStream));

        // And the corresponding buffer on the host side
        cudaSafeCall(cudaFreeHost(m_hGroupMatrix));
        cudaSafeCall(cudaMallocHost(&m_hGroupMatrix, sizeof(bool) * input.size() * input.size()));
        m_allocatedInputSize = input.size();

    }
    m_thisInputSize = input.size();
    if (input.size() > 0)
    {
        // Copy input to host for later use in group merging
        cudaSafeCall(cudaMemcpyAsync(m_hInput, input.data(), sizeof(INPUT) * input.size(), cudaMemcpyDeviceToHost, m_hostMemcpyStream));
        cudaSafeCall(cudaEventRecord(m_hostInputReadyEvent, m_hostMemcpyStream));

        // Compute group membership on GPU - this checks each pair of inputs to see
        // if they are similar enough to be in the same group
        const dim3 blockDim(blockSize, blockSize);
        const dim3 gridDim(iDivUp(input.size(), blockDim.x), iDivUp(input.size(), blockDim.y));
        computeGroupMembership<INPUT><<<gridDim, blockDim, 0, cudaStream>>>(m_dGroupMatrix, input.data(), input.size(), sigma, min_cos);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaMemcpyAsync(m_hGroupMatrix, m_dGroupMatrix, sizeof(bool) * input.size() * input.size(), cudaMemcpyDeviceToHost, cudaStream));
    }
    else
    {
        cudaSafeCall(cudaEventRecord(m_hostInputReadyEvent, m_hostMemcpyStream));
    }
    cudaSafeCall(cudaEventRecord(m_outputReadyEvent, cudaStream));
}

template <class INPUT, class OUTPUT>
const tcb::span<const OUTPUT> SuppressAndAverageKeypoints<INPUT, OUTPUT>::getOutput()
{
    if (m_thisInputSize == 0)
    {
        return tcb::span<const OUTPUT>();
    }

    std::vector<bool> used(m_thisInputSize, false);
    m_output.clear();

    cudaSafeCall(cudaEventSynchronize(m_hostInputReadyEvent));
    cudaSafeCall(cudaEventSynchronize(m_outputReadyEvent));
    // Do group merging here on CPU

    // Keep track of which indexes have already been added
    // to a group. The next one not added is the index of
    // a start of a new group.
    for (int32_t i = 0; i < m_thisInputSize; i++)
    {
        // Starting with the next first unused index
        // group together all of the indexes that are 
        // in the same group as that first index
        if (!used[i])
        {
            m_output.emplace_back();
            for (int32_t j = i; j < m_thisInputSize; j++)
            {
                if (m_hGroupMatrix[i * m_thisInputSize + j])
                {
                    m_output.back().append(m_hInput[j]);
                    used[j] = true;
                }
            }
            m_output.back().end();
        }
    }

    return tcb::span<const OUTPUT>(m_output);
}

// #include "deeptag_ros/stage1_grid.h"
// #include "deeptag_ros/ssd_tag_keypoint.h"
// #include "deeptag_ros/stage1_grid_group.h"
// #include "deeptag_ros/stage1_ssd_group.h"
#include "frc971/orin/stage2_keypoint.h"
#include "frc971/orin/stage2_keypoint_group.h"
// template class SuppressAndAverageKeypoints<Stage1Grid<1>, Stage1GridGroup<1>>;
// template class SuppressAndAverageKeypoints<Stage1Grid<4>, Stage1GridGroup<4>>;
// template class SuppressAndAverageKeypoints<Stage1Grid<5>, Stage1GridGroup<5>>;
// template class SuppressAndAverageKeypoints<Stage1Grid<9>, Stage1GridGroup<9>>;
// template class SuppressAndAverageKeypoints<Stage1Grid<10>, Stage1GridGroup<10>>;
// template class SuppressAndAverageKeypoints<SSDTagKeypoint, Stage1SSDGroup>;
template class SuppressAndAverageKeypoints<Stage2Keypoint, Stage2KeypointGroup>;
