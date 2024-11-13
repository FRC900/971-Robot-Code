#ifndef DECODER_PREPROCESS_H__
#define DECODER_PREPROCESS_H__
#include <array>
#include "cuda_runtime.h"  // for cudaError

#include "frc971/orin/image_format.h"

class DecoderPreprocess
{
    public:
    DecoderPreprocess(void);

    DecoderPreprocess(const DecoderPreprocess &other) = default;
    DecoderPreprocess(DecoderPreprocess &&other) noexcept = default;

    DecoderPreprocess &operator=(const DecoderPreprocess &other) = delete;
    DecoderPreprocess &operator=(DecoderPreprocess &&other) noexcept = default;

    virtual ~DecoderPreprocess();
    cudaError_t decoderPreprocessRGB(const float *hH, void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
                                     float *output, size_t outputWidth, size_t outputHeight,
                                     const float2 &range, cudaStream_t stream);
    cudaError_t decoderPreprocessBGR(const float *hH, void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
                                     float *output, size_t outputWidth, size_t outputHeight,
                                     const float2 &range, cudaStream_t stream);
    // Preprocess from grayscale image for input into net expecting grayscale image
    cudaError_t decoderPreprocessGray(const float *hH, void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
                                      float *output, size_t outputWidth, size_t outputHeight,
                                      const float2 &range, cudaStream_t stream);
    // Preprocess from grayscale image for input into net expecting three channel image
    cudaError_t decoderPreprocessGrayForThreeChannelOutput(const float *hH, void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
                                                           float *output, size_t outputWidth, size_t outputHeight,
                                                           const float2 &range, cudaStream_t stream);

private:
    template <bool isBGR, bool isGray, bool threeOutputChannels>
    cudaError_t launchDecoderPreprocess(const float *hH, void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
                                        float *output, size_t outputWidth, size_t outputHeight,
                                        const float2 &range, cudaStream_t stream);
    float *m_dH;
};

#endif

