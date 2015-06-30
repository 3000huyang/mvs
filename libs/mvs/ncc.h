#ifndef MVS_NCC_HEADER
#define MVS_NCC_HEADER

#include <vector>

#include "mvs/defines.h"

MVS_NAMESPACE_BEGIN

/**
 * Computes the NCC score for the two vectors 'v1' and 'v2' with 'size'
 * elements. The channels is specified, the mean is computed and subtracted
 * for each channel separately. Color values are assumed to be interleaved.
 */
template <typename T>
T
ncc_score (T const* v1, T const* v2, int size, int channels = 1);

/* ---------------------------------------------------------------- */

template <typename T>
T
ncc_score (T const* v1, T const* v2, int size, int channels)
{
    /* Compute mean per channel. */
    std::vector<T> mean_v1(channels, 0.0f);
    std::vector<T> mean_v2(channels, 0.0f);
    for (int i = 0; i < size; ++i)
    {
        mean_v1[i % channels] += v1[i];
        mean_v2[i % channels] += v2[i];
    }
    for (int i = 0; i < channels; ++i)
    {
        mean_v1[i] /= T(size / channels);
        mean_v2[i] /= T(size / channels);
    }

    /* Compute length and dot product. */
    T length_v1(0);
    T length_v2(0);
    T dot_product(0);
    for (int i = 0; i < size; ++i)
    {
        T value_v1 = v1[i] - mean_v1[i % channels];
        T value_v2 = v2[i] - mean_v2[i % channels];
        length_v1 += value_v1 * value_v1;
        length_v2 += value_v2 * value_v2;
        dot_product += value_v1 * value_v2;
    }
    dot_product /= std::sqrt(length_v1 * length_v2);

    return dot_product;
}

MVS_NAMESPACE_END

#endif // MVS_NCC_HEADER
