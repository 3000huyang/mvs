/*
 * Notes for paper:
 *   Least Commitment, Viewpoint-based, Multi-view Stereo
 *   Xiaoyan Hu, Philippos Mordohai
 *
 * Notes Stereo:
 * - Use Plane Sweeping with NCC (7x7 pixel)
 * - Four sets of planes: 1 horizontal, 1 fronto-parallel, 2 rotated 45 deg.
 * - Pairwise NCC scores are averaged (excluding invisible regions)
 * - Keep one or multiple local NCC maxima regardless of sweeping direction
 * - Confidence values using "AML" (see paper) per sweeping direction
 *
 * Notes Fusion:
 * - Keep three local NCC maxima with confidence values
 * - Support radius depends on depth uncertainty (grows quadratic with depth)
 * - A confidence is updated by adding all confidences that support a hypothesis
 * - Depth is updated as confidence-weighted average of all depths that support a hypothesis
 * - Blended hypothesis are penalized if they are occluded on the reference view ray (confidence of occluding depth is subtracted)
 * - Blended hypothesis are rendered and checked for free-space violations by more than support range
 * - Blended hypothesis with largest confidence per pixel is selected as depth
 */
#include <string>
#include <vector>

#include "mve/scene.h"
#include "mve/view.h"
#include "mve/image.h"
#include "mvs/defines.h"

MVS_NAMESPACE_BEGIN

/**
 * Plane sweeping stereo implementation. It takes a reference image
 * and a set of neighboring images together with camera parameters as input
 * and produces a depth map and confidence map as output.
 *
 * The specified number of planes is distributed equally between the inverse
 * depth range from 1/max_depth to 1/min_depth.
 */
class PSS
{
public:
    /**
     * Plane sweeping stereo options.
     */
    struct Options
    {
        Options (void);

        float min_depth;
        float max_depth;
        int num_planes;
        int num_hypothesis;
    };

    /**
     * The neighboring view image and relative re-projection operator.
     *
     * The matrix is computed as: Ti,r = Ki Ri Rr^-1 Kr^-1.
     * The vector is computed as: ti,r = Ki ti - Ki Ri Rr^-1 tr.
     *
     * The reprojection of a pixel Pr with respect to depth d is then:
     * Pi = d * Ti,r * Pr + ti,r
     */
    struct NeighborView
    {
        mve::FloatImage::ConstPtr image;
        math::Matrix3f rmat;
        math::Vec3f rvec;
    };

    /**
     * Input data to the algrithm.
     */
    struct Input
    {
        mve::FloatImage::ConstPtr master;
        std::vector<NeighborView> views;
    };

    /**
     * The resulting depth and confidence map.
     */
    struct Result
    {
        mve::FloatImage::Ptr depth;
        mve::FloatImage::Ptr conf;
    };

public:
    PSS (Options const& options, Input const& input);
    void compute (Result* result);

public: //TMP
    mve::FloatImage::Ptr warp_view (std::size_t id, float const depth);
    void photo_consistency (mve::FloatImage const& view1,
        mve::FloatImage const& view2, mve::FloatImage* result);
    void combine_pc_errors (std::vector<mve::FloatImage::Ptr> const& pc_errors,
        mve::FloatImage* result);
    void compute_result (std::vector<mve::FloatImage> const& cost_volume,
        Result* result);

private:
    Options opts;
    Input input;
};

/* ------------------------ Implementation ------------------------ */

inline
PSS::Options::Options (void)
    : min_depth(0.01f)
    , max_depth(100.0f)
    , num_planes(100)
    , num_hypothesis(1)
{
}

inline
PSS::PSS (Options const& options, Input const& input)
    : opts(options)
    , input(input)
{
}

MVS_NAMESPACE_END
