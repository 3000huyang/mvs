/*
 * Stereo strategy:
 * 1. Seed points are added through API.
 * 2. Seed points are processed and added to queue.
 * 3. Iterate:
 * 3.1. Take one pixel from queue
 * 3.2. Sweep pixel along depth and compute cost curve
 * 3.3. For every minimum cost, add sample to output
 * 3.4. For every minimum cost, add neighbor pixels to queue
 */

#ifndef STEREO_H
#define STEREO_H

#include <queue>
#include <set>

#include "mve/image.h"


class Stereo
{
public:
    /**
     * Stereo options.
     */
    struct Options
    {
        Options (void);
        int window_size;
    };

    /**
     * The neighboring view image and relative re-projection operator.
     */
    struct NeighborView
    {
        mve::FloatImage::ConstPtr image;
        math::Matrix3f rmat;
        math::Vec3f rvec;
    };

    /**
     * Stereo algorithm status.
     */
    enum ProgressStatus
    {
        PROGRESS_IDLE,
        PROGRESS_SEEDS,
        PROGRESS_QUEUE,
        PROGRESS_DONE
    };

    /**
     * Stereo algorithm progress.
     */
    struct Progress
    {
        Progress (void);
        ProgressStatus status;
        int queue_size;
    };

    struct Result
    {
        mve::FloatImage::Ptr depth_map;
    };

public:
    Stereo (Options const& opts);
    /** The master image values are expected to be in [0, 1]. */
    void set_master (mve::FloatImage::ConstPtr master);
    /** The neighboring image values are expected to be in [0, 1]. */
    void add_neighbor (NeighborView const& neighbor);
    /** Seeds points are required (usually SfM points). */
    void add_seed (int x, int y, float depth);
    /** Starts the reconstruction. */
    void reconstruct (Progress* progress, Result* result);

private:
    struct Pixel
    {
        Pixel (void);
        bool operator< (Pixel const& other) const;

        int x;
        int y;
        float depth;
        float confidence;
        float variance;
    };

private:
    void process_seeds (void);
    void process_queue (void);
    void optimize_depth (Pixel* pixel);
    void commit_depth (Pixel const& pixel);
    void optimize_depth (Pixel* pixel, float dmin, float dmax, int num_samples);
    bool patch_for_neighbor (int x, int y, float depth,
        std::size_t neighbor_id, std::vector<float>* patch);
    void score_aggregation (int num_neighbors,
        std::vector<float>* scores, std::vector<float>* cost_curve);
    void assign_best_score (std::vector<float> const& cost_curve,
        float dmin, float dmax, Pixel* pixel);
    mve::FloatImage::Ptr get_depth_map (void);

    // Debugging
    void reproject_test (int seed_id);
    void patch_to_file (std::vector<float> const& patch,
        std::string const& fname);

private:
    Options opts;
    Progress* progress;
    mve::FloatImage::ConstPtr master;
    std::vector<NeighborView> neighbors;
    std::vector<Pixel> seeds;
    mve::Image<Pixel>::Ptr depth;
    std::priority_queue<Pixel> queue;
};

/* ---------------------------------------------------------------- */

inline
Stereo::Options::Options (void)
    : window_size(5)
{
}

inline
Stereo::Progress::Progress (void)
    : status(PROGRESS_IDLE)
    , queue_size(0)
{
}

inline
Stereo::Pixel::Pixel (void)
    : x(0)
    , y(0)
    , depth(0.0f)
    , confidence(0.0f)
    , variance(0.0f)
{
}

inline bool
Stereo::Pixel::operator< (Pixel const& rhs) const
{
    return this->confidence < rhs.confidence;
}

inline
Stereo::Stereo (Options const& opts)
    : opts(opts)
    , progress(NULL)
{
}

inline void
Stereo::set_master (mve::FloatImage::ConstPtr master)
{
    this->master = master;
}

inline void
Stereo::add_neighbor (NeighborView const& neighbor)
{
    this->neighbors.push_back(neighbor);
}

inline void
Stereo::add_seed (int x, int y, float depth)
{
    Pixel seed;
    seed.x = x;
    seed.y = y;
    seed.depth = depth;
    this->seeds.push_back(seed);
}

#endif // STEREO_H

