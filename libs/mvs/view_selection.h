/*
 * Multi-View Stereo view selection algorithm.
 * Written by Simon Fuhrmann.
 */

#ifndef MVS_VIEW_SELECTION_HEADER
#define MVS_VIEW_SELECTION_HEADER

#include <vector>

#include "math/vector.h"
#include "mve/scene.h"
#include "mve/camera.h"
#include "mve/bundle.h"
#include "mvs/defines.h"

MVS_NAMESPACE_BEGIN

/**
 * A global view selection algorithm. It selects for a given master view
 * a desired number of suitable neighboring views. A suitable view
 * is greedily determined based on already selected views and visibility
 * of features in the master view.
 */
class ViewSelection
{
public:
    struct Options
    {
        Options (void);

        int master_id;
        int num_views;
        float min_parallax;
    };

public:
    explicit ViewSelection (mve::Scene::Ptr scene);
    void select (Options const& options, std::vector<int>* result);

private:
    struct CachedView
    {
        bool valid;
        mve::CameraInfo cam;
        math::Vec3f campos;
    };
    typedef std::vector<CachedView> ViewCache;

private:
    float compute_score (std::size_t view_id,
        Options const& options,
        mve::Bundle::Features const& features,
        ViewCache const& cache,
        std::vector<int> const& result);

    float compute_parallax (math::Vec3f const& fpos,
        math::Vec3f const& campos1, math::Vec3f const& campos2);

private:
    mve::Scene::Ptr scene;
};

/* ------------------------ Implementation ------------------------ */

inline
ViewSelection::Options::Options (void)
    : master_id(-1)
    , num_views(10)
    , min_parallax(10.0f)
{
}

inline
ViewSelection::ViewSelection (mve::Scene::Ptr scene)
    : scene(scene)
{
}

MVS_NAMESPACE_END

#endif /* MVS_VIEW_SELECTION_HEADER */
