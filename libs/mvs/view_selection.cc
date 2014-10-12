#include <iostream>

#include "mvs/view_selection.h"

MVS_NAMESPACE_BEGIN

void
ViewSelection::select (Options const& options, std::vector<int>* result)
{
    mve::Scene::ViewList const& views = this->scene->get_views();
    if (options.master_id < 0 || options.master_id >= (int)views.size())
        throw std::invalid_argument("Invalid master view ID");

    /* Cache suitable views. */
    ViewCache cache(views.size());
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        cache[i].valid = false;
        if (views[i] == NULL)
            continue;
        if (views[i]->get_camera().flen == 0.0f)
            continue;
        // TODO Check if embedding exists.

        cache[i].valid = true;
        cache[i].cam = views[i]->get_camera();
        cache[i].cam.fill_camera_pos(cache[i].campos.begin());
    }

    if (!cache[options.master_id].valid)
        throw std::invalid_argument("Invalid master view");

    /* Select bundle features from the master view. */
    mve::Bundle::ConstPtr bundle = this->scene->get_bundle();
    mve::Bundle::Features const& bundle_features = bundle->get_features();
    mve::Bundle::Features master_features;
    master_features.reserve(1000);
    for (std::size_t i = 0; i < bundle_features.size(); ++i)
    {
        if (bundle_features[i].contains_view_id(options.master_id))
            master_features.push_back(bundle_features[i]);
    }

    /* Keep selecting views. */
    result->clear();
    while ((int)result->size() < options.num_views)
    {
        /* Find view with best score. */
        float best_score = 0.0f;
        int best_view = -1;

        for (std::size_t i = 0; i < cache.size(); ++i)
        {
            if ((int)i == options.master_id || !cache[i].valid)
                continue;

            float score = this->compute_score(i, options, master_features,
                cache, *result);
            if (score > best_score)
            {
                best_score = score;
                best_view = static_cast<int>(i);
            }
        }

        /* Break if no more views are available. */
        if (best_view < 0)
            break;

        result->push_back(best_view);
        cache[best_view].valid = false;
    }
}

float
ViewSelection::compute_score (std::size_t view_id,
    Options const& options,
    mve::Bundle::Features const& features,
    ViewCache const& cache,
    std::vector<int> const& result)
{
    float total_score = 0.0f;

    /* Iterate over all features. */
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (!features[i].contains_view_id(view_id))
            continue;

        float score = 1.0f;
        math::Vec3f fpos(features[i].pos);

        /* Compute parallax with master view. */
        float parallax = this->compute_parallax(fpos,
            cache[view_id].campos, cache[options.master_id].campos);
        if (parallax < options.min_parallax)
            score *= std::sqrt(parallax / options.min_parallax);

        /* Check resolution compared to master view. */
        // TODO

        /* Check feature parallax with already selected views. */
        for (std::size_t j = 0; j < result.size(); ++j)
        {
            parallax = this->compute_parallax(fpos,
                cache[view_id].campos, cache[result[j]].campos);
            if (parallax < options.min_parallax)
                score *= std::sqrt(parallax / options.min_parallax);
        }

        total_score += score;
    }

    return total_score;
}

float
ViewSelection::compute_parallax (math::Vec3f const& fpos,
    math::Vec3f const& campos1, math::Vec3f const& campos2)
{
    math::Vec3f const dir1 = (fpos - campos1).normalized();
    math::Vec3f const dir2 = (fpos - campos2).normalized();
    float const dot = math::clamp(dir1.dot(dir2), -1.0f, 1.0f);
    return MATH_RAD2DEG(std::acos(dot));
}

MVS_NAMESPACE_END
