#include <iostream>
#include <vector>

#include "util/exception.h"
#include "util/string.h"
#include "util/timer.h"
#include "math/vector.h"
#include "mve/image_io.h"
#include "mve/image_tools.h"
#include "mvs/view_selection.h"
#include "mvs/pss.h"

MVS_NAMESPACE_BEGIN

namespace
{
    std::string
    get_filename (std::string const& prefix, int number)
    {
        return prefix + "-" + util::string::get_filled(number, 3, '0') + ".png";
    }
}

void
PSS::compute (Result* result)
{
    if (this->opts.num_planes < 2)
        throw std::invalid_argument("Invalid number of planes");
    if (this->opts.num_hypothesis < 1)
        throw std::invalid_argument("Invalid number of hypothesis");
    if (this->input.views.empty())
        throw std::invalid_argument("No neighboring views given");
    if (this->input.master->channels() != 3)
        throw std::invalid_argument("Expecting 3-channel image");
    for (std::size_t i = 0; i < this->input.views.size(); ++i)
        if (this->input.views[i].image->channels() != 3)
            throw std::invalid_argument("Expecting 3-channel image");

    /* Blur master image a bit, because warping also causes blur. */
    mve::image::blur_gaussian<float>(this->input.master, 1.0f);

    util::WallTimer timer;

    /* Iterate over all planes. */
    float const id_min = 1.0f / this->opts.max_depth;
    float const id_max = 1.0f / this->opts.min_depth;
    for (int i = 0; i < this->opts.num_planes; ++i)
    {
        std::cout << "Processing plane " << i << "..." << std::endl;
        float const alpha = (float)i / (float)(this->opts.num_planes - 1);
        float const idepth = alpha * id_max + (1.0f - alpha) * id_min;

        /* Iterate over all neighboring views and warp to reference. */
        for (std::size_t j = 0; j < this->input.views.size(); ++j)
        {
            mve::FloatImage::Ptr warped = this->warp_view(j, 1.0f / idepth);
            mve::FloatImage::Ptr pc = this->photo_consistency(*warped, *this->input.master);

#if 0
            std::string filename = get_filename("/tmp/warped", i);
            //std::cout << "Saving warped frame " << filename << std::endl;
            //mve::image::save_file(mve::image::float_to_byte_image(warped), filename);
            filename = get_filename("/tmp/sad", i);
            mve::image::save_file(mve::image::float_to_byte_image(pc), filename);
#endif
        }
    }

    std::cout << "Done. Took " << timer.get_elapsed() << " ms." << std::endl;
}

mve::FloatImage::Ptr
PSS::warp_view (std::size_t id, float depth)
{
    math::Matrix3f const& rmat = this->input.views[id].rmat;
    math::Vec3f const& rvec = this->input.views[id].rvec;

    int iw = this->input.master->width();
    int ih = this->input.master->height();
    int ic = this->input.master->channels();
    float width = static_cast<float>(iw);
    float height = static_cast<float>(ih);

    /* Reproject the corners of the master view to the neighbor. */
    math::Vec3f const cm0(-0.5f, -0.5f, 1.0f);
    math::Vec3f const cm1(-0.5f, height - 0.5f, 1.0f);
    math::Vec3f const cm2(width - 0.5f, height - 0.5f, 1.0f);
    math::Vec3f const cm3(width - 0.5f, -0.5f, 1.0f);
    math::Vec3f const ci0 = rmat * cm0 * depth + rvec;
    math::Vec3f const ci1 = rmat * cm1 * depth + rvec;
    math::Vec3f const ci2 = rmat * cm2 * depth + rvec;
    math::Vec3f const ci3 = rmat * cm3 * depth + rvec;

    mve::FloatImage::Ptr warped = mve::FloatImage::create(iw, ih, ic);
    mve::FloatImage::ConstPtr image = this->input.views[id].image;
    float* warped_ptr = warped->get_data_pointer();
    for (int y = 0; y < ih; ++y)
    {
        float const y_alpha = (float)y / (float)(ih - 1);
        for (int x = 0; x < iw; ++x, warped_ptr += 3)
        {
            float const x_alpha = (float)x / (float)(iw - 1);

            /* Bilinear interpolation of the corners. */
            float const w0 = (1.0f - x_alpha) * (1.0f - y_alpha);
            float const w1 = (1.0f - x_alpha) * y_alpha;
            float const w2 = x_alpha * y_alpha;
            float const w3 = x_alpha * (1.0f - y_alpha);
            math::Vec3f p = w0 * ci0 + w1 * ci1 + w2 * ci2 + w3 * ci3;
            image->linear_at(p[0] / p[2], p[1] / p[2], warped_ptr);

            // TODO Check if outside image
        }
    }

    return warped;
}

mve::FloatImage::Ptr
PSS::photo_consistency (mve::FloatImage const& view1,
    mve::FloatImage const& view2)
{
    int const width = view1.width();
    int const height = view2.height();

    mve::FloatImage::Ptr pc = mve::FloatImage::create(width, height, 1);
    for (int i = 0, j = 0, y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x, ++i, j += 3)
        {
            pc->at(i) = 0;
            pc->at(i) += std::abs(view1.at(j + 0) - view2.at(j + 0));
            pc->at(i) += std::abs(view1.at(j + 1) - view2.at(j + 1));
            pc->at(i) += std::abs(view1.at(j + 2) - view2.at(j + 2));
            //pc->at(i) /= 3.0f;
        }

    //pc = mve::image::blur_gaussian<float>(pc, 3.0f);
    pc = mve::image::blur_boxfilter<float>(pc, 3);

    return pc;
}



MVS_NAMESPACE_END



#if 0

void
PSS::reconstruct (Options const& opts, Result* result)
{
    std::cout << "Starting reconstruction for view " << opts.view_id
        << " at scale " << opts.scale
        << " with " << opts.num_hypothesis << " hypothesis." << std::endl;

    /* Retrieve view and image. */
    mve::View::Ptr view = this->views->get_view(opts.view_id);
    if (view == NULL)
        throw util::Exception("Invalid master view ID",
            util::string::get(opts.view_id));

    /* Find min/max depth. */
    float min_depth, max_depth;
    this->find_depth_range(view, &min_depth, &max_depth);
    std::cout << "Computed depth range: " << min_depth
        << " to " << max_depth << std::endl;

    /* View selection. */
    std::vector<int> selected_views;
    {
        pss::ViewSelection::Options vs_opts;
        vs_opts.num_views = 20;
        pss::ViewSelection vs(vs_opts, this->views);
        vs.select(opts.view_id, &selected_views);
        if (selected_views.empty())
            throw util::Exception("View selection failed");
    }

    std::cout << "View selection:";
    for (std::size_t i = 0; i < selected_views.size(); ++i)
        std::cout << " " << selected_views[i];
    std::cout << std::endl;
}

void
PSS::find_depth_range (mve::View::Ptr view, float* min_depth, float* max_depth)
{
    math::Matrix4f wtc;
    view->get_camera().fill_world_to_cam(wtc.begin());
    math::Matrix3f proj;
    view->get_camera().fill_calibration(proj.begin(), 1.0f,  1.0f);

    mve::Bundle::Features const& features = this->bundle->get_features();
    std::vector<float> depths;
    depths.reserve(features.size());
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        mve::Bundle::Feature3D const& f3d = features[i];
        math::Vec3f fpos(f3d.pos);
        fpos = wtc.mult(fpos, 1.0f);
        float const depth = fpos[2];
        if (depth <= 0.0f)
            continue;

        fpos = proj.mult(fpos);
        fpos /= fpos[2];
        if (fpos[0] < 0.0f || fpos[0] > 1.0f
            || fpos[1] < 0.0f || fpos[1] > 1.0f)
            continue;
        depths.push_back(depth);
    }

    std::sort(depths.begin(), depths.end());
    *min_depth = depths[0];
    *max_depth = depths[19 * depths.size() / 20];
    *min_depth = std::max(*min_depth, 1e-4f);
    *max_depth = std::min(*max_depth, 1e4f);
}

#endif


