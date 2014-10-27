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
    //mve::image::blur_gaussian<float>(this->input.master, 1.0f);

    util::WallTimer timer;
    int const width = input.master->width();
    int const height = input.master->height();

    /* Iterate over all planes and compute cost volume. */
    float const id_min = 1.0f / this->opts.max_depth;
    float const id_max = 1.0f / this->opts.min_depth;
    std::vector<mve::FloatImage> cost_volume(this->opts.num_planes);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < this->opts.num_planes; ++i)
    {
#pragma omp critical
        std::cout << "\rProcessing plane " << i << " of "
            << this->opts.num_planes << "..." << std::flush;
        float const alpha = (float)i / (float)(this->opts.num_planes - 1);
        float const idepth = alpha * id_max + (1.0f - alpha) * id_min;

        /* Iterate over all neighboring views and warp to reference. */
        std::vector<mve::FloatImage::Ptr> pc_errors(this->input.views.size());
        for (std::size_t j = 0; j < this->input.views.size(); ++j)
        {
            pc_errors[j] = mve::FloatImage::create(width, height, 1);
            mve::FloatImage::Ptr warped = this->warp_view(j, 1.0f / idepth);
            this->photo_consistency(*this->input.master, *warped, &*pc_errors[j]);
            //pc_errors[j] = mve::image::blur_gaussian<float>(pc_errors[j], 2.0f);
            //pc_errors[j] = mve::image::blur_boxfilter<float>(pc_errors[j], 3); // Does not work?
        }

        /* Combine the photo-consistency errors for all views. */
        cost_volume[i].allocate(width, height, 1);
        this->combine_pc_errors(pc_errors, &cost_volume[i]);

#if 0
        std::string filename = get_filename("/tmp/total", i);
        mve::image::save_file(mve::image::float_to_byte_image(cost_volume[i].duplicate()), filename);
#endif
    }
    std::cout << std::endl;

    /* Compute result. */
    std::cout << "Computing output depth map..." << std::endl;
    this->compute_result(cost_volume, result);

    std::cout << "Done. Took " << timer.get_elapsed() << " ms." << std::endl;
}

mve::FloatImage::Ptr
PSS::warp_view (std::size_t id, float const depth)
{
    math::Matrix3f const& rmat = this->input.views[id].rmat;
    math::Vec3f const& rvec = this->input.views[id].rvec;

    int const iw = this->input.master->width();
    int const ih = this->input.master->height();
    int const ic = this->input.master->channels();
    float const width = static_cast<float>(iw);
    float const height = static_cast<float>(ih);

    /* Reproject the centers of the corner pixels to the neighbor view. */
    math::Vec3f const cm0(0.5f, 0.5f, 1.0f);
    math::Vec3f const cm1(0.5f, height - 0.5f, 1.0f);
    math::Vec3f const cm2(width - 0.5f, height - 0.5f, 1.0f);
    math::Vec3f const cm3(width - 0.5f, 0.5f, 1.0f);
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
        for (int x = 0; x < iw; ++x, warped_ptr += ic)
        {
            float const x_alpha = (float)x / (float)(iw - 1);

            /* Bilinear interpolation of the corners. */
            float const w0 = (1.0f - x_alpha) * (1.0f - y_alpha);
            float const w1 = (1.0f - x_alpha) * y_alpha;
            float const w2 = x_alpha * y_alpha;
            float const w3 = x_alpha * (1.0f - y_alpha);
            math::Vec3f p = w0 * ci0 + w1 * ci1 + w2 * ci2 + w3 * ci3;

            float const fx = p[0] / p[2];
            float const fy = p[1] / p[2];
            if (fx < 0.0f || fx > width || fy < 0.0f || fy > height)
            {
                std::fill(warped_ptr, warped_ptr + ic,
                    std::numeric_limits<float>::quiet_NaN());
//                warped_ptr[0] = 1.0f;
//                warped_ptr[1] = 0.0f;
//                warped_ptr[2] = 1.0f;
                continue;
            }

            image->linear_at(fx - 0.5f, fy - 0.5f, warped_ptr);
        }
    }

    return warped;
}

void
PSS::photo_consistency (mve::FloatImage const& view1,
    mve::FloatImage const& view2, mve::FloatImage* result)
{
    int const width = view1.width();
    int const height = view2.height();
    for (int i = 0, j = 0, y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x, ++i, j += 3)
        {
            result->at(i) = 0.0f;
            result->at(i) += std::abs(view1.at(j + 0) - view2.at(j + 0));
            result->at(i) += std::abs(view1.at(j + 1) - view2.at(j + 1));
            result->at(i) += std::abs(view1.at(j + 2) - view2.at(j + 2));
            result->at(i) /= 3.0f;
        }
}

void
PSS::combine_pc_errors (std::vector<mve::FloatImage::Ptr> const& pc_errors,
        mve::FloatImage* result)
{
    int const width = result->width();
    int const height = result->height();
    for (int i = 0, y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x, ++i)
        {
            float total_error = 0.0f;
            float norm = 0.0f;

#if 1 // Average
            for (std::size_t j = 0; j < pc_errors.size(); ++j)
            {
                float const pc = pc_errors[j]->at(i);
                if (std::isnan(pc))
                    continue;
                total_error += pc;
                norm += 1.0f;
            }
#endif

#if 0 // Occlusion robust average
            std::vector<float> values;
            for (std::size_t j = 0; j < pc_errors.size(); ++j)
            {
                float const pc = pc_errors[j]->at(i);
                if (std::isnan(pc))
                    continue;
                values.push_back(pc);
                total_error += pc;
                norm += 1;
            }

            if (values.size() > 4)
            {
                total_error = 0.0f;
                norm = 0.0f;
                std::sort(values.begin(), values.end());
                for (std::size_t j = 0; j < 2 * values.size() / 3; ++j)
                {
                    total_error += values[j];
                    norm += 1.0f;
                }
            }
#endif

            result->at(i) = total_error / norm;
        }
}

void
PSS::compute_result (std::vector<mve::FloatImage> const& cost_volume,
    PSS::Result* result)
{
    if (cost_volume.empty())
        throw std::invalid_argument("Empty cost volume");

    float const id_min = 1.0f / this->opts.max_depth;
    float const id_max = 1.0f / this->opts.min_depth;
    int const width = cost_volume[0].width();
    int const height = cost_volume[0].height();

    result->depth = mve::FloatImage::create(width, height, 1);
    result->conf = mve::FloatImage::create(width, height, 1);
    result->depth->fill(0.0f);
    for (int y = 0, i = 0; y < height; ++y)
        for (int x = 0; x < width; ++x, ++i)
        {
            float min_error = 1.0f;
            int min_index = -1;
            for (int p = 0; p < this->opts.num_planes; ++p)
            {
                mve::FloatImage const& plane = cost_volume[p];
                float const value = plane.at(i);
                if (value < min_error)
                {
                    min_error = value;
                    min_index = p;
                }
            }

            float const alpha = (float)min_index / (float)(this->opts.num_planes - 1);
            float const idepth = alpha * id_max + (1.0f - alpha) * id_min;
            float const depth = 1.0f / idepth;

            result->conf->at(i) = min_error;
            if (min_error < 0.5f)
                result->depth->at(i) = depth;
        }
}

MVS_NAMESPACE_END
