#include <fstream>
#include <iostream>
#include <stdexcept>

#include "mve/image.h"
#include "mve/image_tools.h"
#include "mve/image_io.h" // TMP
#include "mve/depthmap.h"

#include "mvs/ncc.h"

#include "stereo.h"

#define QUADRATIC_REFINEMENT 1
#define NUM_DEPTH_SAMPLES 50
#define DEPTH_MIN_FACTOR 0.95f
#define DEPTH_MAX_FACTOR 1.05f
#define MIN_NCC_SCORE 0.7f

void
Stereo::reconstruct (Progress* progress, Result* result)
{
    if (progress == NULL)
        throw std::invalid_argument("NULL progress given");
    this->progress = progress;
    (*this->progress) = Progress();

    if (this->master == NULL)
        throw std::invalid_argument("Master not initialized");
    if (this->neighbors.empty())
        throw std::invalid_argument("Neighbors not initialized");
    if (this->seeds.empty())
        throw std::invalid_argument("Seed points not initialized");

    this->opts.window_size = std::max(3, (this->opts.window_size / 2) * 2 + 1);
    this->depth.reset();
    this->queue = std::priority_queue<Pixel>();
    this->depth = mve::Image<Pixel>::create(
        this->master->width(), this->master->height(), 1);

    this->progress->status = PROGRESS_SEEDS;
    this->process_seeds();

    this->progress->status = PROGRESS_QUEUE;
    this->process_queue();

    result->depth_map = this->get_depth_map();
    this->progress->status = PROGRESS_DONE;
}

void
Stereo::process_seeds (void)
{
    std::cout << "Processing " << this->seeds.size()
        << " seed points..." << std::endl;

    //std::size_t i = 10; // TMP
    //this->reproject_test(i);
    for (std::size_t i = 0; i < this->seeds.size(); ++i)
    {
        /* Optimize seed depth value. */
        Pixel seed = this->seeds[i];
        this->optimize_depth(&seed);
        this->commit_depth(seed);
    }
}

void
Stereo::process_queue (void)
{
    std::cout << "Processing queue..." << std::endl;
    int iteration = 0;
    while (!this->queue.empty())
    {
        Pixel pixel = this->queue.top();
        this->queue.pop();
        this->optimize_depth(&pixel);
        this->commit_depth(pixel);

        iteration += 1;
        if (iteration % 1000 == 0)
            std::cout << "Queue: " << this->queue.size() << std::endl;
    }
}

void
Stereo::optimize_depth (Pixel* pixel)
{
    int const width = this->master->width();
    int const height = this->master->height();
    int const whs = this->opts.window_size / 2;
    int const index = pixel->y * width + pixel->x;

    /* Skip pixels that are already optimized. */
    if (pixel->confidence < this->depth->at(index).confidence)
        return;

    pixel->confidence = 0.0f;
    pixel->variance = 0.0f;

    /* Skip pixels outside master view. */
    if (pixel->x < whs || pixel->x >= width - whs
        || pixel->y < whs || pixel->y >= height - whs)
        return;

    /* Skip pixels with invalid depth. */
    if (pixel->depth == 0.0f)
        return;

    /* Compute depth range. */
    float const dmin = pixel->depth * DEPTH_MIN_FACTOR;
    float const dmax = pixel->depth * DEPTH_MAX_FACTOR;

    /* Compute depth sampling delta. */
    // TODO: Project depth min and max into image with largest parallax, then compute sampling from distance
    int num_samples = NUM_DEPTH_SAMPLES;

    /* Computes the optimal depth between dmin and dmax and updates pixel. */
    this->optimize_depth(pixel, dmin, dmax, num_samples);

#if 0
    if (pixel->confidence == 0.0f)
        std::cout << "Optimization failed for " << pixel->x << " " << pixel->y << std::endl;
    else
        std::cout << "Optimized depth for "
            << pixel->x << " " << pixel->y << " to " << pixel->depth
            << " (conf " << pixel->confidence << ")" << std::endl;
#endif
}

void
Stereo::commit_depth (Pixel const& pixel)
{
    int const width = this->master->width();
    int const index = pixel.y * width + pixel.x;

    if (pixel.confidence == 0.0f || pixel.depth == 0.0f
        || pixel.confidence <= this->depth->at(index).confidence)
        return;

    /* Add pixel to the final image. */
    this->depth->at(index) = pixel;

    /* Add neighbors to queue. */
    if (this->depth->at(index + 1).confidence < pixel.confidence - 0.05f)
    {
        Pixel neighbor = pixel;
        neighbor.x += 1;
        this->queue.push(neighbor);
    }
    if (this->depth->at(index - 1).confidence < pixel.confidence - 0.05f)
    {
        Pixel neighbor = pixel;
        neighbor.x -= 1;
        this->queue.push(neighbor);
    }
    if (this->depth->at(index + width).confidence < pixel.confidence - 0.05f)
    {
        Pixel neighbor = pixel;
        neighbor.y += 1;
        this->queue.push(neighbor);
    }
    if (this->depth->at(index - width).confidence < pixel.confidence - 0.05f)
    {
        Pixel neighbor = pixel;
        neighbor.y -= 1;
        this->queue.push(neighbor);
    }
}

mve::FloatImage::Ptr
Stereo::get_depth_map (void)
{
    int const width = this->master->width();
    int const height = this->master->height();
    mve::FloatImage::Ptr dm = mve::FloatImage::create(width, height, 1);
    dm->fill(0.0f);
    for (int i = 0; i < width * height; ++i)
        dm->at(i) = this->depth->at(i).depth;
    return dm;
}

bool
Stereo::patch_for_neighbor (int x, int y, float depth,
    std::size_t neighbor_id, std::vector<float>* patch)
{
    int const whs = this->opts.window_size / 2;
    NeighborView const& nb = this->neighbors[neighbor_id];
    int const width = nb.image->width();
    int const height = nb.image->height();

    /* Reproject samples and generate the patch. */
    for (int i = 0, py = y - whs; py <= y + whs; ++py)
        for (int px = x - whs; px <= x + whs; ++px)
        {
            /* Reproject sample point. */
            math::Vec3f p = nb.rmat * math::Vec3f(px + 0.5f, py + 0.5f, 1.0f) * depth + nb.rvec;
            p[0] = p[0] / p[2] - 0.5f;
            p[1] = p[1] / p[2] - 0.5f;

            /* Check if sample is valid. */
            if (p[0] < -0.5f || p[0] > width - 0.5f
                || p[1] < -0.5f || p[1] > height - 0.5f)
                return false;

            float values[3];
            nb.image->linear_at(p[0], p[1], values);
            for (int j = 0; j < 3; ++j, ++i)
                patch->at(i) = values[j];
        }

    return true;
}

void
Stereo::optimize_depth (Pixel* pixel, float dmin, float dmax, int num_samples)
{
    int const whs = this->opts.window_size / 2;
    int const patch_values = MATH_POW2(this->opts.window_size) * 3;
    int const num_neighbors = static_cast<int>(this->neighbors.size());

    /* Fill patch for master image. */
    std::vector<float> patch1(patch_values);
    for (int i = 0, y = pixel->y - whs; y <= pixel->y + whs; ++y)
        for (int x = pixel->x - whs; x <= pixel->x + whs; ++x)
        {
            float const* data = &this->master->at(x, y, 0);
            for (int c = 0; c < 3; ++c, ++i)
                patch1[i] = data[c];
        }

#if MVS_PATCH_TO_FILES
    patch_to_file(patch1, "/tmp/patch-master.png");
#endif

    std::vector<float> scores(num_neighbors * num_samples, -1.0f);
    float const dinc = (dmax - dmin) / (num_samples - 1);
#pragma omp parallel for
    for (int i = 0; i < num_samples; ++i)
    {
        std::vector<float> patch2(patch_values);
        float const depth = dmin + (float)i * dinc;
        //std::cout << "Processing patch " << i << " at depth " << depth << std::endl;
        for (int n = 0; n < num_neighbors; ++n)
        {
            /* Fill patch for neighboring image. */
            bool success = this->patch_for_neighbor(pixel->x, pixel->y, depth, n, &patch2);
            if (!success)
                continue;

            /* Compute NCC score for patches. */
            float score = mvs::ncc_score(&patch1[0], &patch2[0], patch_values, 3);
            scores[i * num_neighbors + n] = score;

#if MVS_PATCH_TO_FILES
            std::string filename = "/tmp/patch-neighbor-n"
                + util::string::get_filled(n, 3, '0')
                + "-d" + util::string::get(depth) + ".png";
            std::cout << "Saving patch " << filename << ", ncc: " << score << std::endl;
            patch_to_file(patch2, filename);
#endif
        }
    }

    /* Aggregate NCC scores for all neighbor into a single cost curve. */
    std::vector<float> cost_curve(num_samples);
    this->score_aggregation(num_neighbors, &scores, &cost_curve);

#define MVS_GNUPLOT_OUTPUT 0
#if MVS_GNUPLOT_OUTPUT
    std::ofstream out("/tmp/cost_curve.gpdata");
    for (int i = 0; i < num_samples; ++i)
    {
        float const depth = dmin + (float)i * dinc;
        out << depth;
        float avg = 0.0f;
        for (std::size_t n = 0; n < this->neighbors.size(); ++n)
        {
            out << " " << scores[i * num_neighbors + n];
            avg += scores[i * num_neighbors + n];
        }
        avg /= this->neighbors.size();
        out << " " << avg;
        out << " " << cost_curve[i];
        out << std::endl;
    }
    out.close();
#endif

    this->assign_best_score(cost_curve, dmin, dmax, pixel);
}

void
Stereo::score_aggregation (int num_neighbors,
    std::vector<float>* scores, std::vector<float>* cost_curve)
{
    int const num_samples = scores->size() / num_neighbors;
    int const num_best = (num_neighbors + 1) / 2;
    for (int s = 0; s < num_samples; ++s)
    {
        float* raw_scores = &scores->at(s * num_neighbors);
        std::sort(raw_scores, raw_scores + num_neighbors);
        float average = 0.0f;
        for (int n = 0; n < num_best; ++n)
            average += raw_scores[num_neighbors - n - 1];
        average /= static_cast<float>(num_best);
        cost_curve->at(s) = average;
    }
}

void
Stereo::assign_best_score (std::vector<float> const& cost_curve,
    float dmin, float dmax, Pixel* pixel)
{
    int const num_samples = static_cast<int>(cost_curve.size());
    float const dinc = (dmax - dmin) / (num_samples - 1);

    int max_id = -1;
    float max_score = -1.0f;
    for (int i = 0; i < num_samples; ++i)
    {
        if (cost_curve[i] > max_score)
        {
            max_score = cost_curve[i];
            max_id = i;
        }
    }

    float depth_offset = 0.0f;
#if QUADRATIC_REFINEMENT
    if (max_id != 0 || max_id != num_samples - 1)
    {
        float v[3] = { cost_curve[max_id - 1],
            cost_curve[max_id], cost_curve[max_id + 1] };

        float extremum = (v[0] - v[1]) / (2.0f * v[0] + 2.0f * v[2] - 4.0f * v[1]);
        depth_offset = dinc * extremum;
    }
#endif


    if (max_score < MIN_NCC_SCORE)
    {
        pixel->depth = 0.0f;
        pixel->confidence = 0.0f;
    }
    else
    {
        pixel->depth = dmin + max_id * dinc + depth_offset;
        pixel->confidence = (max_score - MIN_NCC_SCORE) / (1.0f - MIN_NCC_SCORE);
    }
}

// ---- DEBUG ----

void
Stereo::reproject_test (int seed_id)
{
    /* Pick one seed, draw in neighboring images. */
    Pixel seed = this->seeds[seed_id];
    mve::ByteImage::Ptr img = mve::image::float_to_byte_image(this->master);
    img->at(seed.x, seed.y, 0) = 255;
    img->at(seed.x, seed.y, 1) = 0;
    img->at(seed.x, seed.y, 2) = 0;
    mve::image::save_file(img, "/tmp/master.png");

    for (std::size_t i = 0; i < this->neighbors.size(); ++i)
    {
        NeighborView neighbor = this->neighbors[i];

        img = mve::image::float_to_byte_image(neighbor.image);
        math::Vec3f pos_master(seed.x + 0.5f, seed.y + 0.5f, 1.0f);
        math::Vec3f pos_neighbor = neighbor.rmat * pos_master * seed.depth + neighbor.rvec;
        pos_neighbor[0] = pos_neighbor[0] / pos_neighbor[2] - 0.5f;
        pos_neighbor[1] = pos_neighbor[1] / pos_neighbor[2] - 0.5f;
        int pos_x = std::min(neighbor.image->width() - 1, std::max(0, static_cast<int>(pos_neighbor[0] + 0.5f)));
        int pos_y = std::min(neighbor.image->height() - 1, std::max(0, static_cast<int>(pos_neighbor[1] + 0.5f)));

        img->at(pos_x, pos_y, 0) = 255;
        img->at(pos_x, pos_y, 1) = 0;
        img->at(pos_x, pos_y, 2) = 0;
        mve::image::save_file(img, "/tmp/neighbor-" + util::string::get_filled(i, 4, '0') + ".png");
    }
}

void
Stereo::patch_to_file (std::vector<float> const& patch,
    std::string const& fname)
{
    int const width = this->opts.window_size;
    mve::ByteImage::Ptr image = mve::ByteImage::create(width, width, 3);
    for (std::size_t i = 0; i < patch.size(); ++i)
        image->at(i) = static_cast<uint8_t>(patch[i] * 255.0f + 0.5f);
    image = mve::image::rescale<uint8_t>(image, mve::image::RESCALE_NEAREST, width * 10, width * 10);
    mve::image::save_file(image, fname);
}
