// plot "/tmp/scores.txt" with lines, "/tmp/selected.txt" lc rgb "blue"

#include <string>
#include <fstream>

#include "util/arguments.h"
#include "mve/image.h"
#include "mve/image_io.h"
#include "mve/image_tools.h"

mve::FloatImage::Ptr
gradient_image (mve::ByteImage::ConstPtr img)
{
    if (img->channels() > 1)
        throw std::runtime_error("Supports only 1 channel image");

    int const iw = img->width();
    int const ih = img->height();
    mve::FloatImage::Ptr out = mve::FloatImage::create(iw, ih, 1);

    int idx = 0;
    for (int y = 0; y < ih; ++y)
        for (int x = 0; x < iw; ++x, ++idx)
        {
            int dx = (x == 0 || x == iw - 1)
                ? 0 : img->at(idx - 1) - img->at(idx + 1);
            int dy = (y == 0 || y == ih - 1)
                ? 0 : img->at(idx - iw) - img->at(idx + iw);
            int grad_mag = dx * dx + dy * dy;
            out->at(idx) = std::sqrt(static_cast<float>(grad_mag));
        }

    return out;
}

float
gradient_score (mve::FloatImage::Ptr image)
{
    float norm = static_cast<float>(image->get_value_amount());
    float sum = 0.0f;
    for (int i = 0; i < image->get_value_amount(); ++i)
        sum += image->at(i);
    return sum / norm;
}

struct ImageScore
{
    std::string filename;
    float gradient;
    float weight;
};

int
main (int argc, char** argv)
{
    /* Setup argument parser. */
    util::Arguments args;
    args.set_usage(argv[0], "[ OPTIONS ] IMAGES");
    args.set_exit_on_error(true);
    args.set_nonopt_minnum(1);
    args.set_description(""); // TODO
    args.add_option('n', "num-images", true, "Select number of images [100]");
    args.add_option('\0', "debug-gradient", true, "Debug filename for gradients []");
    args.add_option('\0', "debug-selected", true, "Debug filename for selected []");
    args.parse(argc, argv);

    std::string debug_gradient;
    std::string debug_selected;
    int num_images = 100;

    std::vector<ImageScore> scores;

    /* Parse arguments and store file names of images to load. */
    for (util::ArgResult const* arg = args.next_result(); arg != NULL; arg = args.next_result())
    {
        if (arg->opt != NULL)
        {
            if (arg->opt->lopt == "num-images")
                num_images = arg->get_arg<int>();
            else if (arg->opt->lopt == "debug-gradient")
                debug_gradient = arg->arg;
            else if (arg->opt->lopt == "debug-selected")
                debug_selected = arg->arg;
            else
            {
                std::cerr << "Invalid option: " << arg->opt << std::endl;
                return 1;
            }

            continue;
        }

        ImageScore score;
        score.filename = arg->arg;
        scores.push_back(score);
    }

    /* Load images and compute gradient. */
    int num_done = 0;
#pragma omp parallel for
    for (std::size_t i = 0; i < scores.size(); ++i)
    {
        ImageScore& score = scores[i];
#pragma omp critical
        {
            std::cerr << "\rProcessing " << num_done << " of "
                << scores.size() << "..." << std::flush;
            num_done += 1;
        }
        mve::ByteImage::Ptr image = mve::image::load_file(score.filename);
        image = mve::image::desaturate<uint8_t>(image, mve::image::DESATURATE_AVERAGE);
        mve::FloatImage::Ptr grad = gradient_image(image);
        score.gradient = gradient_score(grad);
        score.weight = 1.0f;
    }
    std::cerr << std::endl;

    /* Compute useful Gaussian sigma value. */
    float gaussian_sigma = scores.size() / num_images / 4.0f;
    int const num_neighbors = static_cast<int>(gaussian_sigma * 3.0f);

    /* Debug gradients to file. */
    if (!debug_gradient.empty())
    {
        std::ofstream out(debug_gradient.c_str());
        if (!out.good())
        {
            std::cerr << "Error opening score file!" << std::endl;
            return 1;
        }
        for (std::size_t j = 0; j < scores.size(); ++j)
        {
            out << j << " " << scores[j].gradient << std::endl;
        }
        out.close();
    }

    /* Debug selected images to file. */
    std::ofstream out;
    if (!debug_selected.empty())
    {
        out.open(debug_selected.c_str());
        if (!out.good())
        {
            std::cerr << "Error opening selected file!" << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < num_images; ++i)
    {
        /* Find image with best score. */
        float max_score = 0.0f;
        int max_score_id = 0;
        for (int j = 0; j < (int)scores.size(); ++j)
        {
            float this_score = scores[j].gradient * scores[j].weight;
            if (this_score > max_score)
            {
                max_score = this_score;
                max_score_id = j;
            }
        }

        std::cout << scores[max_score_id].filename << std::endl;

        /* Debug selected image to file. */
        if (out.good())
            out << max_score_id << " " << scores[max_score_id].gradient << std::endl;

        /* Down-weight neighboring images. */
        int min_iter = std::max(0, max_score_id - num_neighbors);
        int max_iter = std::min((int)scores.size() - 1, max_score_id + num_neighbors);
        for (int j = min_iter; j <= max_iter; ++j)
        {
            float weight = std::exp(-MATH_POW2(j - max_score_id) / (2.0f * MATH_POW2(gaussian_sigma)));
            scores[j].weight *= 1.0f - weight;
        }
    }

    out.close();

    return 0;
}
