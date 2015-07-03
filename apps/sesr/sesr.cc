/**
 * Signed error stereo reconstruction.
 * Written by Simon Fuhrmann.
 */

#include <iostream>
#include <string>

#include "util/string.h"
#include "util/arguments.h"
#include "mve/scene.h"
#include "mve/image.h"
#include "mve/depthmap.h"
#include "mve/image_tools.h"
#include "mvs/view_selection.h"

#include "stereo.h"

struct AppSettings
{
    std::string scene_path;
    std::string source_image;
    std::string target_depth;
    std::string target_image;
    std::string target_conf;
    int master_id;
    int num_neighbors;
    int scale;
};

mve::ByteImage::Ptr
rescale_image (AppSettings const& conf, mve::ByteImage::Ptr image)
{
    mve::ByteImage::Ptr ret = image;
    for (int scale = conf.scale; scale > 0; --scale)
        ret = mve::image::rescale_half_size<uint8_t>(ret);
    return ret;
}

int
reconstruct (AppSettings const& conf)
{
    /* Load scene. */
    mve::Scene::Ptr scene = mve::Scene::create(conf.scene_path);

    /* View selection. */
    std::vector<int> selected_views;
    {
        std::cout << "Starting view selection for master view "
            << conf.master_id << "..." << std::endl;
        mvs::ViewSelection::Options vs_opts;
        vs_opts.master_id = conf.master_id;
        vs_opts.num_views = conf.num_neighbors;
        vs_opts.image_name = conf.source_image;
        mvs::ViewSelection vs(scene);
        vs.select(vs_opts, &selected_views);

        std::cout << "Selected views:";
        for (std::size_t i = 0; i < selected_views.size(); ++i)
            std::cout << " " << selected_views[i];
        std::cout << std::endl;
    }

    /* Get master image. */
    mve::View::Ptr master_view = scene->get_view_by_id(conf.master_id);
    if (master_view == NULL)
        throw std::runtime_error("NULL master view");
    mve::CameraInfo master_cam = master_view->get_camera();
    if (master_cam.flen == 0.0f)
        throw std::invalid_argument("Invalid master camera");
    mve::ByteImage::Ptr master_img
        = master_view->get_byte_image(conf.source_image);
    if (master_img == NULL)
        throw std::runtime_error("NULL master image");
    master_img = rescale_image(conf, master_img);
    int const master_width = master_img->width();
    int const master_height = master_img->height();
    math::Matrix3f master_cal, master_rot;
    math::Vec3f master_trans;
    master_cam.fill_calibration(master_cal.begin(), master_width, master_height);
    master_cam.fill_world_to_cam_rot(master_rot.begin());
    master_cam.fill_camera_translation(master_trans.begin());

    std::cout << "Loaded view ID "
        << conf.master_id << " (master, "
        << master_width << "x" << master_height << ")..."
        << std::endl;

    /* Initialize stereo. */
    Stereo::Options stereo_opts;
    Stereo stereo(stereo_opts);
    stereo.set_master(mve::image::byte_to_float_image(master_img));

    /* Add neighboring images. */
    for (std::size_t i = 0; i < selected_views.size(); ++i)
    {
        mve::View::Ptr neighbor_view = scene->get_view_by_id(selected_views[i]);
        mve::ByteImage::Ptr neighbor_img = neighbor_view->get_byte_image(conf.source_image);
        neighbor_img = rescale_image(conf, neighbor_img);

        mve::CameraInfo neighbor_cam = neighbor_view->get_camera();
        int const neighbor_width = neighbor_img->width();
        int const neighbor_height = neighbor_img->height();

        std::cout << "Loaded view ID "
            << selected_views[i] << " (neighbor, "
            << neighbor_width << "x" << neighbor_height << ")..."
            << std::endl;

        Stereo::NeighborView neighbor;
        neighbor.image = mve::image::byte_to_float_image(neighbor_img);
        master_cam.fill_reprojection(neighbor_cam, master_width, master_height,
            neighbor_width, neighbor_height,
            neighbor.rmat.begin(), neighbor.rvec.begin());

        stereo.add_neighbor(neighbor);
    }

    /* Add seed points. */
    std::cout << "Adding SfM seed points..." << std::endl;
    mve::Bundle::ConstPtr bundle = scene->get_bundle();
    mve::Bundle::Features const& feats = bundle->get_features();
    for (std::size_t i = 0; i < feats.size(); ++i)
    {
        bool use_feature = false;
        mve::Bundle::Feature3D const& feat = feats[i];
        if (feat.contains_view_id(conf.master_id))
            use_feature = true;
        for (std::size_t j = 0; !use_feature && j < selected_views.size(); ++j)
            if (feat.contains_view_id(selected_views[j]))
                use_feature = true;
        if (!use_feature)
            continue;

        math::Vec3f pos3d(feat.pos);
        math::Vec3f pos2d = master_cal * (master_rot * pos3d + master_trans);
        pos2d[0] = pos2d[0] / pos2d[2] - 0.5f;
        pos2d[1] = pos2d[1] / pos2d[2] - 0.5f;
        stereo.add_seed(static_cast<int>(pos2d[0] + 0.5f),
            static_cast<int>(pos2d[1] + 0.5f), pos2d[2]);
    }

    /* Run stereo. */
    std::cout << "Starting stereo reconstruction..." << std::endl;
    Stereo::Progress progress;
    Stereo::Result result;
    stereo.reconstruct(&progress, &result);

    math::Matrix3f invcal;
    master_cam.fill_inverse_calibration(invcal.begin(),
        master_width, master_height);
    mve::image::depthmap_convert_conventions<float>
        (result.depth_map, invcal, true);

    master_view->set_image(result.depth_map, conf.target_depth);
    master_view->save_view();

    return 0;
}

int
main (int argc, char** argv)
{
    /* Setup argument parser. */
    util::Arguments args;
    args.set_usage(argv[0], "[ OPTIONS ] SCENE_PATH");
    args.set_exit_on_error(true);
    args.set_nonopt_maxnum(1);
    args.set_nonopt_minnum(1);
    //args.set_helptext_indent(22);
    args.set_description("Runs stereo for the given view.");
    args.add_option('m', "master-id", true, "Master view ID to reconstruct");
    args.add_option('i', "image", true, "Source image embedding [undistorted]");
    args.add_option('s', "scale", true, "Reconstruction scale (0 is original size) [2]");
    args.add_option('\0', "target-depth", true, "Target depthmap embedding [auto]");
    args.add_option('\0', "target-conf", true, "Target confidence map embedding [auto]");
    args.add_option('\0', "target-image", true, "Target scaled image embedding [auto]");
    args.parse(argc, argv);

    /* Default settings. */
    AppSettings conf;
    conf.scene_path = args.get_nth_nonopt(0);
    conf.source_image = "undistorted";
    conf.master_id = -1;
    conf.scale = 2;
    conf.num_neighbors = 10;

    /* Parse arguments. */
    for (util::ArgResult const* arg = args.next_option();
        arg != NULL; arg = args.next_option())
    {
        if (arg->opt->lopt == "image")
            conf.source_image = arg->arg;
        else if (arg->opt->lopt == "master-id")
            conf.master_id = arg->get_arg<int>();
        else if (arg->opt->lopt == "scale")
            conf.scale = arg->get_arg<int>();
        else if (arg->opt->lopt == "target-depth")
            conf.target_depth = arg->arg;
        else if (arg->opt->lopt == "target-conf")
            conf.target_conf = arg->arg;
        else if (arg->opt->lopt == "target-image")
            conf.target_image = arg->arg;
        else
            throw std::invalid_argument("Unrecognized option");
    }

    if (conf.target_depth.empty())
        conf.target_depth = "depth-L" + util::string::get(conf.scale);
    if (conf.target_conf.empty())
        conf.target_conf = "conf-L" + util::string::get(conf.scale);
    if (conf.scale > 0 && conf.target_image.empty())
        conf.target_image = "undist-L" + util::string::get(conf.scale);

    /* Sanity checks. */
    if (conf.master_id < 0)
    {
        args.generate_helptext(std::cerr);
        std::cerr << std::endl << "No master view ID given. Exiting." << std::endl;
        return 1;
    }

    if (conf.target_depth == conf.source_image
        || conf.target_image == conf.source_image)
    {
        std::cerr << "Source and target image/depth are the same." << std::endl;
        return 1;
    }

    return reconstruct(conf);
}
