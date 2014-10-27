#include <iostream>
#include <string>

#include "util/string.h"
#include "util/arguments.h"
#include "mve/scene.h"
#include "mve/image.h"
#include "mve/depthmap.h"
#include "mve/image_tools.h"
#include "mvs/pss.h"
#include "mvs/view_selection.h"

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

void
find_depth_range (mve::Scene::Ptr scene, int master_id,
    float* min_depth, float* max_depth)
{
    mve::Bundle::ConstPtr bundle = scene->get_bundle();
    mve::CameraInfo const& cam = scene->get_view_by_id(master_id)->get_camera();
    math::Matrix3f cam_rot;
    math::Vec3f cam_trans;
    cam.fill_world_to_cam(cam_rot.begin());
    cam.fill_camera_translation(cam_trans.begin());

    std::vector<float> depths;
    mve::Bundle::Features const& feats = bundle->get_features();
    for (std::size_t i = 0; i < feats.size(); ++i)
    {
        mve::Bundle::Feature3D const& f3d = feats[i];
        if (!f3d.contains_view_id(master_id))
            continue;
        math::Vec3f x = cam_rot * math::Vec3f(f3d.pos) + cam_trans;
        depths.push_back(x[2]);
    }

    std::sort(depths.begin(), depths.end());
    *min_depth = depths[1 * depths.size() / 20];
    *max_depth = depths[9 * depths.size() / 10];
}

int
reconstruct_view (AppSettings const& conf,
    mve::Scene::Ptr scene, std::size_t master_id)
{
    /* Run view selection. */
    std::vector<int> selected_views;
    {
        mvs::ViewSelection::Options vs_opts;
        vs_opts.master_id = master_id;
        vs_opts.num_views = conf.num_neighbors;
        mvs::ViewSelection vs(scene);
        vs.select(vs_opts, &selected_views);

        std::cout << "Selected views:";
        for (std::size_t i = 0; i < selected_views.size(); ++i)
            std::cout << " " << selected_views[i];
        std::cout << std::endl;
    }

    /* Run PSS. */
    mvs::PSS::Result pss_result;
    try
    {
        std::cout << "Loading view ID "
            << master_id << " (master)..." << std::endl;

        /* Prepare PSS input. */
        mve::View::Ptr master_view = scene->get_view_by_id(master_id);
        if (master_view == NULL)
            throw std::invalid_argument("Invalid master view");

        mve::ByteImage::Ptr master_image = master_view->get_byte_image(conf.source_image);
        if (master_image == NULL)
            throw std::invalid_argument("Invalid master image");
        int const master_width = master_image->width();
        int const master_height = master_image->height();

        mve::CameraInfo master_cam = master_view->get_camera();
        if (master_cam.flen == 0.0f)
            throw std::invalid_argument("Invalid master camera");

        mvs::PSS::Input pss_input;
        pss_input.master = mve::image::byte_to_float_image(master_image);
        for (std::size_t i = 0; i < selected_views.size(); ++i)
        {
            std::cout << "Loading view ID "
                << selected_views[i] << " (neighbor)..." << std::endl;

            mve::View::Ptr neighbor_view = scene->get_view_by_id(selected_views[i]);
            mve::ByteImage::Ptr neighbor_image = neighbor_view->get_byte_image(conf.source_image);
            mve::CameraInfo neighbor_cam = neighbor_view->get_camera();
            int const neighbor_width = neighbor_image->width();
            int const neighbor_height = neighbor_image->height();

            mvs::PSS::NeighborView pss_neighbor;
            pss_neighbor.image = mve::image::byte_to_float_image(neighbor_image);
            master_cam.fill_reprojection(neighbor_cam,
                master_width, master_height, neighbor_width, neighbor_height,
                pss_neighbor.rmat.begin(), pss_neighbor.rvec.begin());

            pss_input.views.push_back(pss_neighbor);
        }

        /* Prepare PSS Options. */
        mvs::PSS::Options pss_opts;
        find_depth_range(scene, master_id,
            &pss_opts.min_depth, &pss_opts.max_depth);
        pss_opts.num_planes = 200;

        std::cout << "Depth range: " << pss_opts.min_depth
            << " to " << pss_opts.max_depth << std::endl;

        /* Run PSS. */
        mvs::PSS pss(pss_opts, pss_input);
        pss.compute(&pss_result);

        /* Convert depth map conventions. */
        math::Matrix3f master_inv_proj;
        master_cam.fill_inverse_calibration(master_inv_proj.begin(),
            pss_result.depth->width(), pss_result.depth->height());
        mve::image::depthmap_convert_conventions<float>(pss_result.depth,
            master_inv_proj, true);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "PSS returned an error. Exiting." << std::endl;
        return 1;
    }

    if (pss_result.depth == NULL)
    {
        std::cerr << "No result depth map. Exiting." << std::endl;
        return 1;
    }


    /* Save result back. */
    mve::View::Ptr view = scene->get_view_by_id(master_id);
    if (!conf.target_depth.empty())
        view->set_image(conf.target_depth, pss_result.depth);
    if (!conf.target_conf.empty())
        view->set_image(conf.target_conf, pss_result.conf);
    view->save_mve_file();

    return 0;
}


int
reconstruct (AppSettings const& conf)
{
    mve::Scene::Ptr scene;
    try
    {
        scene = mve::Scene::create(conf.scene_path);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (conf.master_id < 0)
    {
        /* TODO: Reconstruct all views. */
        std::cerr << "Please specify a master view ID." << std::endl;
        return 1;
    }

    return reconstruct_view(conf, scene, conf.master_id);
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
    args.set_description("Runs Plane Sweeping Stereo (PSS) for the given view.");
    args.add_option('i', "image", true, "Source image embedding [undistorted]");
    args.add_option('m', "master-id", true, "Master view ID to reconstruct");
    args.add_option('s', "scale", true, "Reconstruction scale (0 is original size) [2]");
    args.add_option('\0', "target-depth", true, "Target depthmap embedding [auto]");
    args.add_option('\0', "target-conf", true, "Target confidence map embedding []");
    args.add_option('\0', "target-image", true, "Target scaled image embedding [auto]");
    args.parse(argc, argv);

    /* Default settings. */
    AppSettings conf;
    conf.scene_path = args.get_nth_nonopt(0);
    conf.source_image = "undistorted";
    conf.master_id = -1;
    conf.scale = 2;
    conf.num_neighbors = 15;

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
