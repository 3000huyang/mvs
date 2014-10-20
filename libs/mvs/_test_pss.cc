#include <iostream>

#include "mve/scene.h"
#include "mve/image.h"
#include "mve/image_tools.h"
#include "mve/image_io.h"
#include "mvs/pss.h"

int main (int argc, char** argv)
{
    std::string scene_path = "/data/dev/phd/datasets/mve_paris_panther";
    mve::Scene::Ptr scene = mve::Scene::create(scene_path);

    int master_id = 18;
    int neighbor_id = 23;

    mve::Bundle::ConstPtr bundle = scene->get_bundle();
    mve::Bundle::Feature3D feat1 = bundle->get_features()[2856];
    math::Vec3f feat1pos(feat1.pos);
    mve::Bundle::Feature3D feat2 = bundle->get_features()[432];
    math::Vec3f feat2pos(feat2.pos);

    mve::View::Ptr master_view = scene->get_view_by_id(master_id);
    mve::View::Ptr neighbor_view = scene->get_view_by_id(neighbor_id);

    mve::ByteImage::Ptr master_image = master_view->get_byte_image("undist-L1");
    mve::ByteImage::Ptr neighbor_image = neighbor_view->get_byte_image("undist-L1");
    int const master_width = master_image->width();
    int const master_height = master_image->height();
    int const neighbor_width = neighbor_image->width();
    int const neighbor_height = neighbor_image->height();

    mve::CameraInfo master_cam = master_view->get_camera();
    mve::CameraInfo neighbor_cam = neighbor_view->get_camera();
    math::Vec3f master_campos, neighbor_campos;
    math::Vec3f master_trans, neighbor_trans;
    math::Matrix3f master_rot, neighbor_rot;
    math::Matrix3f master_calib, master_invcalib, neighbor_calib;
    master_cam.fill_camera_pos(master_campos.begin());
    master_cam.fill_camera_translation(master_trans.begin());
    master_cam.fill_world_to_cam_rot(master_rot.begin());
    master_cam.fill_inverse_calibration(master_invcalib.begin(), master_width, master_height);
    master_cam.fill_calibration(master_calib.begin(), master_width, master_height);
    neighbor_cam.fill_camera_pos(neighbor_campos.begin());
    neighbor_cam.fill_camera_translation(neighbor_trans.begin());
    neighbor_cam.fill_world_to_cam_rot(neighbor_rot.begin());
    neighbor_cam.fill_calibration(neighbor_calib.begin(), neighbor_width, neighbor_height);

    float feat1depth = (master_rot * feat1pos + master_trans)[2];
    float feat2depth = (master_rot * feat2pos + master_trans)[2];


#if 0
    mve::ByteImage::Ptr test = neighbor_image->duplicate();
    math::Vec3f px = master_calib * (master_rot * feat1pos + master_trans);
    px /= px[2];

    std::cout << "Feature pos: " << feat1pos << std::endl;

    std::cout << "Master pixel: " << px << ", depth " << feat1depth << std::endl;
    master_image->at(px[0], px[1], 0) = 255;
    master_image->at(px[0], px[1], 1) = 0;
    master_image->at(px[0], px[1], 2) = 0;
    mve::image::save_file(master_image, "/tmp/master.png");


    math::Vec3f rp = master_rot.transposed() * (master_invcalib * px * feat1depth - master_trans);
    std::cout << "Out projected: " << rp << std::endl;
    rp = neighbor_calib * (neighbor_rot * rp + neighbor_trans);
    rp /= rp[2];

    std::cout << "Neigbor pixel: " << rp << std::endl;
    test->at(rp[0], rp[1], 0) = 255;
    test->at(rp[0], rp[1], 1) = 0;
    test->at(rp[0], rp[1], 2) = 0;
    mve::image::save_file(test, "/tmp/warped.png");
#endif

#if 1

    mvs::PSS::NeighborView pss_neighbor;
    pss_neighbor.image = mve::image::byte_to_float_image(neighbor_image);
    pss_neighbor.rmat = neighbor_calib * neighbor_rot
        * master_rot.transposed() * master_invcalib;
    pss_neighbor.rvec = neighbor_calib * neighbor_trans
        - neighbor_calib * neighbor_rot * master_rot.transposed() * master_trans;

    mvs::PSS::Input pss_input;
    pss_input.master = mve::image::byte_to_float_image(master_image);
    pss_input.views.push_back(pss_neighbor);

    mvs::PSS::Options pss_opts;
    pss_opts.min_depth = feat1depth;
    pss_opts.max_depth = feat2depth;
    pss_opts.num_planes = 50;

    mvs::PSS::Result pss_result;
    mvs::PSS pss(pss_opts, pss_input);
    pss.compute(&pss_result);

    mve::FloatImage::Ptr warped = pss.warp_view(0, pss_opts.min_depth);
    mve::image::save_file(mve::image::float_to_byte_image(warped), "/tmp/warped.png");
    mve::image::save_file(master_image, "/tmp/master.png");
#endif
    return 0;
}
