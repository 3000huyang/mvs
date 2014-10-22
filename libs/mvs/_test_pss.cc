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
    mve::View::Ptr master_view = scene->get_view_by_id(master_id);
    mve::ByteImage::Ptr master_image = master_view->get_byte_image("undist-L1");
    int const master_width = master_image->width();
    int const master_height = master_image->height();

    std::vector<int> neighbors;
    neighbors.push_back(15);
    neighbors.push_back(16);
    neighbors.push_back(17);
    neighbors.push_back(18);
    neighbors.push_back(19);
    neighbors.push_back(20);
    neighbors.push_back(90);
    neighbors.push_back(91);
    neighbors.push_back(92);
    neighbors.push_back(93);
    neighbors.push_back(94);

    mvs::PSS::Options pss_opts;
    pss_opts.min_depth = 3.0f; //feat1depth;
    pss_opts.max_depth = 6.0f; //feat2depth;
    pss_opts.num_planes = 100;

    mvs::PSS::Input pss_input;
    pss_input.master = mve::image::byte_to_float_image(master_image);

    for (std::size_t i = 0; i < neighbors.size(); ++i)
    {
        mve::View::Ptr neighbor_view = scene->get_view_by_id(neighbors[i]);
        mve::CameraInfo master_cam = master_view->get_camera();
        mve::CameraInfo neighbor_cam = neighbor_view->get_camera();

        mve::ByteImage::Ptr neighbor_image = neighbor_view->get_byte_image("undist-L1");
        int const neighbor_width = neighbor_image->width();
        int const neighbor_height = neighbor_image->height();

        mvs::PSS::NeighborView pss_neighbor;
        pss_neighbor.image = mve::image::byte_to_float_image(neighbor_image);
        master_cam.fill_reprojection(neighbor_cam,
            master_width, master_height, neighbor_width, neighbor_height,
            pss_neighbor.rmat.begin(), pss_neighbor.rvec.begin());
        pss_input.views.push_back(pss_neighbor);
    }

    mvs::PSS::Result pss_result;
    mvs::PSS pss(pss_opts, pss_input);
    pss.compute(&pss_result);

    return 0;
}
