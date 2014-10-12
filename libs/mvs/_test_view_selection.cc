#include <iostream>
#include <string>
#include <vector>

#include "mve/scene.h"
#include "mvs/view_selection.h"

int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Syntax: " << argv[0] << " SCENE" << std::endl;
        return 1;
    }

    std::string scene_path = argv[1];
    mve::Scene::Ptr scene = mve::Scene::create(scene_path);

    mvs::ViewSelection::Options opts;
    opts.master_id = 10;
    opts.num_views = 5;

    std::vector<int> vs_result;
    mvs::ViewSelection vs(scene);
    vs.select(opts, &vs_result);

    std::cout << "View selection:";
    for (std::size_t i = 0; i < vs_result.size(); ++i)
        std::cout << " " << vs_result.at(i);
    std::cout << std::endl;

    return 0;
}
