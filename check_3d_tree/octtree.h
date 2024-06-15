//
// Created by Corrado Mio on 14/06/2024.
//

#ifndef CHECK_3D_TREE_OCTTREE_H
#define CHECK_3D_TREE_OCTTREE_H

#include <vector>
#include <language.h>

/**
 * octtree_t is only an index on another data structure, for example an
 * array, containing the data.
 */

namespace octtree {

    struct point_t {
        float x,y,z, w;

        point_t() = default;
        point_t(float x, float y, float z, float w=0)
            : x(x), y(y), z(z), w(w) { };
        point_t(const point_t& point) = default;
        point_t& operator =(const point_t& point) = default;
    };

    class octtree_s;

    struct node_s {
        point_t coords;

        // parent, nullptr for the root
        octtree_s *up;
        // sub spaces
        octtree_s *down[8];

        node_s();
    };

    struct octtree_s {

        node_s* root;

    };

    template<typename T>
    class octtree_t: public octtree_s {

        struct node_t: public node_s {
            std::vector<T*> content;
        };

    public:
        octtree_t() {
            self.root = new node_t();
        }
    };

}


#endif //CHECK_3D_TREE_OCTTREE_H
