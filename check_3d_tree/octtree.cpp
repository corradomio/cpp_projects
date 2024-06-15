//
// Created by Corrado Mio on 14/06/2024.
//

#include "octtree.h"

namespace octtree {

    node_s::node_s() {
        self.up = nullptr;
        for(int i=0; i<8; ++i)
            self.down[i] = 0;
    }

}