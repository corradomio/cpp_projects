//
// Created by Corrado Mio on 06/10/2020.
//

#include <nana/gui.hpp>
#include <nana/gui/widgets/button.hpp>

using namespace nana;

int main1()
{
    form fm;
    fm.caption(L"Hello, World!");
    button btn(fm, rectangle{20, 20, 150, 30});
    btn.caption(L"Quit");
    btn.events().click(API::exit);
    fm.show();
    exec();

    return 0;
}