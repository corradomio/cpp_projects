//
// Created by Corrado Mio on 27/01/2023.
//

//using namespace std;

int fun(int p1, int p2, int p3) {
    if (p1 < p2)
        p3 = p1;
    else if (p2 < p1)
        p3 = p2;
}

int main8(){
    int r = fun(1,2,3);
}
