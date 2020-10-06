#include <iostream>
#include <math.h>



class Solido {
public:
    virtual std::string v_whoami(){ return "Un solido"; }    // ATTENZIONE: NOTARE IL ""virtual""
    std::string         s_whoami(){ return "Un solido"; }

    virtual double v_volume(){ return 0; }    // ATTENZIONE: NOTARE IL ""virtual""
            double s_volume(){ return 0; }
};

class Cubo : public Solido {
    double lato;
public:
    Cubo(double l): lato(l) { }

    virtual std::string v_whoami(){ return "Un cubo"; }    // ATTENZIONE: NOTARE IL ""virtual""
    std::string         s_whoami(){ return "Un cubo"; }

    virtual double v_volume(){ return lato*lato*lato; }
    double         s_volume(){ return lato*lato*lato; }
};

class Sfera : public Solido {
    double raggio;
public:
    Sfera(double r) : raggio(r) {}

    virtual std::string v_whoami() { return "Una sfera"; }    // ATTENZIONE: NOTARE IL ""virtual""
    std::string         s_whoami() { return "Una sfera"; }

    virtual double v_volume() { return 4. / 3. * M_PI * raggio * raggio * raggio; }
            double s_volume() { return 4. / 3. * M_PI * raggio * raggio * raggio; }
};

class Bilancia {
public:
    double v_peso(Solido *solido, double pesoSpecifico) {
        return pesoSpecifico * solido->v_volume();
    }

    double s_peso(Solido *solido, double pesoSpecifico) {
        return pesoSpecifico * solido->s_volume();
    }
};



void printWhoami(const char *name, Solido *p) {
    std::cout << "ptr di tipo Solido, valore di tipo ptr a " << name << ":" << std::endl
        << "    v:" << p->v_whoami() << std::endl
        << "    s:" << p->s_whoami() << std::endl;
}

void printCuboWhoami(const char *name, Cubo *p) {
    std::cout << "ptr di tipo Cubo, valore di tipo ptr a " << name << ":" << std::endl
        << "    v:" << p->v_whoami() << std::endl
        << "    s:" << p->s_whoami() << std::endl;
}

void printSferaWhoami(const char *name, Sfera *p) {
    std::cout << "ptr di tipo Sfera, valore di tipo ptr a " << name << ":" << std::endl;
    std::cout << "    v:" << p->v_whoami() << std::endl;
    std::cout << "    s:" << p->s_whoami() << std::endl;
}


int main() {
    Solido s;
    Cubo cu(2);
    Sfera sf(2);

    Solido *ps   = new Solido;
    Solido *pscu = new Cubo(2);
    Solido *pssf = new Sfera(2);
    Cubo   *pcu  = new Cubo(2);
    Sfera  *psf  = new Sfera(2);

    printWhoami("Solido", &s);
    printWhoami("Cubo", &cu);
    printWhoami("Sfera", &sf);

    std::cout <<std::endl;

    printWhoami("Solido", ps);
    printWhoami("Cubo", pscu);
    printWhoami("Sfera", pssf);
    printWhoami("Cubo", pcu);
    printWhoami("Sfera", psf);

    std::cout <<std::endl;

    printCuboWhoami("Cubo", &cu);
    printCuboWhoami("Cubo", (Cubo*)pscu);
    printCuboWhoami("Cubo", pcu);

    std::cout <<std::endl;

    printSferaWhoami("Sfera", &sf);
    printSferaWhoami("Sfera", (Sfera*)pssf);
    printSferaWhoami("Sfera", psf);

}
