//
// Created by Corrado Mio on 19/09/2020.
//
#include <algorithm>
#include <boostx/date_time_op.h>
#include "infections.h"
#include <stdx/cmath.h>
#include <stdx/ranges.h>
#include <stdx/to_string.h>

using namespace hls::khalifa::summer;
using namespace stdx::math;
using namespace boost::posix_time;


// --------------------------------------------------------------------------
// disease_t
// --------------------------------------------------------------------------

double disease_t::infective(const history_t& histo, int t) {
    int t0 = histo.when_infected();
    if (t0 == invalid)
        return 0.;

    int dt = t - t0;
    if (latent < dt && dt <= removed)
        return 1.;
    else
        return 0.;
}


double disease_t::symptomatic(const history_t& histo, int t) {
    int t0 = histo.when_infected();
    if (t0 == invalid)
        return 0.;

    int dt = t - t0;
    if (asymptomatic < dt && dt <= removed)
        return 1.;
    else
        return 0.;
}

// --------------------------------------------------------------------------
// prob_hist_t
// --------------------------------------------------------------------------

double prob_hist_t::get(int t) const {
    size_t n = hist.size();
    for (size_t i=n-1; i>0; --i)
        if (hist[i].t <= t)
            return hist[i].prob;
        else
            continue;
    return hist[0].prob;
}


void prob_hist_t::set(int t, double p)  {
    tprob_t& entry = hist[hist.size()-1];
    if (t == entry.t)
        entry.prob = p;

    // for the infected users, the data structure is already initialized
    // in this case, t can be PREVIOUS than entry.t
    else if (t < entry.t)
        return;
    else if (p != entry.prob)
        hist.emplace_back(t, p);
    if (p > 0 && positive == invalid)
        positive = t;
}


double prob_hist_t::update(int t, double u) {
    double p = get(t);
    double q = 1 - (1 - p)*(1 - u);
    set(t, q);
    return q;
}


double prob_hist_t::scale(int t, double u) {
    double p = get(t);
    double q = p*(1 - p*u);
    set(t, q);
    return q;
}


// --------------------------------------------------------------------------
// Infections
// --------------------------------------------------------------------------

Infections::Infections() {
    d = 2;
    beta = 0.001;
    a = 0;
    l = 0;
    r = 0;
    tp = 0.01;
    sp = 0.20;
    ce = 1.0;
    tau = 1.0;

    _disease.latent = 0;
    _disease.asymptomatic = 0;
    _disease.removed = -invalid;

    _only_infections = true;
    seed = 123;
}


// --------------------------------------------------------------------------
// Infected users
// --------------------------------------------------------------------------

/// Quota [0,1] of infected users
Infections& Infections::infected(double quota) {
    return infected(dworld().users(quota));
}


/// Number of infected users
Infections& Infections::infected(int n) {
    return infected(dworld().users(n));
}


/// Set the list of infected ids
Infections& Infections::infected(const s_users& users) {
    _infected.clear();
    _infected.insert(users.begin(), users.end());
    _infections.clear();
    return *this;
}


// --------------------------------------------------------------------------
// Propagate
// --------------------------------------------------------------------------

Infections& Infections::propagate() {
    std::cout << "Infection::propagate ..." << std::endl;

    init_simulation();
    init_infected();

    propagate_infection();

    std::cout << "Infection::end" << std::endl;
    return *this;
}


// --------------------------------------------------------------------------
// Propagate::initialize
// --------------------------------------------------------------------------

/**
 * Initialize the simulation parameters
 */
void Infections::init_simulation() {
    std::cout << "Infection::init_simulation" << std::endl;

    // day in time slots
    dts = int(time_duration(24, 0, 0)/dworld().interval_td());

    // cell side
    double D = dworld().side();

    // dt: time slot in fraction of day
    // used because beta is specified in term of day
    double dt = dworld().interval_td()/time_duration(24,0,0);

    // (1-exp(-beta*dt))
    double betadt = (1 - exp(-beta*dt));
    // (d/D)^2
    double dratio = sq(d/D);

    // factor of infection tau=(1-exp(-beta*dt))*(d/D)^2
    tau = betadt*dratio;

    // disease information
    _disease.asymptomatic = a*dts;  // asymptomatic in time slots
    _disease.latent = l*dts;        // latent in time slots
    _disease.removed = r*dts;       // removed in time slots
}


/**
 * Initialize the infected users and the other users
 */
void Infections::init_infected() {
    std::cout << "Infection::init_infected" << std::endl;

    _infections.clear();

    // initialize the '_infections' data structure
    for (const user_t& user : dworld().users())
        _infections[user].set(user, _disease, test_prob());

    // scan the encounters to identify the timeslot when an infected user
    // is encountered for the first time. This timeslot MINUS the lts (latent
    // time slots) will be the 'infected' timeslot.
    bool complete = false;
    size_t n_users = dworld().users().size();
    std::unordered_set<user_t> processed;

    // t->user->{user,...}
    const tms_users &encs = dworld().get_time_encounters();
    for (auto it = encs.cbegin(); it != encs.cend() && !complete; ++it) {
        // time slot
        int t = it->first;

        const ms_users& musers = it->second;

        for (auto uit = musers.cbegin(); uit != musers.end() && !complete; ++uit) {
            const user_t& user = uit->first;

            // it is an already processed user
            if (stdx::contains(processed, user))
                continue;

            // it is an infected user
            if (stdx::contains(_infected, user))
                _infections[user].is_infected(t - _disease.latent);
            else
                _infections[user].not_infected(t);

            // register the user
            processed.insert(user);
            complete = (n_users == processed.size());
        }
    }
}


void Infections::propagate_infection() {

    // t -> u1 -> {u21,...}
    const tms_users & encs = dworld().get_time_encounters();

    //
    //  1) for each time slot
    //      2) for each users set
    //          3) apply the contact model
    //          4) compute the aggregate infection probability
    //          5) update the infections probability for each user
    //

    // previous day
    int pd = 0;

    // 1) for each time slot 't'
    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = it->first;
        int d = t/dts;

        // if it is the new day, update the prob based on the test prob
        if (pd != d) {
            for(const user_t& u1 : dworld().users())
                update_for_newday(t, u1);
            pd = d;
        }

        // user -> {user,...}
        const ms_users& musers = it->second;

        // 2) for each user 'u1'
        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
            const user_t&  u1 = uit->first;
            const s_users& uset = uit->second;

            // 4.1) for each encounter 'u2'
            for(auto eit = uset.cbegin(); eit != uset.cend(); ++eit) {
                const user_t& u2 = *eit;

                update_for_encounter(t, u1, u2);
            }
        }
    }
}

void Infections::update_for_encounter(int t, const user_t& u1, const user_t& u2) {
    if (u1 == u2) return;

    const history_t& u2histo = _infections[u2];

    // user probability to be infected (NOT used: for debug)
    double u1p = _infections[u1].prob(t);

    // u2 infectious probability (with latency)
    double u2p   = u2histo.prob(t)*_disease.infective(u2histo, t);

    // probability to be infected
    double p = tau*ce*u2p;

    // update the u1 probability to be infected
    if (p != 0)
    _infections[u1].infected.update(t, p);
}

void Infections::update_for_newday(int t, const user_t &u1) {

    const history_t& u1histo = _infections[u1];

    // user probability to be tested (NOT usd: for debug)
    double u1t = u1histo.tested.get(t);

    // user infected probability
    double u1p = u1histo.prob(t);

    // if the user is symptomatic
    double u1s = _disease.symptomatic(u1histo, t);

    // user probability to be symptomatic
    double p = u1p*sp*u1s;

    // update the probability to be testes
    if (p != 0)
    _infections[u1].tested.update(t, p);

    // decrease the infection probability for the test probability
    double ti = _infections[u1].tested.get(t);

    _infections[u1].infected.scale(t, ti);
}


// --------------------------------------------------------------------------
// IO
//

static std::string with_ext(const std::string& filename, const std::string& ext) {
    int pos = filename.rfind('.');
    return filename.substr(0, pos) + ext;
}


void Infections::save_info(const std::string& filename) const {
    std::ofstream ofs(with_ext(filename, "_info.csv"));
    ofs << "name,value" << std::endl

        << "side,"<< dworld().side() << /*" m" <<*/ std::endl
        << "interval,"<< dworld().interval() << /*" min" <<*/ std::endl
        << "n_users,"<< dworld().users().size() << std::endl

        << "n_infected," << infected().size()  << std::endl
        << "cr_over_side," << (((double)contact_range())/dworld().side())  << std::endl
        << "beta," << beta  << std::endl
        << "tau,"<< tau  << std::endl

        << "contact_range," << contact_range() << /*" m" <<*/ std::endl
        << "infection_rate," << infection_rate() << /*" rate/day" <<*/ std::endl

        << "asymptomatic_days," << asymptomatic_days() << /*" days" <<*/ std::endl
        << "latent_days," << latent_days() << /*" days" <<*/ std::endl
        << "removed_days," << removed_days() << /*" days" <<*/ std::endl

        << "test_prob," << test_prob() << /*" days" <<*/ std::endl
        << "symptomatic_prob," << symptomatic_prob() << /*" days" <<*/ std::endl
        << "contact_efficiency," << contact_efficiency() << /*" days" <<*/ std::endl

        << "infected," << stdx::str(_infected) << std::endl;
}


void Infections::save_table(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // list of ALL users, SORTED
    const s_users& susers = dworld().users();
    std::vector<user_t> users(susers.cbegin(), susers.cend());
    sort(users.begin(), users.end());

    // header
    std::ofstream ofs(with_ext(filename, ".csv"));
    ofs << "day,timeslot";
    for (const user_t& user : users)
        ofs << "," << user << "";
    ofs << std::endl;

    // slots per day
    int dslots = max(1, (int)(interval/dworld().interval_td()));
    // end (latest) slot
    int eslot = (ptime(date(2020, 4, 1)) - ptime(date(2020,1,1)))/dworld().interval_td();

    for(int t=0; t < eslot; t += dslots) {
        int d = t/dts;

        ofs << d << "," << t;

        for (const user_t &user : users) {
            double prob = _infections.at(user).prob(t);
            ofs << stdx::format(",%.5g", prob);
        }
        ofs << std::endl;
    }

    std::cout << "Infections::done" << std::endl;
}


void Infections::save_daily(const std::string& filename, file_format format) const {
    if (format == file_format::CSV)
        save_daily_csv(filename);
    else
        save_daily_xml(filename);
}


void Infections::save_daily_csv(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.csv"));

    ofs << "day,u1,u2,index,u1_after,u1_before,u2_prob" << std::endl;

    //t -> u1 -> [(u1,u2,u1a, u1b, u2p), ...]

    // for all timeslots
    for(auto it = _daily_infections.cbegin(); it != _daily_infections.cend(); ++it) {
        int d = it->first;

        const users_encs_t& uencs = it->second;

        // for all users
        for(auto uit = uencs.cbegin(); uit != uencs.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const uencs_t& u1encs = uit->second;

            int i = 0;
            for (auto u1enc : u1encs) {
                ofs << d << "," << "," << u1 << "," << u1enc.u2 << "," << i++ << ","
                    << u1enc.u1_after << "," << u1enc.u1_before << "," << u1enc.u2_prob << ","
                    << std::endl;
            }

        }
    }

    std::cout << "Infections::done" << std::endl;
}


void Infections::save_daily_xml(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.xml"));

    //ofs << "day,user,encounter,prob" << std::endl;
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    ofs << "<infections>\n";

    // for all timeslots
    for(auto it = _daily_infections.cbegin(); it != _daily_infections.cend(); ++it) {
        int d = it->first;

        ofs << "<i d=\"" << d << "\">\n";

        const users_encs_t& uencs = it->second;

        // for all users
        for(auto uit = uencs.cbegin(); uit != uencs.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const uencs_t& u1encs = uit->second;

            ofs << "  <u1 id=\"" << u1 << "\">" << std::endl;

            int i = 0;
            for (auto u1enc : u1encs) {
                ofs << "    <u2 id=\"" << u1enc.u2 << "\" "
                    << "u1a=\""  << u1enc.u1_after  << "\" "
                    << "u1b=\"" << u1enc.u1_before << "\" "
                    << "u2p=\""   << u1enc.u2_prob   << "\" "
                    << "/>"
                    << std::endl;
            }

            ofs << "  </u1>" << std::endl;
        }

        ofs << "</i>" << std::endl;
    }

    ofs << "</infections>\n";
    std::cout << "Infections::done" << std::endl;
}
