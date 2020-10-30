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

double disease_t::infective(int t0, int t) {
    if (t0 == invalid)
        return 0.;

    int dt = t - t0;
    return latent < dt && dt <= removed;
}


double disease_t::symptomatic(int t0, int t) {
    if (t0 == invalid)
        return 0.;

    int dt = t - t0;
    return asymptomatic < dt && dt <= removed;
}

// --------------------------------------------------------------------------
// prob_hist_t
// --------------------------------------------------------------------------

double prob_hist_t::get(int t) const {
    size_t l = 0;
    size_t u = hist.size()-1;
    size_t m;
    while((u-l) > 1) {
        m = (u-l)/2;
        if (t < hist[m].t)
            u = m - 1;
        else if (t > hist[m].t)
            l = m + 1;
        else
            break;
    }
    return hist[m].prob;
}


void prob_hist_t::set(int t, double p)  {
    tprob_t& entry = hist[hist.size()-1];
    if (t == entry.t)
        entry.prob = p;
    else if (p != entry.prob)
        hist.emplace_back(t, p);
    if (p > 0 && infected == invalid)
        infected = t;
}


double prob_hist_t::update(int t, double u) {
    double p = get();
    p = 1 - (1 - p)*(1 - u);
    set(t, p);
    return p;
}


double prob_hist_t::scale(int t, double u) {
    double p = get();
    p = p*(1 - p*u);
    set(t, p);
    return p;
}



// --------------------------------------------------------------------------
// history_t
// --------------------------------------------------------------------------

double history_t::prob(int t) const {
    int when = infected.when_infected();
    if (when == invalid)
        return 0.;
    if (t < when+disease_p->latent)
        return 0;
    if (t > when+disease_p->removed)
        return 0;
    else
        return infected.get(t);
}

double history_t::symptomatic(int t) const {
    int when = infected.when_infected();
    if (when == invalid)
        return 0.;
    if (t < when+disease_p->asymptomatic)
        return 0;
    else
        return infected.get(t);
}


// --------------------------------------------------------------------------
// Infections
// --------------------------------------------------------------------------

Infections::Infections() {
    d = 2;
    beta = 0.001;
    l = 0;
    r = 0;
    s = 0;
    tp = 0.01;
    sp = 0.20;
    seed = 123;
    _cmode_day = none;
    _only_infections = true;
}


// --------------------------------------------------------------------------
// Infected users
// --------------------------------------------------------------------------

/// Quota [0,1] of infected users
Infections& Infections::infected(double quota) {
    //int n = lround(quota*dworld().users().size());
    //return infected(n);
    return infected(dworld().users(quota));
}


/// Number of infected users
Infections& Infections::infected(int n) {
    //const s_users& susers = dworld().users();
    //int nusers = susers.size();
    //
    // // all users are infected
    //if (n >= nusers)
    //    return infected(susers);
    //
    // // convert set[user] -> array[user]
    //std::vector<user_t> vusers(susers.begin(), susers.end());
    //
    // // selected infected users
    //std::unordered_set<user_t> selected;
    //
    // // loop until |selected| == n
    //while(n != selected.size()) {
    //    const user_t& user = vusers[rnd.next_int(nusers)];
    //    selected.insert(user);
    //}
    //
    //return infected(selected);
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

    // (d/D)^2
    double dratio = sq(d/D);
    // (1-exp(-beta*dt))
    double betadt = (1 - exp(-beta*dt));

    // factor of infection tau=(1-exp(-beta*dt))*(d/D)^2
    _disease.tau = betadt*dratio;
    _disease.dt = dt;

    _disease.latent = l*dts;        // latent in time slots
    _disease.asymptomatic = s*dts;  // asymptomatic in time slots
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
                _infections[user].infective(t);
            else
                _infections[user].not_infected(t);

            // register the user
            processed.insert(user);
            complete = (n_users == processed.size());
        }
    }
}


// --------------------------------------------------------------------------
// Propagate::propagate_infection
// --------------------------------------------------------------------------

const s_users & Infections::apply_contact_model(int t, const s_users& uset) {
    int tsday;
    switch (_cmode) {
        case contact_mode::none:
            return uset;
        case contact_mode::random:
            _cmode_users.clear();
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        case contact_mode::daily:
            tsday = t/dts;
            if (_cmode_day == tsday)
                return _cmode_users;
            _cmode_day = tsday;
            _cmode_users.clear();
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        case contact_mode::user:
            if (_cmode_day == 0)
                return _cmode_users;
            _cmode_day = 0;
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        default:
            return uset;
    }
}


void Infections::propagate_infection() {

    double tau = _disease.tau;

    // t -> user -> {user,...}
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
            for(const user_t& user : dworld().users())
                update_for_newday(t, user);
            pd = d;
        }

        // user -> {user,...}
        const ms_users& musers = it->second;

        // 2) for each user 'u1'
        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
            const user_t&  u1 = uit->first;
            const s_users& uset = uit->second;

            // 3) apply the contact model
            s_users users = apply_contact_model(t, uset);

            // 4.1) for each encounter 'u2'
            for(auto eit = uset.cbegin(); eit != uset.cend(); ++eit) {
                const user_t& u2 = *eit;

                update_for_encounter(t, u1, u2);
            }
        }
    }
}

void Infections::update_for_encounter(int t, const user_t& u1, const user_t& u2){
    if (u1 == u2) return;

    double tau = _disease.tau;

    // u1 infection probability BEFORE the encounter
    double u1_before = _infections[u1].prob();
    // u2 infectious probability
    double u2_prob   = _infections[u2].prob(t);

    // probability to be infected
    double p = tau*u2_prob;

    // u1 infection probability AFTER the encounter
    double u1_after  = _infections[u1].infected.update(t, p);

    // save the encounter
    if (((!_only_infections) || (u1_after != u1_before) || (u2_prob != 0.)) && false)
        _daily_infections[d][u1].emplace_back(u1, u2, u1_after, u1_before, u2_prob);
}

void Infections::update_for_newday(int t, const user_t &u1) {
    // user infection probability
    double up = _infections[u1].prob();

    // user probability to be symptomatic
    double sp = _infections[user].symptomatic(t);

    // probability to have symptoms
    double p = symptomatic_prob()*up*sp;

    // update the probability to be testes
    _infections[user].tested.update(t, p);

    // decrease the infection probability for the test probability
    double tp = _infections[user].tested.get(t);
    _infections[user].infected.scale(t, tp);
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
        << "contact_range," << contact_range() << /*" m" <<*/ std::endl
        << "infection_rate," << infection_rate() << /*" rate/day" <<*/ std::endl
        << "cr_over_s," << (((double)contact_range())/dworld().side())  << std::endl
        << "beta," << beta  << std::endl
        << "dt," << _disease.dt  << std::endl
        << "tau,"<< _disease.tau  << std::endl
        << "latent_days," << latent_days() << /*" days" <<*/ std::endl
        << "removed_days," << removed_days() << /*" days" <<*/ std::endl
        << "n_infected," << _infected.size()  << std::endl
        << "infected," << stdx::str(_infected) << std::endl;
}


void Infections::save_table(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // header
    std::ofstream ofs(with_ext(filename, ".csv"));
    ofs << "day,timeslot";
    for (const user_t& user : dworld().users())
        ofs << "," << user << "";
    ofs << std::endl;

    // slots per day
    int dslots = max(1, (int)(interval/dworld().interval_td()));
    // end (latest) slot
    int eslot = (ptime(date(2020, 4, 1)) - ptime(date(2020,1,1)))/dworld().interval_td();

    for(int t=0; t < eslot; t += dslots) {
        int d = t/dts;

        ofs << d << "," << t;

        for (const user_t &user : dworld().users()) {
            double prob = _infections.at(user).prob(t);
            ofs << stdx::format(",%.5g", prob);
        }
        ofs << std::endl;
    }

    std::cout << "Infections::done" << std::endl;
}


//static void collect_daily(tmb_users& daily_encs, const tms_users& encs, int dts) {
//
//    // 1) for each time slot
//    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
//        int d = (it->first)/dts;
//        const ms_users& musers = it->second;
//
//        // 2) for each (user -> encounters)
//        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
//            const user_t& user = uit->first;
//            const s_users& users = uit->second;
//
//            daily_encs[d][user].insert(users.cbegin(), users.cend());
//        }
//    }
//}

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
