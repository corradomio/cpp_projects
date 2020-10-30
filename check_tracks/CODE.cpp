
const int invalid = -9999;

/**
 * Infection state of each user.
 * 'infected' is used to mark the time slot for the first infection
 *
 * It is necessary to use a special trick for the users marked as 'infected'
 * because, at the begin, the corrected 'infected' timeslot is unknown.
 * The trick consists in to update 'infected' the FIRST time that the
 * infection probability is updated. In this case, the correct infected timeslot
 * is
 *
 *      t - latent_days_ts
 *
 * This number can be negative but this is not a problem.
 */
class state_t {
    int _lts;       // latent time slots
    int _rts;       // removed time slots
    int _infected;  // time slot when received the infection
    double _prob;
    public:
    state_t(){ }

    void set(int lts, int rts);

    // initial infected user (prob = 1)
    void infective(int t);
    void not_infected(int t);

    int infected() const { return _infected; }

    // get & set & update
    double prob(int t) const;
    double prob() const { return _prob; }

    // update the probability as p' = 1 - (1-p)(1-u)
    double update(int t, double u);

    // update the probability as: p' = p*(1-r*p)
    double daily(int t, double r);
};




// --------------------------------------------------------------------------
// state_t
// --------------------------------------------------------------------------

void state_t::set(int lts, int rts) {
    _infected = invalid;
    _prob = 0;
    _lts = lts;
    _rts = rts;
}


void state_t::infective(int t) {
    _infected = t - _lts;
    _prob = 1.;
}


void state_t::not_infected(int t) {
    _infected = invalid;
    _prob = 0.;
}


double state_t::prob(int t) const {
    // not infected
    if (_infected == invalid)
        return 0.;
    // latent period
    if (t < _infected + _lts)
        return 0.;
    // infective period
    if (t <= _infected + _rts)
        return _prob;
        // removed
    else
        return 0.;
}


double state_t::update(int t, double u) {
    if(u != 0.)
        _prob = 1. - (1. - _prob)*(1. - u);

    if (_prob != 0 && _infected == invalid)
        _infected = t;
    return _prob;
}


double state_t::daily(int t, double r) {
    if (_prob != 0.)
        _prob = _prob*(1 - r*_prob);
    return _prob;
}


