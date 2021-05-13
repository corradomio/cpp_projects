#include <iostream>
#include <string>

using namespace std;

struct matrix {
    virtual ~matrix() {}
    // ...
};

struct dense_matrix    : matrix { /* ... */ };
struct diagonal_matrix : matrix { /* ... */ };

#include <yorel/yomm2/cute.hpp>

using yorel::yomm2::virtual_;

register_class(matrix);
register_class(dense_matrix, matrix);
register_class(diagonal_matrix, matrix);

declare_method(string, to_json, (virtual_<const matrix&>));

define_method(string, to_json, (const dense_matrix& m)) {
return "json for dense matrix...";
}

define_method(string, to_json, (const diagonal_matrix& m)) {
return "json for diagonal matrix...";
}

int main() {
    yorel::yomm2::update_methods();

    shared_ptr<const matrix> a = make_shared<dense_matrix>();
    shared_ptr<const matrix> b = make_shared<diagonal_matrix>();

    cout << to_json(*a) << "\n"; // json for dense matrix
    cout << to_json(*b) << "\n"; // json for diagonal matrix

    return 0;
}