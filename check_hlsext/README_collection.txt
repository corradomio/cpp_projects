https://gist.github.com/jeetsukumaran/307264



template <
    typename Key,
    typename T,
    class Hash = hash<Key>,
    class Pred = equal_to<Key>,
    class AllocH = allocator< pair<const Key,T> >,  // unordered_map allocator
    class AllocV = allocator<T> > // vector allocator
class hash_map {
public:
  typedef std::pair<Key, T> value_type;

private:
  using first_index = std::vector<T>; // C++11 typedef substitute syntax
  using second_index = std::unordered_map<Key, T>;

public:
  using first_index_iterator = typename first_index::iterator;
  using second_index_iterator = typename second_index::iterator;
  //defaults
  using iterator = first_index_iterator;
  using const_iterator = first_index_const_iterator;

  iterator begin();
  const_iterator begin() const;

  iterator end();
  const_iterator end() const;

  bool empty() const;

  iterator find(const Key&);
  const_iterator find(const Key&) const;

  std::pair<iterator, bool> insert(const value_type&);
  void erase(iterator);
  void clear();
};

You are not supposed to add your collections to the std namespace, however if you proceed nevertheless, I strongly suggest you use namespace versioning when you publish your library headers:

// hash_map.hpp
namespace std
{
  namespace ex_v1  // my std namespace extensions v1
  {
      template <...> class hash_map { ... }
  }

  using namespace ex_v1;
}