
#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "Maybe.hpp"

using namespace std;

template <typename T> using uptr = unique_ptr<T>;

template <typename T> using sptr = shared_ptr<T>;

template <typename T> using wptr = weak_ptr<T>;

template <typename T> inline shared_ptr<T> u2sptr(unique_ptr<T> &rhs) {
  return shared_ptr<T>(move(rhs));
}

template <typename T> inline shared_ptr<T> u2sptr(unique_ptr<T> &&rhs) {
  return shared_ptr<T>(move(rhs));
}

template <class Container, class Function> Function for_each(Container &container, Function fn) {
  return for_each(container.begin(), container.end(), fn);
}

template <typename Container, class UnaryPredicate>
Maybe<typename Container::value_type> find_if(Container &container, UnaryPredicate fn) {
  auto it = find_if(container.begin(), container.end(), fn);

  if (it == container.end()) {
    return Maybe<typename Container::value_type>::none;
  } else {
    return Maybe<typename Container::value_type>(*it);
  }
}

// This is for C++11 only, C++14 has make_unique defined in the STL
namespace std {
template <class T> struct _Unique_if { typedef unique_ptr<T> _Single_object; };

template <class T> struct _Unique_if<T[]> { typedef unique_ptr<T[]> _Unknown_bound; };

template <class T, size_t N> struct _Unique_if<T[N]> { typedef void _Known_bound; };

template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args &&... args) {
  return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T> typename _Unique_if<T>::_Unknown_bound make_unique(size_t n) {
  typedef typename remove_extent<T>::type U;
  return unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args &&...) = delete;
}
