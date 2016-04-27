/*
 * Maybe.h
 *
 *  Created on: 12/12/2013
 *      Author: dan
 */

#pragma once

#include <cassert>
#include <utility>

static_assert(sizeof(char) == 1, "char isnt 1 byte");

template <typename T> class Maybe final {
  bool hasVal;
  char dummy[sizeof(T)];
  T *valPtr() const { return reinterpret_cast<T *>(const_cast<char *>(dummy)); }
  void cleanup() {
    if (valid()) {
      hasVal = false;
      valPtr()->~T();
    }
  }
  void copy_from(const Maybe &other) {
    if (other.valid()) {
      new (valPtr()) T(other.val());
      hasVal = true;
    }
  }
  void move_from(Maybe &&other) {
    if (other.valid()) {
      new (valPtr()) T(std::move(*(other.valPtr())));
      hasVal = true;
      other.hasVal = false;
    }
  }

public:
  static const Maybe none;

  Maybe() : hasVal(false) {}
  explicit Maybe(const T &val) : hasVal(true) { new (valPtr()) T(val); }
  explicit Maybe(T &&val) : hasVal(true) { // todo: noexcept if T() is noexcept
    new (valPtr()) T(std::move(val));
  }
  Maybe(const Maybe &other) : hasVal(other.hasVal) { copy_from(other); }
  Maybe(Maybe &&other) : hasVal(other.hasVal) { move_from(std::move(other)); }
  ~Maybe() { cleanup(); }

  bool valid() const { return hasVal; }

  Maybe &operator=(const Maybe &other) {
    cleanup();
    copy_from(other);
    return *this;
  }

  Maybe &operator=(Maybe &&other) {
    cleanup();
    move_from(std::move(other));
    return *this;
  }

  bool operator==(const Maybe &other) {
    return (valid() == other.valid()) && (!valid() || val() == other.val());
  }

  bool operator!=(const Maybe &other) {
    return (valid() != other.valid()) || (valid() && val() != other.val());
  }

  const T &val() const {
    assert(valid());
    return *valPtr();
  }
  T &val() {
    assert(valid());
    return *valPtr();
  }
  const T &valOr(const T &defaultVal) const { return valid() ? *valPtr() : defaultVal; }
  T &valOr(T &defaultVal) { return valid() ? *valPtr() : defaultVal; }
};

template <class T> const Maybe<T> Maybe<T>::none;
