/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_PROPERTY_LIST_HPP__
#define __PARAS_PROPERTY_LIST_HPP__

#include <utility>
#include <type_traits>
#include <mutex>

namespace sycl {

    namespace property {

        namespace queue {

            struct in_order {} ;
            struct enable_profiling {} ;
        }

        struct no_init {} ;

        namespace buffer {
            struct use_host_ptr {};

            class use_mutex {
                std::mutex *mutex_ptr_;
            public:
                use_mutex(std::mutex &mutexRef) : mutex_ptr_(&mutexRef) {}
                std::mutex *get_mutex_ptr() const noexcept { return mutex_ptr_; }
            };
        }

        namespace image {
            struct use_host_ptr {};
            class use_mutex {
                std::mutex *mutex_ptr_;
            public:
                use_mutex(std::mutex &mutexRef) : mutex_ptr_(&mutexRef) {}
                std::mutex *get_mutex_ptr() const noexcept { return mutex_ptr_; }
            };
        }

        namespace reduction {
            struct initialize_to_identity {};
        }
    }

    inline constexpr property::no_init no_init{};

    class context;

    namespace detail {

        template <typename T> struct is_property : std::false_type {};
        template <> struct is_property<property::queue::in_order> : std::true_type {};
        template <> struct is_property<property::queue::enable_profiling> : std::true_type {};
        template <> struct is_property<property::no_init> : std::true_type {};
        template <> struct is_property<property::buffer::use_host_ptr> : std::true_type {};
        template <> struct is_property<property::buffer::use_mutex> : std::true_type {};
        template <> struct is_property<property::image::use_host_ptr> : std::true_type {};
        template <> struct is_property<property::image::use_mutex> : std::true_type {};
        template <> struct is_property<property::reduction::initialize_to_identity> : std::true_type {};

        template <typename T, typename = void>
        struct has_get_context_member : std::false_type {};
        template <typename T>
        struct has_get_context_member<T, std::void_t<decltype(std::declval<T>().get_context())>>
            : std::true_type {};
    }

template <typename T>
struct is_property : detail::is_property<T> {};

template <typename T>
inline constexpr bool is_property_v = is_property<T>::value;

template <typename Property, typename SyclObject>
struct is_property_of : is_property<Property> {};

template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v = is_property_of<Property, SyclObject>::value;

class property_list {

    private:

        bool in_order_ = false ;
        bool profiling_ = false ;
        bool no_init_ = false ;
        bool use_host_ptr_ = false ;
        bool initialize_to_identity_ = false ;
        std::mutex *mutex_ptr_ = nullptr ;
        bool has_context_bound_ = false ;

        void *context_ptr_ = nullptr ;

        void store(property::queue::in_order) {
            in_order_ = true ;
        }

        void store(property::queue::enable_profiling) {
            profiling_ = true ;
        }

        void store(property::no_init) {
            no_init_ = true ;
        }

        void store(property::reduction::initialize_to_identity) {
            initialize_to_identity_ = true ;
        }

        void store(property::buffer::use_host_ptr) {
            use_host_ptr_ = true ;
        }

        void store(const property::buffer::use_mutex &prop) {
            mutex_ptr_ = prop.get_mutex_ptr() ;
        }

        template <typename T,
                  typename = std::enable_if_t<detail::has_get_context_member<T>::value>>
        void store(const T &prop) {
            has_context_bound_ = true ;
            using CtxT = decltype(prop.get_context()) ;
            context_ptr_ = static_cast<void*>(new CtxT(prop.get_context())) ;
        }

    public:

        property_list() noexcept = default;

        template <typename... Properties,
                  typename = std::enable_if_t<
                      ((detail::is_property<std::decay_t<Properties>>::value ||
                        detail::has_get_context_member<std::decay_t<Properties>>::value) && ...)>>
        property_list(Properties&&... props) : property_list() {
            (store(std::forward<Properties>(props)), ...) ;
        }

        bool has_in_order() const { return in_order_ ; }
        bool has_profiling() const { return profiling_ ; }
        bool has_no_init() const { return no_init_ ; }

        template <typename PropT>
        bool has_property() const noexcept {
            if constexpr (std::is_same_v<PropT, property::queue::in_order>) {
                return in_order_;
            } else if constexpr (std::is_same_v<PropT, property::queue::enable_profiling>) {
                return profiling_;
            } else if constexpr (std::is_same_v<PropT, property::no_init>) {
                return no_init_;
            } else if constexpr (std::is_same_v<PropT, property::reduction::initialize_to_identity>) {
                return initialize_to_identity_;
            } else if constexpr (std::is_same_v<PropT, property::buffer::use_host_ptr>) {
                return use_host_ptr_;
            } else if constexpr (std::is_same_v<PropT, property::buffer::use_mutex>) {
                return mutex_ptr_ != nullptr;
            } else if constexpr (detail::has_get_context_member<PropT>::value) {
                return has_context_bound_;
            } else {
                return false;
            }
        }

        template <typename PropT>
        PropT get_property() const {
            if constexpr (std::is_same_v<PropT, property::buffer::use_mutex>) {
                return property::buffer::use_mutex(*mutex_ptr_);
            } else if constexpr (detail::has_get_context_member<PropT>::value) {
                using CtxT = decltype(std::declval<PropT>().get_context()) ;
                return PropT(*static_cast<CtxT*>(context_ptr_)) ;
            } else {
                return PropT{};
            }
        }

};

}

#endif
