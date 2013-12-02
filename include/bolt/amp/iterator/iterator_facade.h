/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

#pragma once
#if !defined( BOLT_AMP_ITERATOR_FACADE_H )
#define BOLT_AMP_ITERATOR_FACADE_H

/*! \file bolt/amp/iterator/iterator_facade.h
    \brief 
*/

namespace bolt {
namespace amp {

    template <
        class Derived             // The derived iterator type being constructed
            , class Value
            , class CategoryOrTraversal
            , class Reference = Value&
            , class Difference = std::ptrdiff_t
    >
    class iterator_facade
# ifdef BOOST_ITERATOR_FACADE_NEEDS_ITERATOR_BASE
        : public boost::detail::iterator_facade_types<
        Value, CategoryOrTraversal, Reference, Difference
        >::base
#  undef BOOST_ITERATOR_FACADE_NEEDS_ITERATOR_BASE
# endif
    {
    private:
        //
        // Curiously Recurring Template interface.
        //
        Derived& derived()
        {
            return *static_cast<Derived*>(this);
        }

        Derived const& derived() const
        {
            return *static_cast<Derived const*>(this);
        }

        typedef boost::detail::iterator_facade_types<
            Value, CategoryOrTraversal, Reference, Difference
        > associated_types;

        typedef boost::detail::operator_arrow_dispatch<
            Reference
            , typename associated_types::pointer
        > operator_arrow_dispatch_;

    protected:
        // For use by derived classes
        typedef iterator_facade<Derived, Value, CategoryOrTraversal, Reference, Difference> iterator_facade_;

    public:

        typedef typename associated_types::value_type value_type;
        typedef Reference reference;
        typedef Difference difference_type;

        typedef typename operator_arrow_dispatch_::result_type pointer;

        typedef typename associated_types::iterator_category iterator_category;

        reference operator*() const
        {
            return iterator_core_access::dereference(this->derived());
        }

        pointer operator->() const
        {
            return operator_arrow_dispatch_::apply(*this->derived());
        }

        typename boost::detail::operator_brackets_result<Derived, Value, reference>::type
            operator[](difference_type n) const
        {
                typedef boost::detail::use_operator_brackets_proxy<Value, Reference> use_proxy;

                return boost::detail::make_operator_brackets_result<Derived>(
                    this->derived() + n
                    , use_proxy()
                    );
            }

        Derived& operator++()
        {
            iterator_core_access::increment(this->derived());
            return this->derived();
        }

# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
        typename boost::detail::postfix_increment_result<Derived, Value, Reference, CategoryOrTraversal>::type
            operator++(int)
        {
                typename boost::detail::postfix_increment_result<Derived, Value, Reference, CategoryOrTraversal>::type
                    tmp(this->derived());
                ++*this;
                return tmp;
            }
# endif

        Derived& operator--()
        {
            iterator_core_access::decrement(this->derived());
            return this->derived();
        }

        Derived operator--(int)
        {
            Derived tmp(this->derived());
            --*this;
            return tmp;
        }

        Derived& operator+=(difference_type n)
        {
            iterator_core_access::advance(this->derived(), n);
            return this->derived();
        }

        Derived& operator-=(difference_type n)
        {
            iterator_core_access::advance(this->derived(), -n);
            return this->derived();
        }

        Derived operator-(difference_type x) const
        {
            Derived result(this->derived());
            return result -= x;
        }

# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
        // There appears to be a bug which trashes the data of classes
        // derived from iterator_facade when they are assigned unless we
        // define this assignment operator.  This bug is only revealed
        // (so far) in STLPort debug mode, but it's clearly a codegen
        // problem so we apply the workaround for all MSVC6.
        iterator_facade& operator=(iterator_facade const&)
        {
            return *this;
        }
# endif
    };

    

}
};

#endif