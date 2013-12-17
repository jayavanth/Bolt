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
#if !defined( BOLT_CL_COUNTING_ITERATOR_H )
#define BOLT_CL_COUNTING_ITERATOR_H
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include <boost/iterator/iterator_facade.hpp>

/*! \file bolt/cl/iterator/counting_iterator.h
    \brief Return Same Value or counting Value on dereferencing.
*/


namespace bolt {
namespace amp {

    struct counting_iterator_tag
        : public fancy_iterator_tag
        {   // identifying tag for random-access iterators
        };

        template< typename value_type >
        class counting_iterator: public boost::iterator_facade< counting_iterator< value_type >, value_type,
          counting_iterator_tag, value_type, int >
        {
        public:
             typedef typename boost::iterator_facade< counting_iterator< value_type >, value_type, 
             counting_iterator_tag, value_type, int >::difference_type difference_type;
             typedef concurrency::array_view< value_type > arrayview_type;

             typedef counting_iterator<value_type> const_iterator;
           

            //  Basic constructor requires a reference to the container and a positional element
            counting_iterator( value_type init, const control& ctl = control::getDefault( ) ): 
                m_initValue( init ), m_Index( 0 ) {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
           template< typename OtherType >
           counting_iterator( const counting_iterator< OtherType >& rhs ):m_Index( rhs.m_Index ),
               m_initstValue( rhs.m_initstValue ) {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            counting_iterator< value_type >& operator= ( const counting_iterator< value_type >& rhs )
            {
                if( this == &rhs )
                    return *this;

                m_initstValue = rhs.m_initstValue;
                m_Index = rhs.m_Index;
                return *this;
            }
                
            counting_iterator< value_type >& operator+= ( const  difference_type & n )
            {
                advance( n );
                return *this;
            }
                
            const counting_iterator< value_type > operator+ ( const difference_type & n ) const
            {
                counting_iterator< value_type > result( *this );
                result.advance( n );
                return result;
            }

            const counting_iterator< value_type > & getBuffer( const_iterator itr ) const
            {
                return *this;
            }
            

            const counting_iterator< value_type > & getContainer( ) const
            {
                return *this;
            }

            difference_type distance_to( const counting_iterator< value_type >& rhs ) const
            {
                return rhs.m_Index - m_Index;
            }

            //  Public member variables
            difference_type m_Index;

       // private:
            //  Implementation detail of boost.iterator
            friend class boost::iterator_core_access;

            //  Used for templatized copy constructor and the templatized equal operator
            template < typename > friend class counting_iterator;

            //  For a counting_iterator, do nothing on an advance
            void advance( difference_type n )
            {
                m_Index += n;
            }

            void increment( )
            {
                advance( 1 );
            }

            void decrement( )
            {
                advance( -1 );
            }

            difference_type getIndex() const
            {
                return m_Index;
            }

            template< typename OtherType >
            bool equal( const counting_iterator< OtherType >& rhs ) const
            {
                bool sameIndex = (rhs.m_initValue == m_initValue) && (rhs.m_Index == m_Index);

                return sameIndex;
            }



            typename boost::iterator_facade< counting_iterator< value_type >, value_type, 
            counting_iterator_tag, value_type, int >::reference dereference( ) const
            {
              return m_initValue + m_Index;

            }


            int operator[](int x) const restrict(cpu,amp) // Uncomment if using iterator in inl
            {
              int temp = x + m_initValue;
              return temp;
            }


            value_type m_initValue;
        };
    //)


    template< typename Type >
    counting_iterator< Type > make_counting_iterator( Type constValue )
    {
        counting_iterator< Type > tmp( constValue );
        return tmp;
    }

}
}


#endif
