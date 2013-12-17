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
#if !defined( BOLT_CL_CONSTANT_ITERATOR_H )
#define BOLT_CL_CONSTANT_ITERATOR_H
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include <boost/iterator/iterator_facade.hpp>

/*! \file bolt/cl/iterator/constant_iterator.h
    \brief Return Same Value or Constant Value on dereferencing.
*/


namespace bolt {
namespace amp {

    struct constant_iterator_tag
        : public fancy_iterator_tag
        {   // identifying tag for random-access iterators
        };

        template< typename value_type >
        class constant_iterator: public boost::iterator_facade< constant_iterator< value_type >, value_type,
          constant_iterator_tag, value_type, int >
        {
        public:
             typedef typename boost::iterator_facade< constant_iterator< value_type >, value_type, 
             constant_iterator_tag, value_type, int >::difference_type difference_type;
             typedef concurrency::array_view< value_type > arrayview_type;

             typedef constant_iterator<value_type> const_iterator;
           

            struct Payload
            {
                value_type m_Value;
            };

            //  Basic constructor requires a reference to the container and a positional element
            constant_iterator( value_type init, const control& ctl = control::getDefault( ) ): 
                m_constValue( init ), m_Index( 0 )
            {
              //int m_Size = 1;
              //std::vector<value_type> temp(1, m_constValue);
              //concurrency::extent<1> ext( static_cast< int >( m_Size ) );
              //m_devMemory = new concurrency::array<value_type, 1>(ext, temp.begin(), ctl.getAccelerator( ).default_view );
              
            }

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
           template< typename OtherType >
           constant_iterator( const constant_iterator< OtherType >& rhs ): m_devMemory( rhs.m_devMemory ),
               m_Index( rhs.m_Index ), m_constValue( rhs.m_constValue )
           {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            constant_iterator< value_type >& operator= ( const constant_iterator< value_type >& rhs )
            {
                if( this == &rhs )
                    return *this;

                //m_devMemory = rhs.m_devMemory;
                m_constValue = rhs.m_constValue;
                m_Index = rhs.m_Index;
                return *this;
            }
                
            constant_iterator< value_type >& operator+= ( const  difference_type & n )
            {
                advance( n );
                return *this;
            }
                
            const constant_iterator< value_type > operator+ ( const difference_type & n ) const
            {
                constant_iterator< value_type > result( *this );
                result.advance( n );
                return result;
            }

            //value_type constant_iterator< value_type > operator[] (unsigned int &i) restrict(cpu, amp)
            //{
            //  return m_constValue;
            //}

            arrayview_type getBuffer() const
            {
               // return m_devMemory;
            }

            //arrayview_type getBuffer(const_iterator itr) const
            //{
            //  //difference_type offset = 1;//itr.getIndex();
            //  //concurrency::extent<1> ext( static_cast< int >( 1 /*m_Size-offset*/ ));
            //  //return m_devMemory->section( Concurrency::index<1>((int)1), ext );
            //

            //  ////return m_devMemory->reinterpret_as<value_type>();
            //  return m_constValue;
            //  
            //}



            //int getBuffer(const_iterator itr) const
            //{
            //  //difference_type offset = 1;//itr.getIndex();
            //  //concurrency::extent<1> ext( static_cast< int >( 1 /*m_Size-offset*/ ));
            //  //return m_devMemory->section( Concurrency::index<1>((int)1), ext );
            //
            //  return m_constValue;
            //  
            //}

            const constant_iterator< value_type > & getBuffer( const_iterator itr ) const
            {
                return *this;
            }
            

            const constant_iterator< value_type > & getContainer( ) const
            {
                return *this;
            }

            //Payload gpuPayload( ) const
            //{
            //    Payload payload = { m_constValue };
            //    return payload;
            //}

            //const difference_type gpuPayloadSize( ) const
            //{
            //    return sizeof( Payload );
            //}

            difference_type distance_to( const constant_iterator< value_type >& rhs ) const
            {
                //return static_cast< typename iterator_facade::difference_type >( 1 );
                return rhs.m_Index - m_Index;
            }

            //  Public member variables
            difference_type m_Index;

       // private:
            //  Implementation detail of boost.iterator
            friend class boost::iterator_core_access;

            //  Used for templatized copy constructor and the templatized equal operator
            template < typename > friend class constant_iterator;

            //  For a constant_iterator, do nothing on an advance
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
            bool equal( const constant_iterator< OtherType >& rhs ) const
            {
                bool sameIndex = (rhs.m_constValue == m_constValue) && (rhs.m_Index == m_Index);

                return sameIndex;
            }



            typename boost::iterator_facade< constant_iterator< value_type >, value_type, 
            constant_iterator_tag, value_type, int >::reference dereference( ) const
            {
              //value_type tmp = *(m_devMemory->data());
              //return tmp;

              //return *(m_devMemory->data());

              return m_constValue;

            }


            int operator[](int x) const restrict(cpu,amp) // Uncomment if using iterator in inl
            {
              int xy = m_constValue;
              return xy;
              //return *(m_Ptr->data());
            }


            //concurrency::array<value_type>* m_devMemory;
            //concurrency::array_view<value_type>* m_Ptr;
            value_type m_constValue;
        };
    //)


    template< typename Type >
    constant_iterator< Type > make_constant_iterator( Type constValue )
    {
        constant_iterator< Type > tmp( constValue );
        return tmp;
    }

}
}


#endif
