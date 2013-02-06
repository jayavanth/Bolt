/***************************************************************************
*   Copyright 2012 Advanced Micro Devices, Inc.                                     
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

#pragma OPENCL EXTENSION cl_amd_printf : enable

/* Memory Access Patters
 * A: memory accesses of all threads are coalesced
 * B: memory accesses of threads within work-group are coalesced
 * C: memory accesses of single thread are coalesced
 *
 * L: Loop; ideal number of threads
 * S: Single; Thread / Element
 */


// 1 thread / element: 166 GB/s
template < typename iType, typename oType >
__kernel
void copy_0(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    uint gloIdx = get_global_id(0);
#if BOUNDARY_CHECK
    if (gloIdx < numElements)
#endif
        dst[gloIdx] = src[gloIdx];
};


// Single: 1 thread / element [/ BURST]
// A: all threads coalesced
template < typename iType, typename oType >
__kernel
void copy_I_SA(
    __global iType * restrict src,
    __global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0)*BURST_SIZE;

     __private iType tmp[BURST_SIZE];
    __global   iType *tSrc = &src[ begin ];
    __global   oType *tDst = &dst[ begin ];

    for ( int j = 0; j < BURST_SIZE; j++)
    {
        tmp[j] = tSrc[j];
    }
    for ( int k = 0; k < BURST_SIZE; k++)
    {
        tDst[k] = tmp[k];
    }
};

// Loop: ideal num threads
// A: all threads coalesced
template < typename iType, typename oType >
__kernel
void copy_II_LA(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0)*BURST_SIZE;
    const int incr = get_global_size(0)*BURST_SIZE;
    const int numIter = numElements / ( get_global_size(0)*BURST_SIZE );

    __private iType tmp[BURST_SIZE];
    __global  iType *tSrc = &src[ begin ];
    __global  oType *tDst = &dst[ begin ];

    for (
        int i = 0;
        i < numIter;
        i ++ )
    {
        tSrc = &src[ begin+i*incr ]; //+= incr;
        tDst = &dst[ begin+i*incr ]; //+= incr;
        
        for ( int j = 0; j < BURST_SIZE; j++)
        {
            //printf( "Thread[%07i](%02i) idx=%i\n",
            //    get_global_id(0),
            //    i, begin+i*incr+j);
            tmp[j] = tSrc[j];
        }
        for ( int k = 0; k < BURST_SIZE; k++)
        {
            tDst[k] = tmp[k];
        }
    }
};

// Single: 1 thread / element [/ BURST]
// B: threads in wg coalesced
template < typename iType, typename oType >
__kernel
void copy_III_SB(
    __global iType * restrict src,
    __global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0)*BURST_SIZE;
    
    __private iType tmp[BURST_SIZE];
    __global  iType *tSrc = &src[ begin ];
    __global  oType *tDst = &dst[ begin ];

    for ( int j = 0; j < BURST_SIZE; j++)
    {
        tmp[j] = tSrc[j];
    }
    for ( int k = 0; k < BURST_SIZE; k++)
    {
        tDst[k] = tmp[k];
    }
};

// Loop: ideal num threads
// B: threads in wg coalesced
template < typename iType, typename oType >
__kernel
void copy_IV_LB(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0)*BURST_SIZE;
    const int incr = get_global_size(0)*BURST_SIZE;
    const int numIter = numElements / ( get_global_size(0)*BURST_SIZE );
    
    __private iType tmp[BURST_SIZE];
    __global  iType *tSrc = &src[ begin ];
    __global  oType *tDst = &dst[ begin ];

    for (
        int i = 0;
        i < numIter;
        i ++ )
    {
        tSrc = &src[ begin+i*incr ]; //+= incr;
        tDst = &dst[ begin+i*incr ]; //+= incr;
        
        for ( int j = 0; j < BURST_SIZE; j++)
        {
            tmp[j] = tSrc[j];
        }
        for ( int k = 0; k < BURST_SIZE; k++)
        {
            tDst[k] = tmp[k];
        }
    }
};

// Single: 1 thread / element [/ BURST]
// C: only thread burst is coalesced
template < typename iType, typename oType >
__kernel
void copy_V_SC(
    __global iType * restrict src,
    __global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0) * ( numElements / get_global_size(0) );
    
    __private iType tmp[BURST_SIZE];
    __global  iType *tSrc = &src[ begin ];
    __global  oType *tDst = &dst[ begin ];

    for ( int j = 0; j < BURST_SIZE; j++)
    {
        tmp[j] = tSrc[j];
    }
    for ( int k = 0; k < BURST_SIZE; k++)
    {
        tDst[k] = tmp[k];
    }
};

// Loop: ideal num threads
// C: only thread burst is coalesced
template < typename iType, typename oType >
__kernel
void copy_VI_LC(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    const int begin = get_global_id(0) * ( numElements / get_global_size(0) );
    const int incr = BURST_SIZE;
    const int numIter = numElements / ( get_global_size(0)*BURST_SIZE );
    
    __private iType tmp[BURST_SIZE];
    __global  iType *tSrc = &src[ begin ];
    __global  oType *tDst = &dst[ begin ];

    for (
        int i = 0;
        i < numIter;
        i ++ )
    {
        tSrc = &src[ begin+i*incr ]; //+= incr;
        tDst = &dst[ begin+i*incr ]; //+= incr;
        
        for ( int j = 0; j < BURST_SIZE; j++)
        {
            tmp[j] = tSrc[j];
        }
        for ( int k = 0; k < BURST_SIZE; k++)
        {
            tDst[k] = tmp[k];
        }
    }
};

