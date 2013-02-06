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
//#define USE_AMD_HSA 1
//#define BURST_SIZE 4
//#define EXCLUSIVE 0
//#define NUM_ELEMENTS 1024*1024
//#define WG_SIZE 256
//#define LOG2_WG_SIZE 8
//#define NUM_WG 280
//#define NUM_ELEMENTS_PER_WG NUM_ELEMENTS/NUM_WG // on cpu
//#define NUM_ELEMENTS_PER_ITER WG_SIZE*BURST_SIZE
//#define NUM_BLOCK_ITER NUM_ELEMENTS_PER_WG/NUM_ELEMENTS_PER_ITER
#define USE_SCAN_20 1
#if USE_SCAN_20

/******************************************************************************
 *  Fast Kernel A
 *****************************************************************************/
template< typename iType, typename oType, typename initType, typename BinaryFunction >
kernel void scan_I_A(
    global  oType           *output, // don't access
    global  iType           *input,
            initType         init,
    local   oType           *lds,
    global  BinaryFunction  *binaryOp,
    global  oType           *intermediateScanArray )
{
#define gloId get_global_id( 0 )
#define groId get_group_id( 0 )
#define locId get_local_id( 0 )

    const int wgBegin = groId*NUM_ELEMENTS_PER_WG;
    __local oType prevScan;
    prevScan = input[0];

    // block iterations
    for (int blockIter = 0; blockIter < NUM_BLOCK_ITER; blockIter++ )
    {
        int elemAddr = wgBegin+blockIter*NUM_ELEMENTS_PER_ITER+get_local_id(0)*BURST_SIZE;
        oType sum = input[elemAddr+0];
#if BURST_SIZE>1
        oType in1 = input[elemAddr+1];
#endif
#if BURST_SIZE>2
        oType in2 = input[elemAddr+2];
        oType in3 = input[elemAddr+3];
#endif
#if BURST_SIZE>4
        oType in4 = input[elemAddr+4];
        oType in5 = input[elemAddr+5];
        oType in6 = input[elemAddr+6];
        oType in7 = input[elemAddr+7];
#endif
#if BURST_SIZE>1
        sum = (*binaryOp)( sum, in1 );
#endif
#if BURST_SIZE>2
        sum = (*binaryOp)( sum, in2 );
        sum = (*binaryOp)( sum, in3 );
#endif
#if BURST_SIZE>4
        sum = (*binaryOp)( sum, in4 );
        sum = (*binaryOp)( sum, in5 );
        sum = (*binaryOp)( sum, in6 );
        sum = (*binaryOp)( sum, in7 );
#endif
        //  Computes a scan within a workgroup
        int offset = 1;
        for( int locRedIter = 0; locRedIter < LOG2_WG_SIZE; locRedIter++ )
        {
            lds[ locId ] = sum;
            barrier( CLK_LOCAL_MEM_FENCE );
            if (locId >= offset)
            {
                oType tmp = lds[ locId - offset ];
                sum = (*binaryOp)( sum, tmp );
            }
            barrier( CLK_LOCAL_MEM_FENCE );
            offset *= 2;
        }
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            oType tmp = prevScan;
            prevScan = (blockIter==0) ? sum : (*binaryOp)(tmp, sum);
        }

        
        // write out scan value for each wg
        // this will be faster than writing value for each
        // thread (since we have suboptimal access pattern in this kernel
        // then next kernel can have ideal access pattern
        // write to tmp array here
#if 0
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            intermediateScanArray[blockIter*NUM_WG+groId] = prevScan;
        }
#endif
#if 0
        intermediateScanArray[ elemAddr ] = sum;
#endif
    }
#if 1
    if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
    {
        intermediateScanArray[groId] = prevScan;
    }
#endif
}

/******************************************************************************
 *  Fast Kernel B
 *****************************************************************************/
template< typename iType, typename oType, typename initType, typename BinaryFunction >
kernel void scan_I_B(
    global  oType           *output, // don't access
    global  iType           *input,
            initType         init,
    local   oType           *lds,
    global  BinaryFunction  *binaryOp,
    global  oType           *intermediateScanArray )
{
#define gloId get_global_id( 0 )
#define groId get_group_id( 0 )
#define locId get_local_id( 0 )

    const int wgBegin = groId*NUM_ELEMENTS_PER_WG;
    __local oType prevScan;

    // global reduction
    int globElemAddr = locId*BURST_SIZE;
    oType gloSum = intermediateScanArray[ globElemAddr+0 ];
    oType gloIn1 = intermediateScanArray[ globElemAddr+1 ];
    oType gloIn2 = intermediateScanArray[ globElemAddr+2 ];
    oType gloIn3 = intermediateScanArray[ globElemAddr+3 ];
    gloSum = (*binaryOp)( gloSum, gloIn1 );
    gloSum = (*binaryOp)( gloSum, gloIn2 );
    gloSum = (*binaryOp)( gloSum, gloIn3 );

    int offset = 1;
    for( int gloRedIter = 0; gloRedIter < LOG2_WG_SIZE; gloRedIter++ )
    {
        lds[ locId ] = sum;
        barrier( CLK_LOCAL_MEM_FENCE );
        if (locId >= offset)
        {
            oType tmp = lds[ locId - offset ];
            sum = (*binaryOp)( sum, tmp );
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        offset *= 2;
    }
    
    if ( gloId > 0 )
    {
        int prefixThread = (gloId-1)/BURST_SIZE;
        int prefixThreadReg = (gloId-1)%BURST_SIZE;
        if ( locId==prefixThread )
        {
            preScan = prefixThread==0 ? gloSum : prefixThread==1 ? gloIn1 : prefixThread==2 ? gloIn2 : gloIn3;
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );



    // block iterations
    for (int blockIter = 0; blockIter < NUM_BLOCK_ITER; blockIter++ )
    {
        int elemAddr = wgBegin+blockIter*NUM_ELEMENTS_PER_ITER+get_local_id(0)*1 /*BURST_SIZE*/;
        oType sum = input[elemAddr];

        // perpetuate prefix
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            oType tmp = prevScan;
            prevScan = (blockIter==0) ? sum : (*binaryOp)(tmp, sum);
        }

        //  Computes a scan within a workgroup
        int offset = 1;
        for( int locRedIter = 0; locRedIter < LOG2_WG_SIZE; locRedIter++ )
        {
            lds[ locId ] = sum;
            barrier( CLK_LOCAL_MEM_FENCE );
            if (locId >= offset)
            {
                oType tmp = lds[ locId - offset ];
                sum = (*binaryOp)( sum, tmp );
            }
            barrier( CLK_LOCAL_MEM_FENCE );
            offset *= 2;
        }

        // write final value
        output[ elemAddr ] = sum;

        // perpetuate prefix
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            oType tmp = prevScan;
            prevScan = (blockIter==0) ? sum : (*binaryOp)(tmp, sum);
        }
    } // block iter
} // kernel


template< typename iType, typename oType, typename initType, typename BinaryFunction >
kernel void scan_II_A(
    global  oType           *output,
    global  iType           *input,
            initType         init,
    local   oType           *lds,
    global  BinaryFunction  *binaryOp,
    global  oType           *intermediateScanArray )
{
#define gloId get_global_id( 0 )
#define groId get_group_id( 0 )
#define locId get_local_id( 0 )

    const uint wgBegin = groId*NUM_ELEMENTS_PER_WG;
    __local oType prevScan;
    prevScan = input[0];

    // block iterations
    for (int blockIter = 0; blockIter < NUM_BLOCK_ITER; blockIter++ )
    {
        int elemAddr = wgBegin+blockIter*NUM_ELEMENTS_PER_ITER+get_local_id(0)*BURST_SIZE;
        oType in0 = input[elemAddr+0];
        oType in1 = input[elemAddr+1];
        oType sum = (*binaryOp)( in0, in1 );
        //  Computes a scan within a workgroup
        int offset = 1;
        for( int locRedIter = 0; locRedIter < LOG2_WG_SIZE; locRedIter++ )
        {
            lds[ locId ] = sum;
            barrier( CLK_LOCAL_MEM_FENCE );
            if (locId >= offset)
            {
                oType tmp = lds[ locId - offset ];
                sum = (*binaryOp)( sum, tmp );
            }
            barrier( CLK_LOCAL_MEM_FENCE );
            offset *= 2;
        }
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            oType tmp = prevScan;
            prevScan = (blockIter==0) ? sum : (*binaryOp)(tmp, sum);
        }

        
        // write out scan value for each wg
        // this will be faster than writing value for each
        // thread (since we have suboptimal access pattern in this kernel
        // then next kernel can have ideal access pattern
        // write to tmp array here
#if 0
        if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
        {
            intermediateScanArray[blockIter*NUM_WG+groId] = prevScan;
        }
#endif
#if 1
        intermediateScanArray[ elemAddr ] = sum;
#endif
    }
#if 0
    if (locId == (WG_SIZE-1) ) // last thread in wg stores final scan value
    {
        intermediateScanArray[groId] = prevScan;
    }
#endif
}


#else




/******************************************************************************
 *  Not Using HSA
 *****************************************************************************/

#define NUM_ITER 16
#define MIN(X,Y) X<Y?X:Y;
#define MAX(X,Y) X>Y?X:Y;
/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
template< typename Type, typename BinaryFunction >
kernel void perBlockAddition( 
    global Type* output,
    global Type* postSumArray,
    const uint vecSize,
    global BinaryFunction* binaryOp )
{
    BinaryFunction bf = *binaryOp;
// 1 thread per element
#if 1
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );

    //  Abort threads that are passed the end of the input vector
    if( gloId >= vecSize )
        return;
        
    Type scanResult = output[ gloId ];

    // accumulate prefix
    if (groId > 0)
    {
        Type postBlockSum = postSumArray[ groId-1 ];
        Type newResult = bf( scanResult, postBlockSum );
        output[ gloId ] = newResult;
    }
#endif

// section increments
#if 0
    size_t wgSize  = get_local_size( 0 );
    size_t wgIdx   = get_group_id( 0 );
    size_t locId   = get_local_id( 0 );
    size_t secSize = wgSize / NUM_ITER; // threads per section (256 element)
    
    size_t secIdx  = NUM_ITER*wgIdx+locId/secSize;
    size_t elemIdx = secIdx*wgSize+locId%secSize;
    Type postBlockSum;
    if (secIdx==0) return;
    postBlockSum = postSumArray[ secIdx-1 ];
    
    size_t maxElemIdx = MIN( vecSize, elemIdx+secSize*NUM_ITER );
    for ( ; elemIdx<maxElemIdx; elemIdx+=secSize)
    {
        Type scanResult   = output[ elemIdx ];
        Type newResult    = (*binaryOp)( scanResult, postBlockSum );
        output[ elemIdx ] = newResult;
    }

#endif

// work-group increments
#if 0
    size_t wgSize  = get_local_size( 0 );
    size_t wgIdx   = get_group_id( 0 );
    size_t locId   = get_local_id( 0 );
    size_t secSize = wgSize;
    
    size_t secIdx  = NUM_ITER*wgIdx; // to be incremented
    size_t elemIdx = secIdx*secSize+locId;
    //Type postBlockSum;
    //if (secIdx==0) return;
    
    size_t maxSecIdx = secIdx+NUM_ITER;
    size_t maxElemIdx = MIN( vecSize, elemIdx+secSize*NUM_ITER);
    // secIdx = MAX( 1, secIdx);
    if (wgIdx==0) // skip
    {
        secIdx++;
        elemIdx+=secSize;
    }
    for ( ; elemIdx<maxElemIdx; secIdx++, elemIdx+=secSize)
    {
        Type postBlockSum = postSumArray[ secIdx-1 ];
        Type scanResult   = output[ elemIdx ];
        Type newResult    = (*binaryOp)( scanResult, postBlockSum );
        output[ elemIdx ] = newResult;
    }
#endif

// threads increment
#if 0

    size_t wgSize  = get_local_size( 0 );
    size_t wgIdx   = get_group_id( 0 );
    size_t locId   = get_local_id( 0 );
    size_t threadsPerSect = wgSize / NUM_ITER;
    
    size_t secIdx  = NUM_ITER*wgIdx;
    size_t elemIdx = secIdx*wgSize+locId*NUM_ITER;

    if (elemIdx < vecSize && secIdx > 0)
    {
        Type scanResult   = output[ elemIdx ];
        Type postBlockSum = postSumArray[ secIdx-1 ];
        Type newResult    = (*binaryOp)( scanResult, postBlockSum );
        output[ elemIdx ] = newResult;
    }
    //secIdx++;
    elemIdx++; //=wgSize;

    for (size_t i = 1; i < NUM_ITER-1; i++)
    {
      Type scanResult   = output[ elemIdx ];
      Type postBlockSum = postSumArray[ secIdx-1 ];
      Type newResult    = (*binaryOp)( scanResult, postBlockSum );
      output[ elemIdx ] = newResult;
      
      // secIdx++;
      elemIdx++; //=wgSize;
    }

    if (elemIdx < vecSize /*&& groId > 0*/)
    {
        Type scanResult = output[ elemIdx ];
        Type postBlockSum = postSumArray[ secIdx-1 ];
        Type newResult = (*binaryOp)( scanResult, postBlockSum );
        output[ elemIdx ] = newResult;
    }

#endif
}


/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
template< typename Type, typename BinaryFunction >
kernel void intraBlockInclusiveScan(
                global Type* postSumArray,
                global Type* preSumArray, 
                Type identity,
                const uint vecSize,
                local Type* lds,
                const uint workPerThread,
                global BinaryFunction* binaryOp
                )
{
    BinaryFunction bf = *binaryOp;
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    // do offset of zero manually
    uint offset;
    Type workSum;
    if (mapId < vecSize)
    {
        // accumulate zeroth value manually
        offset = 0;
        workSum = preSumArray[mapId+offset];
        postSumArray[ mapId + offset ] = workSum;

        //  Serial accumulation
        for( offset = offset+1; offset < workPerThread; offset += 1 )
        {
            if (mapId+offset<vecSize)
            {
                Type y = preSumArray[mapId+offset];
                workSum = bf( workSum, y );
                postSumArray[ mapId + offset ] = workSum;
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    Type scanSum;
    offset = 1;
    // load LDS with register sums
    if (mapId < vecSize)
    {
        lds[ locId ] = workSum;
        barrier( CLK_LOCAL_MEM_FENCE );
    
        if (locId >= offset)
        { // thread > 0
            Type y = lds[ locId - offset ];
            Type y2 = lds[ locId ];
            scanSum = bf( y2, y );
            lds[ locId ] = scanSum;
        } else { // thread 0
            scanSum = workSum;
        }  
    }
    // scan in lds
    for( offset = offset*2; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < vecSize)
        {
            if (locId >= offset)
            {
                Type y = lds[ locId - offset ];
                scanSum = bf( scanSum, y );
                lds[ locId ] = scanSum;
            }
        }
    } // for offset
    barrier( CLK_LOCAL_MEM_FENCE );
    
    // write final scan from pre-scan and lds scan
    for( offset = 0; offset < workPerThread; offset += 1 )
    {
        barrier( CLK_GLOBAL_MEM_FENCE );

        if (mapId < vecSize && locId > 0)
        {
            Type y = postSumArray[ mapId + offset ];
            Type y2 = lds[locId-1];
            y = bf( y, y2 );
            postSumArray[ mapId + offset ] = y;
        } // thread in bounds
    } // for 
} // end kernel


/******************************************************************************
 *  Kernel 0
 *****************************************************************************/
template< typename iType, typename oType, typename initType, typename BinaryFunction >
kernel void perBlockInclusiveScan(
                global oType* output,
                global iType* input,
                initType identity,
                const uint vecSize,
                local oType* lds,
                global BinaryFunction* binaryOp,
                global oType* scanBuffer,
                int exclusive) // do exclusive scan ?
{
    BinaryFunction bf = *binaryOp;
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return; // on SI this doesn't mess-up barriers

    // if exclusive, load gloId=0 w/ identity, and all others shifted-1
    oType val;
    if (exclusive)
    {
        if (gloId > 0)
        { // thread>0
            val = input[gloId-1];
            lds[ locId ] = val;
        }
        else
        { // thread=0
            val = identity;
            lds[ locId ] = val;
        }
    }
    else
    {
        val = input[gloId];
        lds[ locId ] = val;
    }

    //  Computes a scan within a workgroup
    oType sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (locId >= offset)
        {
            oType y = lds[ locId - offset ];
            sum = bf( sum, y );
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        lds[ locId ] = sum;
    }

    //  Each work item writes out its calculated scan result, relative to the beginning
    //  of each work group
    output[ gloId ] = sum;
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types
    if (locId == 0)
    {
        // last work-group can be wrong b/c ignored
        scanBuffer[ groId ] = lds[ wgSize-1 ];
    }
}

#endif