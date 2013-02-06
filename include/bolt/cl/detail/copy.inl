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

#if !defined( COPY_INL )
#define COPY_INL
#pragma once

#ifndef BURST_SIZE
#define BURST_SIZE 1
#endif

#include <boost/thread/once.hpp>
#include <boost/bind.hpp>
#include <type_traits>

#include "bolt/cl/bolt.h"

#ifdef BOLT_PROFILER_ENABLED
#define BOLT_PROFILER_START_COPY_TRIAL \
    aProfiler.setName("Copy"); \
    aProfiler.startTrial(); \
    aProfiler.setStepName("Origin"); \
    aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#define BOLT_PROFILER_STOP_COPY_TRIAL \
    aProfiler.stopTrial();
#else
#define BOLT_PROFILER_START_COPY_TRIAL
#define BOLT_PROFILER_STOP_COPY_TRIAL
#endif

// bumps dividend up (if needed) to be evenly divisible by divisor
// returns whether dividend changed
// makeDivisible(9,4) -> 12,true
// makeDivisible(9,3) -> 9, false
template< typename Type1, typename Type2 >
bool makeDivisible( Type1& dividend, Type2 divisor)
{
    size_t lowerBits = static_cast<size_t>( dividend & (divisor-1) );
    if( lowerBits )
    { // bump it up
        dividend &= ~lowerBits;
        dividend += divisor;
        return true;
    }
    else
    { // already evenly divisible
      return false;
    }
}

// bumps dividend up (if needed) to be evenly divisible by divisor
// returns whether dividend changed
// roundUpDivide(9,4,?)  -> 12,4,3,true
// roundUpDivide(10,2,?) -> 10,2,5,false
template< typename Type1, typename Type2, typename Type3 >
bool roundUpDivide( Type1& dividend, Type2 divisor, Type3& quotient)
{
    size_t lowerBits = static_cast<size_t>( dividend & (divisor-1) );
    if( lowerBits )
    { // bump it up
        dividend &= ~lowerBits;
        dividend += divisor;
        quotient = dividend / divisor;
        return true;
    }
    else
    { // already evenly divisible
      quotient = dividend / divisor;
      return false;
    }
}

namespace bolt {
namespace cl {

// user control
template<typename InputIterator, typename OutputIterator> 
OutputIterator copy(const bolt::cl::control &ctrl,  InputIterator first, InputIterator last, OutputIterator result, 
    const std::string& user_code)
{
    int n = static_cast<int>( std::distance( first, last ) );
BOLT_PROFILER_START_COPY_TRIAL
    OutputIterator rtrn = detail::copy_detect_random_access( ctrl, first, n, result, user_code, std::iterator_traits< InputIterator >::iterator_category( ) );
BOLT_PROFILER_STOP_COPY_TRIAL
    return rtrn;
}

// default control
template<typename InputIterator, typename OutputIterator> 
OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result, 
    const std::string& user_code)
{
    int n = static_cast<int>( std::distance( first, last ) );
BOLT_PROFILER_START_COPY_TRIAL
    OutputIterator rtrn = detail::copy_detect_random_access( control::getDefault(), first, n, result, user_code, std::iterator_traits< InputIterator >::iterator_category( ) );
BOLT_PROFILER_STOP_COPY_TRIAL
    return rtrn;
}

// default control
template<typename InputIterator, typename Size, typename OutputIterator> 
OutputIterator copy_n(InputIterator first, Size n, OutputIterator result, 
    const std::string& user_code)
{
BOLT_PROFILER_START_COPY_TRIAL
    OutputIterator rtrn = detail::copy_detect_random_access( control::getDefault(), first, n, result, user_code, std::iterator_traits< InputIterator >::iterator_category( ) );
BOLT_PROFILER_STOP_COPY_TRIAL
    return rtrn;
}

// user control
template<typename InputIterator, typename Size, typename OutputIterator> 
OutputIterator copy_n(const bolt::cl::control &ctrl, InputIterator first, Size n, OutputIterator result, 
    const std::string& user_code)
{
BOLT_PROFILER_START_COPY_TRIAL
    OutputIterator rtrn = detail::copy_detect_random_access( ctrl, first, n, result, user_code, std::iterator_traits< InputIterator >::iterator_category( ) );
BOLT_PROFILER_STOP_COPY_TRIAL
    return rtrn;
}

}//end of cl namespace
};//end of bolt namespace


namespace bolt {
namespace cl {
namespace detail {

enum copyTypeName { copy_iType, copy_oType };

/***********************************************************************************************************************
 * Kernel Template Specializer
 **********************************************************************************************************************/
class Copy_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
    public:

    Copy_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName( "copy_0"     );
        addKernelName( "copy_I_SA"  );
        addKernelName( "copy_II_LA" );
        addKernelName( "copy_III_SB");
        addKernelName( "copy_IV_LB" );
        addKernelName( "copy_V_SC"  );
        addKernelName( "copy_VI_LC" );
    }
    
    const ::std::string operator() ( const ::std::vector<::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString = 
            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(0) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"


            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(1) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(2) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(2) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(3) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(3) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(4) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(4) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(5) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(5) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(6) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(6) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"
            ;
    
        return templateSpecializationString;
    }
};



// Wrapper that uses default control class, iterator interface
template<typename InputIterator, typename Size, typename OutputIterator> 
OutputIterator copy_detect_random_access( const bolt::cl::control& ctrl, const InputIterator& first, const Size& n, 
    const OutputIterator& result, const std::string& user_code, std::input_iterator_tag )
{
    static_assert( false, "Bolt only supports random access iterator types" );
    return NULL;
};

template<typename InputIterator, typename Size, typename OutputIterator> 
OutputIterator copy_detect_random_access( const bolt::cl::control& ctrl, const InputIterator& first, const Size& n, 
    const OutputIterator& result, const std::string& user_code, std::random_access_iterator_tag )
{
    if (n > 0)
    {
        copy_pick_iterator( ctrl, first, n, result, user_code );
    }
    return (result+n);
};

/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
    \detail This template is called by the non-detail versions of inclusive_scan, it already assumes random access
 *  iterators.  This overload is called strictrly for non-device_vector iterators
*/
template<typename InputIterator, typename Size, typename OutputIterator> 
typename std::enable_if< 
             !(std::is_base_of<typename device_vector<typename std::iterator_traits<InputIterator>::value_type>::iterator,InputIterator>::value &&
               std::is_base_of<typename device_vector<typename std::iterator_traits<OutputIterator>::value_type>::iterator,OutputIterator>::value),
         void >::type
copy_pick_iterator(const bolt::cl::control &ctrl,  const InputIterator& first, const Size& n, 
        const OutputIterator& result, const std::string& user_code)
{
    typedef std::iterator_traits<InputIterator>::value_type iType;
    typedef std::iterator_traits<OutputIterator>::value_type oType;

    // Use host pointers memory since these arrays are only read once - no benefit to copying.

    // Map the input iterator to a device_vector
    device_vector< iType >  dvInput( first,  n, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, true, ctrl );

    // Map the output iterator to a device_vector
    device_vector< oType > dvOutput( result, n, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctrl );

    copy_enqueue( ctrl, dvInput.begin( ), n, dvOutput.begin( ), user_code );

    // This should immediately map/unmap the buffer
    dvOutput.data( );
}

// This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
// This is called strictrly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator> 
typename std::enable_if< 
              (std::is_base_of<typename device_vector<typename std::iterator_traits<DVInputIterator>::value_type>::iterator,DVInputIterator>::value &&
               std::is_base_of<typename device_vector<typename std::iterator_traits<DVOutputIterator>::value_type>::iterator,DVOutputIterator>::value),
         void >::type
copy_pick_iterator(const bolt::cl::control &ctrl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code)
{
    copy_enqueue( ctrl, first, n, result, user_code );
}

template< typename DVInputIterator, typename Size, typename DVOutputIterator > 
void copy_enqueue(const bolt::cl::control &ctrl, const DVInputIterator& first, const Size& n, 
    const DVOutputIterator& result, const std::string& cl_code)
{
#ifdef BOLT_PROFILER_ENABLED
aProfiler.nextStep();
aProfiler.setStepName("Acquire Kernel");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#endif
    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef std::iterator_traits<DVInputIterator>::value_type iType;
    typedef std::iterator_traits<DVOutputIterator>::value_type oType;
    std::vector<std::string> typeNames(2);
    typeNames[copy_iType] = TypeName< iType >::get( );
    typeNames[copy_oType] = TypeName< oType >::get( );

    /**********************************************************************************
     * Type Definitions - directrly concatenated into kernel string (order may matter)
     *********************************************************************************/
    std::vector<std::string> typeDefs;
    PUSH_BACK_UNIQUE( typeDefs, ClCode< iType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< oType >::get() )

    const size_t workGroupSize  = 256; //kernelWithBoundsCheck.getWorkGroupInfo< CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( ctrl.device( ), &l_Error );
    const size_t numComputeUnits = 28; //ctrl.device( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( ); // = 28
    const size_t numWorkGroupsPerComputeUnit = 10; //ctrl.wgPerComputeUnit( );
    const size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;
    
    const cl_uint numThreadsIdeal = static_cast<cl_uint>( numWorkGroups * workGroupSize );
    cl_uint numElementsPerThread = n / numThreadsIdeal;
    cl_uint numThreadsRUP = n;
    size_t mod = (n & (workGroupSize-1));
    int doBoundaryCheck = 0;
    if( mod )
    {
        numThreadsRUP &= ~mod;
        numThreadsRUP += workGroupSize;
        doBoundaryCheck = 1;
    }
    
    /**********************************************************************************
     * Compile Options
     *********************************************************************************/
    std::string compileOptions;
    std::ostringstream oss;
    oss << " -DBURST_SIZE=" << BURST_SIZE;
    oss << " -DBOUNDARY_CHECK=" << doBoundaryCheck;
    compileOptions = oss.str();

    /**********************************************************************************
     * Request Compiled Kernels
     *********************************************************************************/
    Copy_KernelTemplateSpecializer c_kts;
    std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
        ctrl,
        typeNames,
        &c_kts,
        typeDefs,
        copy_kernels,
        compileOptions);

    /**********************************************************************************
     *  Kernel
     *********************************************************************************/
#ifdef BOLT_PROFILER_ENABLED
size_t k0e_stepNum, k0s_stepNum, k0_stepNum, ret_stepNum;
aProfiler.nextStep();
aProfiler.setStepName("Setup Kernel");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#endif
    ::cl::Event kernelEvent;
    cl_int l_Error;
    try
    {
        int whichKernel = 6;
        cl_uint numThreadsChosen;
        cl_uint workGroupSizeChosen = workGroupSize;
        switch( whichKernel )
        {
        case 0: // 0: 1 thread per element
            numThreadsChosen = numThreadsRUP;
            break;
        case 1: // I:   SA
        case 3: // III: SB
        case 5: // V:   SC
            numThreadsChosen = numThreadsRUP / BURST_SIZE;
            break;
        case 2: // II:  LA
        case 4: // IV:  LB
        case 6: // VI:  LC
            numThreadsChosen = numThreadsIdeal;
            break;
        } // switch

        std::cout << "NumElem: " << n << "; NumThreads: " << numThreadsChosen << "; NumWorkGroups: " << numThreadsChosen/workGroupSizeChosen << std::endl;

        V_OPENCL( kernels[whichKernel].setArg( 0, first->getBuffer()), "Error setArg kernels[ 0 ]" ); // Input keys
        V_OPENCL( kernels[whichKernel].setArg( 1, result->getBuffer()),"Error setArg kernels[ 0 ]" ); // Input buffer
        V_OPENCL( kernels[whichKernel].setArg( 2, static_cast<cl_uint>( n ) ),                 "Error setArg kernels[ 0 ]" ); // Size of buffer


#ifdef BOLT_PROFILER_ENABLED
aProfiler.nextStep();
aProfiler.setStepName("Enqueue Kernel");
k0e_stepNum = aProfiler.getStepNum();
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.nextStep();
aProfiler.setStepName("Submit Kernel");
k0s_stepNum = aProfiler.getStepNum();
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.nextStep();
aProfiler.setStepName("Kernel");
k0_stepNum = aProfiler.getStepNum();
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.set(AsyncProfiler::memory, n*sizeof(iType) + n*sizeof(oType));
#endif
        l_Error = ctrl.commandQueue( ).enqueueNDRangeKernel(
            kernels[whichKernel],
            ::cl::NullRange,
            ::cl::NDRange( numThreadsChosen ),
            ::cl::NDRange( workGroupSizeChosen ),
            NULL,
            &kernelEvent);
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel" );
    }
    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( ) in bolt::cl::copy_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }

    // wait for results
    bolt::cl::wait(ctrl, kernelEvent);

#ifdef BOLT_PROFILER_ENABLED
aProfiler.nextStep();
aProfiler.setStepName("Returning Control To Device");
ret_stepNum = aProfiler.getStepNum();
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.nextStep();
aProfiler.setStepName("Querying Kernel Times");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);

aProfiler.setDataSize(n*sizeof(iType));
std::string strDeviceName = ctrl.device().getInfo< CL_DEVICE_NAME >( &l_Error );
bolt::cl::V_OPENCL( l_Error, "Device::getInfo< CL_DEVICE_NAME > failed" );
aProfiler.setArchitecture(strDeviceName);

    try
    {
        cl_ulong k0enq, k0sub, k0start, k0stop;
        //cl_ulong k1sub, k1start, k1stop;
        //cl_ulong k2sub, k2start, k2stop;
        //cl_ulong ret;

        //cl_ulong k0_start, k0_stop, k1_stop, k2_stop;
        //cl_ulong k1_start, k2_start;
        
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_QUEUED, &k0enq),   "getProfInfo" );
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &k0sub),   "getProfInfo" );
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,  &k0start), "getProfInfo" );
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,    &k0stop),  "getProfInfo" );

        //V_OPENCL( kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &k1sub),   "getProfInfo" );
        //V_OPENCL( kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,  &k1start), "getProfInfo" );
        //V_OPENCL( kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,    &k1stop),  "getProfInfo" );

        //V_OPENCL( kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &k2sub),   "getProfInfo" );
        //V_OPENCL( kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,  &k2start), "getProfInfo" );
        //V_OPENCL( kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,    &k2stop),  "getProfInfo" );
#if 0
        printf("BEFORE\n");
        printf("K0 Enque %10u ns CPU\n", aProfiler.get(k0e_stepNum, AsyncProfiler::startTime));
        printf("K0 Enque %10u ns\n", k0enq);
        printf("K0 Submt %10u ns\n", k0sub);
        printf("K0 Start %10u ns\n", k0start);
        printf("K0 Stop  %10u ns\n", k0stop);
        printf("K1 Submt %10u ns\n", k1sub);
        printf("K1 Start %10u ns\n", k1start);
        printf("K1 Stop  %10u ns\n", k1stop);
        printf("K2 Submt %10u ns\n", k2sub);
        printf("K2 Start %10u ns\n", k2start);
        printf("K2 Stop  %10u ns\n", k2stop);
        printf("Return   %10u ns\n", aProfiler.get(ret_stepNum, AsyncProfiler::startTime) );
        printf("Returned %10u ns\n", aProfiler.get(ret_stepNum, AsyncProfiler::stopTime) );
#endif
        // determine shift between cpu and gpu clock according to kernel 0 enqueue time
        size_t k0_enq_cpu = aProfiler.get(k0e_stepNum, AsyncProfiler::startTime);
        size_t k0_enq_gpu = static_cast<size_t>( k0enq );
        long long shift = k0enq - k0_enq_cpu; // must be signed because can be '-'
        //printf("\nSHIFT % 10u ns\n", shift );

        // apply shift to all steps
        aProfiler.set(k0e_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k0enq  -shift) ); // same
        aProfiler.set(k0e_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k0sub  -shift) );
        aProfiler.set(k0s_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k0sub  -shift) );
        aProfiler.set(k0s_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k0start-shift) );
        aProfiler.set(k0_stepNum,  AsyncProfiler::startTime, static_cast<size_t>(k0start-shift) );
        aProfiler.set(k0_stepNum,  AsyncProfiler::stopTime,  static_cast<size_t>(k0stop -shift) );

        //aProfiler.set(k1s_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k0sub  -shift) );
        //aProfiler.set(k1s_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k1start-shift) );
        //aProfiler.set(k1_stepNum,  AsyncProfiler::startTime, static_cast<size_t>(k1start-shift) );
        //aProfiler.set(k1_stepNum,  AsyncProfiler::stopTime,  static_cast<size_t>(k1stop -shift) );

        //aProfiler.set(k2s_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k1stop -shift) );
        //aProfiler.set(k2s_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k2start-shift) );
        //aProfiler.set(k2_stepNum,  AsyncProfiler::startTime, static_cast<size_t>(k2start-shift) );
        //aProfiler.set(k2_stepNum,  AsyncProfiler::stopTime,  static_cast<size_t>(k2stop -shift) );

        aProfiler.set(ret_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k0stop -shift) );
        // aProfiler.set(ret_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k2_stop-shift) ); // same
#if 0
        printf("\nAFTER\n");
        printf("K0 Enque %10u ns CPU\n", aProfiler.get(k0e_stepNum, AsyncProfiler::startTime) );
        printf("K0 Enque %10u ns GPU\n", aProfiler.get(k0e_stepNum, AsyncProfiler::startTime) );
        printf("K0 Submt %10u ns GPU\n", aProfiler.get(k0s_stepNum, AsyncProfiler::startTime) );
        printf("K0 Start %10u ns GPU\n", aProfiler.get(k0_stepNum,  AsyncProfiler::startTime) );
        printf("K0 Stop  %10u ns GPU\n", aProfiler.get(k0_stepNum,  AsyncProfiler::stopTime ) );
        printf("K1 Submt %10u ns GPU\n", aProfiler.get(k1s_stepNum, AsyncProfiler::startTime) );
        printf("K1 Start %10u ns GPU\n", aProfiler.get(k1_stepNum,  AsyncProfiler::startTime) );
        printf("K1 Stop  %10u ns GPU\n", aProfiler.get(k1_stepNum,  AsyncProfiler::stopTime ) );
        printf("K2 Submt %10u ns GPU\n", aProfiler.get(k2s_stepNum, AsyncProfiler::startTime) );
        printf("K2 Start %10u ns GPU\n", aProfiler.get(k2_stepNum,  AsyncProfiler::startTime) );
        printf("K2 Stop  %10u ns GPU\n", aProfiler.get(k2_stepNum,  AsyncProfiler::stopTime ) );
        printf("Return   %10u ns GPU\n", aProfiler.get(ret_stepNum, AsyncProfiler::startTime) );
        printf("Returned %10u ns CPU\n", aProfiler.get(ret_stepNum, AsyncProfiler::stopTime ) );
#endif
    }
    catch( ::cl::Error& e )
    {
        std::cout << ( "Scan Benchmark error condition reported:" ) << std::endl << e.what() << std::endl;
        return;
    }


#endif // ENABLE_PROFILING


    // profiling
    cl_command_queue_properties queueProperties;
    l_Error = ctrl.commandQueue().getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &queueProperties);
    unsigned int profilingEnabled = queueProperties&CL_QUEUE_PROFILING_ENABLE;
    if ( profilingEnabled ) {
        cl_ulong start_time, stop_time;
        
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time), "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,    &stop_time), "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
        size_t time = stop_time - start_time;
        double gb = (n*(sizeof(iType)+sizeof(oType))/1024.0/1024.0/1024.0);
        double sec = time/1000000000.0;
        std::cout << "Global Memory Bandwidth: " << ( gb / sec) << " ( "
          << time/1000000.0 << " ms)" << std::endl;
    }
};
}//End OF detail namespace
}//End OF cl namespace
}//End OF bolt namespace

#endif
