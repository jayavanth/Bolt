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

#include "common/stdafx.h"

#include <bolt/amp/transform.h>
#include <bolt/amp/functional.h>
#include <bolt/amp/iterator/constant_iterator.h>
#include <bolt/amp/iterator/counting_iterator.h>
#include <bolt/miniDump.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>

#define WAVEFRNT_SIZE 256


class array_iter
{
public:
    //array_iter(concurrency::array<int>& other2) 
    //{
    //    arr = other2;
    //}

    array_iter(concurrency::array<int> &other2):arr(other2) { }

    int operator[](int x) const restrict(cpu,amp)
    {
      return arr[x];
    }
    
    concurrency::array<int>& arr;
};


TEST( TransformUDD, UDDTestDeviceVector)
{
    bolt::amp::control ctl;
    concurrency::accelerator_view av = ctl.getAccelerator().default_view;

    const unsigned int tileSize = WAVEFRNT_SIZE;
    concurrency::extent< 1 > inputExtent( WAVEFRNT_SIZE );
    concurrency::tiled_extent< tileSize > tiledExtentTransform = inputExtent.tile< tileSize >();

    typedef int etype;
    typedef int ptype;
    typedef std::vector<etype> intev;
    typedef std::vector<ptype> intpv;

    etype elements[WAVEFRNT_SIZE];
    ptype permutation[WAVEFRNT_SIZE];

    std::for_each( elements, elements+WAVEFRNT_SIZE, [](int &n) mutable { static int _i=1; n=_i++; } );
    std::fill(permutation, permutation+WAVEFRNT_SIZE, 0);


    bolt::amp::device_vector<int, concurrency::array_view> inputV2(elements, elements + WAVEFRNT_SIZE);
    bolt::amp::device_vector<int, concurrency::array_view> resultV2(permutation,permutation + WAVEFRNT_SIZE);
    intev __inputV(elements, elements+WAVEFRNT_SIZE);

    auto inputV20 = inputV2.begin().getContainer().getBuffer(inputV2.begin());
    auto resultV20 = resultV2.begin().getContainer().getBuffer(resultV2.begin());

    concurrency::array<int> __a( inputExtent, __inputV.begin(), __inputV.end() );

    array_iter a(__a);

    size_t esize = sizeof(elements)/sizeof(etype);

    intev e( elements, elements + 10 );
    intpv p( permutation, permutation + 10 );

    
    concurrency::parallel_for_each(av, tiledExtentTransform, [=](concurrency::tiled_index<tileSize> t_idx) restrict(amp)
    {
        unsigned int globalId = t_idx.global[ 0 ];
        //resultV[globalId] = f(inputV1[globalId], inputV2[globalId]);
        //resultV0[globalId] = inputV1[globalId];
        resultV20[globalId] = a[globalId];
    });
    resultV20.synchronize();
    std::cout<<"     input"<<"       output"<<std::endl;
    for ( int i = 0 ; i < WAVEFRNT_SIZE ; i++ )
    {
        std::cout<<inputV2[i]<<"       "<<resultV20[i]<<std::endl;
    }
}





int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    // Define MEMORYREPORT on windows platfroms to enable debug memory heap checking
#if defined( MEMORYREPORT ) && defined( _WIN32 )
    TCHAR logPath[ MAX_PATH ];
    ::GetCurrentDirectory( MAX_PATH, logPath );
    ::_tcscat_s( logPath, _T( "\\MemoryReport.txt") );

    // We leak the handle to this file, on purpose, so that the ::_CrtSetReportFile() can output it's memory 
    // statistics on app shutdown
    HANDLE hLogFile;
    hLogFile = ::CreateFile( logPath, GENERIC_WRITE, 
        FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

    ::_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );

    ::_CrtSetReportFile( _CRT_ASSERT, hLogFile );
    ::_CrtSetReportFile( _CRT_ERROR, hLogFile );
    ::_CrtSetReportFile( _CRT_WARN, hLogFile );

    int tmp = ::_CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
    tmp |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF;
    ::_CrtSetDbgFlag( tmp );

    // By looking at the memory leak report that is generated by this debug heap, there is a number with 
    // {} brackets that indicates the incremental allocation number of that block.  If you wish to set
    // a breakpoint on that allocation number, put it in the _CrtSetBreakAlloc() call below, and the heap
    // will issue a bp on the request, allowing you to look at the call stack
    // ::_CrtSetBreakAlloc( 1833 );

#endif /* MEMORYREPORT */

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Set the standard OpenCL wait behavior to help debugging
    //bolt::cl::control& myControl = bolt::cl::control::getDefault( );
    // myControl.waitMode( bolt::cl::control::NiceWait );
    //myControl.forceRunMode( bolt::cl::control::MultiCoreCpu );  // choose tbb

    int retVal = RUN_ALL_TESTS( );

    //  Reflection code to inspect how many tests failed in gTest
    ::testing::UnitTest& unitTest = *::testing::UnitTest::GetInstance( );

    unsigned int failedTests = 0;
    for( int i = 0; i < unitTest.total_test_case_count( ); ++i )
    {
        const ::testing::TestCase& testCase = *unitTest.GetTestCase( i );
        for( int j = 0; j < testCase.total_test_count( ); ++j )
        {
            const ::testing::TestInfo& testInfo = *testCase.GetTestInfo( j );
            if( testInfo.result( )->Failed( ) )
                ++failedTests;
        }
    }

    //  Print helpful message at termination if we detect errors, to help users figure out what to do next
    if( failedTests )
    {
        bolt::tout << _T( "\nFailed tests detected in test pass; please run test again with:" ) << std::endl;
        bolt::tout << _T( "\t--gtest_filter=<XXX> to select a specific failing test of interest" ) << std::endl;
        bolt::tout << _T( "\t--gtest_catch_exceptions=0 to generate minidump of failing test, or" ) << std::endl;
        bolt::tout << _T( "\t--gtest_break_on_failure to debug interactively with debugger" ) << std::endl;
        bolt::tout << _T( "\t    (only on googletest assertion failures, not SEH exceptions)" ) << std::endl;
    }

    return retVal;
}
