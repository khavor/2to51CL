rc.exe resources.rc
cl /std:c++17 /EHsc /O2 /arch:AVX2 /I"include" main.cpp resources.res /link /LIBPATH:"lib" delayimp.lib sndfile.lib libfftw3-3.lib /DELAYLOAD:sndfile.dll /DELAYLOAD:libfftw3-3.dll /OUT:bin\2to51CL.exe
