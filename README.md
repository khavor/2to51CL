# 2to51CL
This is a Windows x64 Command Line utility that uses the sndfile library and the libfftw3-3 library to import a stereo audio file and export a multichannel 5.1 audio file in the standard L R C LFE Ls Rs layout.

It uses a Fast Fourier Transform to extract the center channel from the stereo audio, using arguments for center width and falloff slopes. It then subtracts the center channel from the original stereo audio to create the new front left and right channels. The rear surround left and right channels, again using an FFT, are created by extracting audio signals that are panned to the extreme left and right. This also uses arguments for width (the amount taken from the left and right sides) and falloff slope. These rear surround channels are subtracted from the front left and right channels. The LFE channel is created by applying a low pass filter, at a settable cross-over frequency, to the original stereo audio. All the other channels have a high pass filter applied at the same cross-over frequency.

The output is a 6 channel audio file with the standard L R C LFE Ls Rs layout.

Clone this repository somewhere on your hard drive and open the folder in the MSVC x64 Native Tools Command Prompts for VS 2022.
Run compile.bat.
The file compile.bat will create a resource containing the sndfile.dll and libfftw3-3.dll and compile main.cpp to 2to51CL.exe.

Run 2to51CL.exe without arguments to display the help.

The two required dlls, sndfile.dll and libfftw3-3.dll will be embedded in the executable file and written to disk in the same folder as 2to51CL.exe when it's run, so they don't need to be included where ever you install 2to51CL.exe
