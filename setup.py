from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='salmetric_cpp',
    ext_modules=[
        CppExtension(
        	name='salmetric_cpp', 
        	sources=['salmetric.cpp','lodepng.cpp'],
        	extra_compile_args=['-lopencv_imgcodecs','-lopencv_highgui','-lopencv_imgproc']),
    ],
    cmdclass={
        'build_ext': BuildExtension
})