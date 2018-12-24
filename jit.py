from torch.utils.cpp_extension import load
salmetric_cpp = load(name="salmetric_cpp", sources=["salmetric.cpp"], verbose=True)
help(salmetric_cpp)