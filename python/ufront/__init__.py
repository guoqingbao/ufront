import site
import os
from ctypes import cdll
import platform
#Preload package dependencies
packages = site.getsitepackages()
for package in packages:
    if package.find("site-packages") >=0 or package.find("dist-packages") >=0:
        if platform.system() == "Windows":
            path = package + "/ufront/.libs"
        elif platform.system() == "Darwin":
            path = package + "/ufront/.dylibs"
        else:
            path = package + "/ufront.libs"

        if os.path.exists(path):
            files = os.listdir(path)
            orders = ["libicudata", "libicuuc", "liblzma", "libtinfo", "libbsd", "libffi", "libedit", "libxml2", "libomp", "libLLVM", "libUfrontCAPI"]
            libs = [file for x in orders for file in files if file.find(x) >=0]
            for lib in libs:
                if os.path.exists(path + "/" + lib): cdll.LoadLibrary(path + "/" + lib)
            break         

from .pytorch.model import UFrontTorch
from .onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX
# from .keras.model import UFrontKeras
from .ufront import *