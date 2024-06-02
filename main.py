from Py2C import Py2C
# path_w= ["Conv.cc", "Conv.hh", "Pool.cc", "Pool.hh", "Dense.cc", "CNN.hh"]
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
pyc_lib = Py2C("testing_model.h5")
pyc_lib.convert2C()
pyc_lib.WriteCfile()
# pyc_lib.del_one_file("CNN.hh")
# pyc_lib.del_any_file(path_w)
# pyc_lib.del_all_file()
# pyc_lib.set_Fxp_Param((16,6))
# pyc_lib.Write_IEEE754_32bits_Weights_File()
pyc_lib.Write_Float_Weights_File()
#pyc_lib.Write_FixedPoint_Weights_File()

