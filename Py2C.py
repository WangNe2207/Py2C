import os
import struct

import numpy as np
import tensorflow as tf
import math


class Py2C:

    """
    The Py2C class has four inputs and nine functions.

        With input:
        - model_path (string) is path of h5 model file (Note: This function support CNN and ANN)
        - type (string) is type of data such as int, float, fxp (default: "fxp")
        - fxp_para is parameter of fxp if you choose. It has 2 parameters (x,y) with x is sum of bits showing a data and y is integral part of the data
        - if choose_only_output is False, output C code model show full array. Else it will show the only variable being argmax of array
        - resnet is variable to choose whether if your model is resnet or not
        - For Resnet only
            + num_layers_in_block: the number of layer in each resnet's block
            + num_layers_before_block: the number of layer before the series of block in your resnet model
            + num_block: the number of block in your resnet model
        With function:
        - set_Fxp_Param function is to set fxp parameter again
        - convert2C function is to convert the loaded model into C code and it store in self array
        - WriteCfile function is to write C code from convert2C into .cc and .hh file
        - del_one_file function is to delete the particular file
        - del_any_file function is to delete any file in the particular array
        - del_all_file function is to delete all of .cc and .hh file, which has created
        - Write_Float_Weights_File function is to create Float Weights file
        - Write_IEEE754_32bits_Weights_File function is to create IEEE754 32bits Weights file
        - Write_FixedPoint_Weights_File function is to create Fixed Point Weights file
    """

    def __init__(self, model_path, type="float", fxp_para=(20,10), choose_only_output=True, resnet=False, num_layers_in_block=6,num_layers_before_block=3, num_blocks=1):
        self.model = tf.keras.models.load_model(model_path)
        assert fxp_para[0] > 0 and fxp_para[1] > 0, "the 1st or the 2nd Fxp Parameter must be more than zero!!!"
        assert fxp_para[0] >= fxp_para[1], "the 1st Fxp Parameter must be equal or more than the 2nd one!!!"
        if type == "fxp":
            self.fxp_para = fxp_para
        else:
            self.fxp_para = None
        self.choose_only_output = choose_only_output
        self.type = type
        self.resnet = resnet
        self.config = self.model.get_config()
        self.index = 0

        self.base_include = ""
        self.fxp_include = "#include <ap_axi_sdata.h>\ntypedef ap_fixed<" + str(fxp_para[0]) + "," + str(
            fxp_para[1]) + "> fxp;\n"
        self.CNN_include = "#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#include <algorithm>\n#include <string.h>\n"
        self.source_Conv_cc = ""

        self.act_arr = ""
        self.fxp_inc = ""
        self.base_inc = ""
        self.full_source_Conv_cc = []
        self.full_source_Conv_hh = []
        self.full_source_Pool_cc = []
        self.full_source_Pool_hh = []
        self.full_source_Dense_cc = []
        self.full_source_Dense_hh = []
        self.full_source_CNN_cc = []
        self.Weights = []
        self.source_CNN = ""
        self.source_CNN_hh = ""
        self.source_CNN_tb = ""
        self.index = 0
        self.index2D = 0
        self.indexBatch = 0
        self.indexActi = 0
        self.index_P = 0
        self.index_P2D = 0
        self.index_D = 0
        self.index_basic_block = 0
        self.num_layers_in_block = num_layers_in_block
        self.num_layers_before_block = num_layers_before_block
        self.num_blocks = num_blocks
        self.cnt_param = 0
        self.call_function = ""
        self.out = ""
        self.full_source_CNN_cc.append(["", "&InModel[0]"])
        self.path_w = ["Conv.cpp", "Conv.h", "Pool.cpp", "Pool.h", "Dense.cpp", "Dense.h", "CNN.cpp", "CNN.h",
                       "CNN_tb.cpp"]
        print("Model Information")
        # self.model.summary()

    def set_Fxp_Param(self, fxp_para):
        assert fxp_para[0] > 0 and fxp_para[1] > 0, "the 1st or the 2nd Fxp Parameter must be more than zero!!!"
        assert fxp_para[0] >= fxp_para[1], "the 1st Fxp Parameter must be equal or more than the 2nd one!!!"
        self.fxp_para = fxp_para

    def convert2C(self):
            #Resnet =======================================================================================
            for i in range(len(self.config["layers"])):
                # for i in range(6):
                layer = self.config["layers"][i]['config']['name']
                if self.resnet:
                    if ((i == self.num_layers_before_block + 1) and (self.index_basic_block < self.num_blocks)) or ((i == self.num_layers_before_block + 1 + self.num_layers_in_block * self.index_basic_block) and (self.index_basic_block < self.num_blocks)):
                        if (len(self.model.layers[i - 1].output.shape) == 4):
                            out_shape = (self.model.layers[i - 1].output.shape[3], self.model.layers[i - 1].output.shape[1],
                                         self.model.layers[i - 1].output.shape[2])
                            self.call_function += "\t" + self.type + " skip_" + str(self.index_basic_block) + "[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        else:
                            out_shape = (self.model.layers[i - 1].output.shape[1],self.model.layers[i - 1].output.shape[2])
                            self.call_function += "\t" + self.type + " skip_" + str(self.index_basic_block) + "[" + str(
                                out_shape[0] * out_shape[1]) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tmemcpy(skip_" + str(self.index_basic_block), "sizeof(", "));"])
                        self.index_basic_block += 1

                    if layer.find("add") >= 0 :
                        if (len(self.model.layers[i - 1].output.shape) == 4):
                            out_shape = (self.model.layers[i - 1].output.shape[3], self.model.layers[i - 1].output.shape[1],
                                         self.model.layers[i - 1].output.shape[2])
                            self.call_function += "\t" + self.type + " end_basicblock_" + str(self.index_basic_block - 1) + "[" + str(
                                out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\taddArray(", "skip_" ,str(self.index_basic_block - 1) ,"end_basicblock_" , str(self.index_basic_block - 1), str(out_shape[0] * out_shape[1] * out_shape[2]),");"])
                        else:
                            out_shape = (self.model.layers[i - 1].output.shape[1],
                                         self.model.layers[i - 1].output.shape[2])
                            self.call_function += "\t" + self.type + " end_basicblock_" + str(self.index_basic_block - 1) + "[" + str(
                                out_shape[0] * out_shape[1]) + "];\n"
                            self.full_source_CNN_cc.append(
                                ["\taddArray(", "skip_" ,str(self.index_basic_block - 1) ,"end_basicblock_" , str(self.index_basic_block - 1), str(out_shape[0] * out_shape[1]),");"])
                # convert conv2d layer into c array that act like an conv2d layer
                if layer.find("conv2d") >= 0 and layer.find("conv2d_input") < 0:
                    activation = self.config["layers"][i]['config']['activation']
                    in_shape = (self.model.layers[i].input.shape[3], self.model.layers[i].input.shape[1],
                                self.model.layers[i].input.shape[2])
                    out_shape = (self.model.layers[i].output.shape[3], self.model.layers[i].output.shape[1],
                                 self.model.layers[i].output.shape[2])
                    kernel_shape = (
                    self.model.layers[i].get_weights()[0].shape[3], self.model.layers[i].get_weights()[0].shape[2],
                    self.model.layers[i].get_weights()[0].shape[0], self.model.layers[i].get_weights()[0].shape[1])
                    h = np.transpose(self.model.layers[i].get_weights()[0], (3, 2, 0, 1)).reshape(
                        kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
                    for k in h:
                        self.Weights.append(k)
                    for k in self.model.layers[i].get_weights()[1]:
                        self.Weights.append(k)
                    if activation == "relu":
                        self.act_arr = "if ((s+bias[n])<0) Output_Conv[" + str(out_shape[1]) + "*" + str(
                            out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=0; else Output_Conv[" + str(
                            out_shape[1]) + "*" + str(out_shape[2]) + "*n+" + str(out_shape[2]) + "*x+y]=s+bias[n];"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=s+bias[n];"
                    if self.type == "fxp" and self.index2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    stride = self.model.layers[i].strides[0]
                    if self.config["layers"][i]['config']['padding'] == 'same':
                        padding = (math.floor(((out_shape[1] - 1) * stride - in_shape[1] + kernel_shape[2])/2),
                                   math.floor(((out_shape[2] - 1) * stride - in_shape[2] + kernel_shape[3])/2))
                        in_shape_if_padding = (in_shape[0], (in_shape[1] + 2*padding[0]), (in_shape[2] + 2*padding[1]))
                        source_pad_conv_cc = self.fxp_inc + "void Padding_Conv2D_" + str(
                            self.index2D) + "(" + self.type + " input_Pad_Conv[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Conv[" + str(
                            in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[
                                2]) + "]){\n\tloop_for_3_channel_pad_" + str(
                            self.index2D) + ":\n\tfor (int c = 0; c < " + str(
                            in_shape_if_padding[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                            self.index2D) + ":\n\t\tfor (int n = 0; n < " + str(
                            in_shape_if_padding[1]) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                            self.index2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                            in_shape_if_padding[2]) + "; i++){\n\t\t\t\tif (n < " + str(
                            0 + (padding[0])) + " || n >= " + str(
                            in_shape[2] + (padding[0])) + ") output_Pad_Conv[" + str(
                            in_shape_if_padding[1]) + " * " + str(in_shape_if_padding[2]) + " * c + " + str(
                            in_shape_if_padding[2]) + " * n + i]=0;\n\t\t\t\t else \n\t\t\t\t\tif (i < " + str(
                            0 + (padding[0])) + " || i >= " + str(
                            in_shape[2] + (padding[0])) + ") output_Pad_Conv[" + str(
                            in_shape_if_padding[1]) + " * " + str(in_shape_if_padding[2]) + " * c + " + str(
                            in_shape_if_padding[2]) + " * n + i]=0; else output_Pad_Conv[" + str(
                            in_shape_if_padding[1]) + " * " + str(in_shape_if_padding[2]) + " * c + " + str(
                            in_shape_if_padding[2]) + " * n + i] = input_Pad_Conv[" + str(in_shape[1]) + " * " + str(
                            in_shape[2]) + " * c + " + str(in_shape[2]) + " * (n - " + str(
                            int(padding[1])) + ") + i - " + str(
                            int(padding[1])) + "];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_pad_conv_hh = self.fxp_inc + "void Padding_Conv2D_" + str(
                            self.index2D) + "(" + self.type + " input_Pad_Conv[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Conv[" + str(
                            in_shape[0] * (in_shape[1] + 2*padding[0]) * (in_shape[2] + 2*padding[1])) + "]);\n"
                        self.full_source_Conv_cc.append(source_pad_conv_cc)
                        self.full_source_Conv_hh.append(source_pad_conv_hh)
                        self.call_function += "\t" + self.type + " OutPadConv" + str(self.index2D) + "[" + str(
                            in_shape[0] * (in_shape[1] + 2*padding[0]) * (in_shape[2] + 2*padding[1])) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tPadding_Conv2D_" + str(self.index2D) + "(", "OutPadConv" + str(self.index2D), "",
                             ""])

                        self.call_function += "\t" + self.type + " OutConv2D" + str(self.index2D) + "[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        source_Conv_cc = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(
                            in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[
                                2]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[
                                3]) + "]){\n\tloop_for_channel2D_" + str(
                            self.index2D) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(
                            out_shape[0]) + "; n++){\n\t\tloop_for_bp2D_" + str(
                            self.index2D) + ":\n\t\tfor (int x = 0; x < " + str(
                            out_shape[1]) + "; x++){\n\t\t\tloop_for_ap2D_" + str(
                            self.index2D) + ":\n\t\t\tfor (int y = 0; y < " + str(
                            out_shape[2]) + "; y++){\n\t\t\t\t" + self.type + " s = 0;\n\t\t\t\tloop_for_fc_" + str(
                            self.index2D) + ":\n\t\t\t\tfor (int k = 0; k < " + str(
                            kernel_shape[1]) + "; k++){\n\t\t\t\t\tloop_for_fb_" + str(
                            self.index2D) + ":\n\t\t\t\t\tfor (int i = 0; i < " + str(
                            kernel_shape[2]) + "; i++){\n\t\t\t\t\t\tloop_for_fa_" + str(
                            self.index2D) + ":\n\t\t\t\t\t\tfor (int j = 0; j < " + str(
                            kernel_shape[3]) + "; j++){\n\t\t\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                            kernel_shape[2]) + "*" + str(
                            kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*" + str(
                            kernel_shape[3]) + "*k+" + str(
                            kernel_shape[3]) + "*i+j])*(Input_Conv[" + str(
                            in_shape_if_padding[1]) + "*" + str(
                            in_shape_if_padding[2]) + "*k+" + str(
                            in_shape_if_padding[
                                2]) + "*(i+x*stride)+j+y*stride]);}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\t" + self.act_arr + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(
                            in_shape_if_padding[0] * in_shape_if_padding[1] * in_shape_if_padding[
                                2]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"
                    else:
                        self.call_function += "\t" + self.type + " OutConv2D" + str(self.index2D) + "[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                        source_Conv_cc = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[
                                3]) + "]){\n\tint stride = " + str(stride) +";\n\tloop_for_channel2D_" + str(
                            self.index2D) + ":\n\tfor (int n = 0; n < " + str(
                            out_shape[0]) + "; n++){\n\t\tloop_for_bp2D_" + str(
                            self.index2D) + ":\n\t\tfor (int x = 0; x < " + str(
                            out_shape[1]) + "; x++){\n\t\t\tloop_for_ap2D_" + str(
                            self.index2D) + ":\n\t\t\tfor (int y = 0; y < " + str(
                            out_shape[2]) + "; y++){\n\t\t\t\t" + self.type + " s = 0;\n\t\t\t\tloop_for_fc_" + str(
                            self.index2D) + ":\n\t\t\t\tfor (int k = 0; k < " + str(
                            kernel_shape[1]) + "; k++){\n\t\t\t\t\tloop_for_fb_" + str(
                            self.index2D) + ":\n\t\t\t\t\tfor (int i = 0; i < " + str(
                            kernel_shape[2]) + "; i++){\n\t\t\t\t\t\tloop_for_fa_" + str(
                            self.index2D) + ":\n\t\t\t\t\t\tfor (int j = 0; j < " + str(
                            kernel_shape[3]) + "; j++){\n\t\t\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                            kernel_shape[2]) + "*" + str(
                            kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*" + str(
                            kernel_shape[3]) + "*k+" + str(
                            kernel_shape[3]) + "*i+j])*(Input_Conv[" + str(
                            in_shape[1]) + "*" + str(
                            in_shape[2]) + "*k+" + str(
                            in_shape[
                                2]) + "*(i+x*stride)+j+y*stride]);}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\t" + self.act_arr + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Conv2D_" + str(
                            self.index2D) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1] * out_shape[2]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) + "]);\n"

                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.full_source_CNN_cc.append(
                        ["\tConv2D_" + str(self.index2D) + "(", "OutConv2D" + str(self.index2D), "&Weights[" + str(
                            self.cnt_param + kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[
                                3]) + "]",
                         "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] + out_shape[
                        0]
                    self.index2D += 1

                # convert conv1d layer into c array that act like an conv1d layer
                if layer.find("conv1d") >= 0 and layer.find("conv1d_input") < 0:
                    activation = self.config["layers"][i]['config']['activation']
                    in_shape = (self.model.layers[i].input.shape[2], self.model.layers[i].input.shape[1])
                    out_shape = (self.model.layers[i].output.shape[2], self.model.layers[i].output.shape[1])
                    kernel_shape = self.model.layers[i].get_weights()[0].T.shape
                    h = self.model.layers[i].get_weights()[0].T.reshape(
                        kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
                    # print(i - 1)
                    for k in h:
                        self.Weights.append(k)
                    for k in self.model.layers[i].get_weights()[1]:
                        self.Weights.append(k)
                    if activation == "relu":
                        self.act_arr = "if ((s+bias[n])<0) Output_Conv[" + str(
                            out_shape[1]) + "*n+y]=0; else Output_Conv[" + str(out_shape[1]) + "*n+y]=s+bias[n];"
                    else:
                        self.act_arr = "Output_Conv[" + str(out_shape[1]) + "*n+y]=s+bias[n];"
                    if self.type == "fxp" and self.index == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    stride = self.model.layers[i].strides[0]
                    if self.config["layers"][i]['config']['padding'] == 'same':
                        padding = math.floor(((out_shape[1] - 1)*stride - (in_shape[1] - (kernel_shape[2] - 1) - 1))/2)
                        in_shape_if_padding = (in_shape[0] ,(in_shape[1] + 2 * padding))
                        source_pad_conv_cc = self.fxp_inc + "void Padding_Conv2D_" + str(self.index) + "(" + self.type + " input_Pad_Conv[" + str(in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Conv[" + str(in_shape_if_padding[0] * in_shape_if_padding[1]) + "]){\n\tloop_for_3_channel_pad_" + str(self.index) + ":\n\tfor (int c = 0; c < " + str(in_shape_if_padding[0]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(self.index) + ":\n\t\tfor (int n = 0; n < " + str(in_shape_if_padding[1]) + "; n++){\n\t\t\tif (n < " + str(0 + padding) + " || n >= " + str(in_shape[1] + padding) + ") output_Pad_Conv[" + str(in_shape_if_padding[1]) + " * c + n]=0; else output_Pad_Conv[" + str(in_shape_if_padding[1]) + " * c + n] = input_Pad_Conv[" + str(in_shape[1]) + " * c + n - " + str(padding) + "];\n\t\t}\n\t}\n}\n"
                        source_pad_conv_hh = self.fxp_inc + "void Padding_Conv2D_" + str(
                            self.index) + "(" + self.type + " input_Pad_Conv[" + str(
                            in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Conv[" + str(
                            in_shape[0] * (in_shape[1] + 2 * padding)) + "]);\n"
                        self.full_source_Conv_cc.append(source_pad_conv_cc)
                        self.full_source_Conv_hh.append(source_pad_conv_hh)
                        self.call_function += "\t" + self.type + " OutPadConv" + str(self.index) + "[" + str(
                            in_shape[0] * (in_shape[1] + 2 * padding)) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tPadding_Conv2D_" + str(self.index) + "(", "OutPadConv" + str(self.index), "",
                             ""])


                        self.call_function += "\t" + self.type + " OutConv" + str(self.index) + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        source_Conv_cc = self.fxp_inc + "void Conv1D_" + str(self.index) + "(" + self.type + " Input_Conv[" + str(in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(out_shape[0]) + "], " + self.type + " kernel[" + str(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]){\n\tloop_for_channel_" + str(self.index) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(out_shape[0]) + "; n++){\n\t\tloop_for_ap_" + str(self.index) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[1]) + "; y++){\n\t\t\t" + self.type + " s = 0;\n\t\t\tloop_for_fc_" + str(self.index) + ":\n\t\t\tfor (int k = 0; k < " + str(kernel_shape[1]) + "; k++){\n\t\t\t\tloop_for_fa_" + str(self.index) + ":\n\t\t\t\tfor (int j = 0; j < " + str(kernel_shape[2]) + "; j++){\n\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(kernel_shape[2]) + "*n+" + str(kernel_shape[2]) + "*k+j])*(Input_Conv[" + str(in_shape_if_padding[1]) + "*k+j+y*stride]);}\n\t\t\t}\n\t\t\t" + self.act_arr + "\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(
                            self.index) + "(" + self.type + " Input_Conv[" + str(
                            in_shape_if_padding[0] * in_shape_if_padding[1]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"
                    else:
                        self.call_function += "\t" + self.type + " OutConv" + str(self.index) + "[" + str(
                            out_shape[0] *
                            out_shape[1]) + "];\n"
                        source_Conv_cc = self.fxp_inc + "void Conv1D_" + str(
                            self.index) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]){\n\tloop_for_channel_" + str(
                            self.index) + ":\n\tint stride = " + str(stride) + ";\n\tfor (int n = 0; n < " + str(
                            out_shape[0]) + "; n++){\n\t\tloop_for_ap_" + str(
                            self.index) + ":\n\t\tfor (int y = 0; y < " + str(
                            out_shape[1]) + "; y++){\n\t\t\t" + self.type + " s = 0;\n\t\t\tloop_for_fc_" + str(
                            self.index) + ":\n\t\t\tfor (int k = 0; k < " + str(
                            kernel_shape[1]) + "; k++){\n\t\t\t\tloop_for_fa_" + str(
                            self.index) + ":\n\t\t\t\tfor (int j = 0; j < " + str(
                            kernel_shape[2]) + "; j++){\n\t\t\t\t\ts=s+(kernel[" + str(kernel_shape[1]) + "*" + str(
                            kernel_shape[2]) + "*n+" + str(
                            kernel_shape[
                                2]) + "*k+j])*(Input_Conv[" + str(
                            in_shape[1]) + "*k+j+y*stride]);}\n\t\t\t}\n\t\t\t" + self.act_arr + "\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Conv1D_" + str(
                            self.index) + "(" + self.type + " Input_Conv[" + str(
                            in_shape[0] * in_shape[1]) + "]," + self.type + " Output_Conv[" + str(
                            out_shape[0] * out_shape[1]) + "], " + self.type + " bias[" + str(
                            out_shape[0]) + "], " + self.type + " kernel[" + str(
                            kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]);\n"

                    self.full_source_Conv_cc.append(source_Conv_cc)
                    self.full_source_Conv_hh.append(source_Conv_hh)
                    self.full_source_CNN_cc.append(
                        ["\tConv1D_" + str(self.index) + "(", "OutConv" + str(self.index), "&Weights[" + str(
                            self.cnt_param + kernel_shape[0] * kernel_shape[1] * kernel_shape[2]) + "]",
                         "&Weights[" + str(self.cnt_param) + "]"])
                    self.cnt_param += kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + out_shape[0]
                    self.index += 1

                if layer.find("batch_normalization") >= 0:
                    if self.type == "fxp" and self.index_P2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    # 3D, In convolutional layer
                    if (len(self.model.layers[i].input.shape) == 4):
                        in_shape = (self.model.layers[i].input.shape[3], self.model.layers[i].input.shape[1],
                                    self.model.layers[i].input.shape[2])
                        out_shape = (self.model.layers[i].output.shape[3], self.model.layers[i].output.shape[1],
                                     self.model.layers[i].output.shape[2])
                        gamma = self.model.layers[i].get_weights()[0]
                        beta = self.model.layers[i].get_weights()[1]
                        moving_mean = self.model.layers[i].get_weights()[2]
                        moving_variance = self.model.layers[i].get_weights()[3]
                        for k in gamma:
                            self.Weights.append(k)
                        for k in beta:
                            self.Weights.append(k)
                        for k in moving_mean:
                            self.Weights.append(k)
                        for k in moving_variance:
                            self.Weights.append(k)

                        self.call_function += "\t" + str(self.type) + " OutBatchNorm2D_" + str(
                            self.indexBatch) + "[" + str(in_shape[0] * in_shape[1] * in_shape[2]) + "];\n"
                        type_sqrt = ""
                        if self.type == "fxp":
                            source_Conv_cc = "#include <hls_math.h>\n"
                            type_sqrt = "hls::sqrt"
                        else:
                            source_Conv_cc = "#include <cmath>\n"
                            type_sqrt = "sqrt"

                        source_Conv_cc += self.fxp_inc + " void BatchNorm2D_" + str(
                            self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                            self.type) + " Output_BatchNorm[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(self.type) + " gamma[" + str(
                            len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                            self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                            self.type) + " MovVar[" + str(
                            len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(
                            self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(
                            in_shape[0]) + "; i++){\n\t\tfor(int j = 0; j < " + str(
                            in_shape[1] * in_shape[2]) + "; j++){" + "\n\t\t\t Output_BatchNorm[" + str(
                            in_shape[1] * in_shape[2]) + " * i + j] = ((Input_BatchNorm[" + str(in_shape[1] * in_shape[
                            2]) + " * i + j] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps))) * gamma[i] + beta[i];\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = "void BatchNorm2D_" + str(self.indexBatch) + "(" + str(
                            self.type) + " Input_BatchNorm[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(
                            self.type) + " Output_BatchNorm[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + str(self.type) + " gamma[" + str(
                            len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(
                            self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(
                            self.type) + " MovVar[" + str(len(moving_variance)) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tBatchNorm2D_" + str(self.indexBatch) + "(", "OutBatchNorm2D_" + str(self.indexBatch),
                             "&Weights[" + str(self.cnt_param) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma)) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                        self.full_source_Conv_cc.append(source_Conv_cc)
                        self.full_source_Conv_hh.append(source_Conv_hh)
                        self.indexBatch += 1
                        self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

                    # 3D and 2D, in FC
                    if (len(self.model.layers[i].input.shape) == 2):
                        in_shape = self.model.layers[i].input.shape[1]
                        out_shape = self.model.layers[i].output.shape[1]
                        gamma = self.model.layers[i].get_weights()[0]
                        beta = self.model.layers[i].get_weights()[1]
                        moving_mean = self.model.layers[i].get_weights()[2]
                        moving_variance = self.model.layers[i].get_weights()[3]
                        for k in gamma:
                            self.Weights.append(k)
                        for k in beta:
                            self.Weights.append(k)
                        for k in moving_mean:
                            self.Weights.append(k)
                        for k in moving_variance:
                            self.Weights.append(k)
                        self.call_function += "\t" + str(self.type) + " OutBatchNorm2D_" + str(
                            self.indexBatch) + "[" + str(in_shape) + "];\n"

                        type_sqrt = ""
                        if self.type == "fxp":
                            source_Conv_cc = "#include <hls_math.h>\n"
                            type_sqrt = "hls::sqrt"
                        else:
                            source_Conv_cc = "#include <cmath>\n"
                            type_sqrt = "sqrt"

                        source_Conv_cc += self.fxp_inc + " void BatchNorm2D_" + str(self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(in_shape) + "], " + str(self.type) + " Output_BatchNorm[" + str(out_shape) + "], " + str(self.type) + " gamma[" + str(len(gamma)) + "]," + str(self.type) + " beta[" + str(len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(in_shape) + "; i++){\n\t\tOutput_BatchNorm[i] = ((Input_BatchNorm[i] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps)))* gamma[i] + beta[i];\n\t}\n}\n"
                        source_Conv_hh = "void BatchNorm2D_" + str(self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(in_shape) + "], " + str(self.type) + " Output_BatchNorm[" + str(out_shape) + "], " + str(self.type) + " gamma[" + str(len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(len(moving_variance)) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tBatchNorm2D_" + str(self.indexBatch) + "(", "OutBatchNorm2D_" + str(self.indexBatch),
                             "&Weights[" + str(self.cnt_param) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma)) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]",
                             "&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                        self.full_source_Conv_cc.append(source_Conv_cc)
                        self.full_source_Conv_hh.append(source_Conv_hh)
                        self.indexBatch += 1
                        self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

                    # 1D, In Convolutional layer
                    if (len(self.model.layers[i].input.shape) == 3):
                            in_shape = (self.model.layers[i].input.shape[1], self.model.layers[i].input.shape[2])
                            out_shape = (self.model.layers[i].output.shape[1], self.model.layers[i].output.shape[2])
                            gamma = self.model.layers[i].get_weights()[0]
                            beta = self.model.layers[i].get_weights()[1]
                            moving_mean = self.model.layers[i].get_weights()[2]
                            moving_variance = self.model.layers[i].get_weights()[3]
                            for k in gamma:
                                self.Weights.append(k)
                            for k in beta:
                                self.Weights.append(k)
                            for k in moving_mean:
                                self.Weights.append(k)
                            for k in moving_variance:
                                self.Weights.append(k)
                            self.call_function += "\t" + str(self.type) + " OutBatchNorm1D_" + str(
                                self.indexBatch) + "[" + str(in_shape[0] * in_shape[1]) + "];\n"

                            type_sqrt = ""
                            if self.type == "fxp":
                                source_Conv_cc = "#include <hls_math.h>\n"
                                type_sqrt = "hls::sqrt"
                            else:
                                source_Conv_cc = "#include <cmath>\n"
                                type_sqrt = "sqrt"

                            source_Conv_cc += self.fxp_inc + " void BatchNorm1D_" + str(self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(in_shape[0] * in_shape[1]) + "], " + str(self.type) + " Output_BatchNorm[" + str(out_shape[0] * out_shape[1]) + "], " + str(self.type) + " gamma[" + str(len(gamma)) + "]," + str(self.type) + " beta[" + str(len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(len(moving_variance)) + "]) {" + "\n\t" + self.type + " eps = " + str(self.model.layers[i].epsilon) + ";\n\t for(int i = 0; i < " + str(in_shape[1]) + "; i++){\n\t\tfor(int j = 0; j < " + str(in_shape[0]) + "; j++){\n\t\t\tOutput_BatchNorm[" + str(in_shape[0]) + " * i + j] = ((Input_BatchNorm[" + str(in_shape[0]) + " * i + j] - MovMean[i]) / (" + type_sqrt + "(MovVar[i] + eps)))* gamma[i] + beta[i];\n\t\t}\n\t}\n}\n"
                            source_Conv_hh = "void BatchNorm1D_" + str(self.indexBatch) + "(" + str(self.type) + " Input_BatchNorm[" + str(in_shape[0] * in_shape[1]) + "], " + str(self.type) + " Output_BatchNorm[" + str(out_shape[0] * out_shape[1]) + "], " + str(self.type) + " gamma[" + str(len(gamma)) + "], " + str(self.type) + " beta[" + str(len(beta)) + "], " + str(self.type) + " MovMean[" + str(len(moving_mean)) + "], " + str(self.type) + " MovVar[" + str(len(moving_variance)) + "]);\n"
                            self.full_source_CNN_cc.append(["\tBatchNorm1D_" + str(self.indexBatch) + "(","OutBatchNorm1D_" + str(self.indexBatch),"&Weights[" + str(self.cnt_param) + "]","&Weights[" + str(self.cnt_param + len(gamma)) + "]","&Weights[" + str(self.cnt_param + len(gamma) + len(beta)) + "]","&Weights[" + str(self.cnt_param + len(gamma) + len(beta) + len(moving_mean)) + "]"])
                            self.full_source_Conv_cc.append(source_Conv_cc)
                            self.full_source_Conv_hh.append(source_Conv_hh)
                            self.indexBatch += 1
                            self.cnt_param += len(gamma) + len(beta) + len(moving_mean) + len(moving_variance)

                if layer.find("activation") >= 0:
                    # 3D and 1D, FC layer
                    if len(self.model.layers[i].input.shape) == 2:
                        in_shape = self.model.layers[i].input.shape[1]
                        out_shape = self.model.layers[i].output.shape[1]
                        self.call_function += "\t" + str(self.type) + " OutActivation" + str(self.indexActi) + "[" + str(
                            in_shape) + "];\n"
                        source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(
                            self.type) + " Input_Activation[" + str(in_shape) + "], " + str(
                            self.type) + " Output_Activation[" + str(out_shape) + "]){\n\tfor (int i = 0; i < " + str(
                            out_shape) + "; i++){\n\t\tif(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                            self.type) + " Input_Activation[" + str(in_shape) + "], " + str(
                            self.type) + " Output_Activation[" + str(out_shape) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tActivation" + str(self.indexActi) + "(", "OutActivation" + str(self.indexActi)])
                        self.full_source_Conv_cc.append(source_Conv_cc)
                        self.full_source_Conv_hh.append(source_Conv_hh)
                        self.indexActi += 1

                    # 3D , Convolutional Layer
                    if len(self.model.layers[i].input.shape) == 4:
                        in_shape = (self.model.layers[i].input.shape[3],self.model.layers[i].input.shape[1],self.model.layers[i].input.shape[2])
                        out_shape = (self.model.layers[i].output.shape[3],self.model.layers[i].output.shape[1],self.model.layers[i].output.shape[2])
                        self.call_function += "\t" + str(self.type) + " OutActivation" + str(self.indexActi) + "[" + str(
                            out_shape[0]*out_shape[1]*out_shape[2]) + "];\n"
                        source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(self.type) + " Input_Activation[" + str(in_shape[0]*in_shape[1]*in_shape[2]) + "], " + str(self.type) + " Output_Activation[" + str(out_shape[0]*out_shape[1]*out_shape[2]) + "]){\n\tfor (int i = 0; i < " + str(out_shape[0]*out_shape[1]*out_shape[2]) + "; i++){\n\t\tif(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                            self.type) + " Input_Activation[" + str(in_shape[0]*in_shape[1]*in_shape[2]) + "], " + str(
                            self.type) + " Output_Activation[" + str(out_shape[0]*out_shape[1]*out_shape[2]) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tActivation" + str(self.indexActi) + "(", "OutActivation" + str(self.indexActi)])
                        self.full_source_Conv_cc.append(source_Conv_cc)
                        self.full_source_Conv_hh.append(source_Conv_hh)
                        self.indexActi += 1

                    # 1D, Convolutional Layer
                    if len(self.model.layers[i].input.shape) == 3:
                        in_shape = (self.model.layers[i].input.shape[1],self.model.layers[i].input.shape[2])
                        out_shape = (self.model.layers[i].output.shape[1],self.model.layers[i].output.shape[2])
                        self.call_function += "\t" + str(self.type) + " OutActivation" + str(self.indexActi) + "[" + str(
                            out_shape[0]*out_shape[1]) + "];\n"
                        source_Conv_cc = self.fxp_inc + " void Activation" + str(self.indexActi) + "(" + str(self.type) + " Input_Activation[" + str(in_shape[0]*in_shape[1]) + "], " + str(self.type) + " Output_Activation[" + str(out_shape[0]*out_shape[1]) + "]){\n\tfor (int i = 0; i < " + str(out_shape[0]*out_shape[1]) + "; i++){\n\t\tif(Input_Activation[i] > 0){\n\t\t\tOutput_Activation[i] = Input_Activation[i];\n\t\t}else\n\t\t{\n\t\t\tOutput_Activation[i] = 0;\n\t\t}\n\t}\n}\n"
                        source_Conv_hh = self.fxp_inc + "void Activation" + str(self.indexActi) + "(" + str(
                            self.type) + " Input_Activation[" + str(in_shape[0]*in_shape[1]) + "], " + str(
                            self.type) + " Output_Activation[" + str(out_shape[0]*out_shape[1]) + "]);\n"
                        self.full_source_CNN_cc.append(
                            ["\tActivation" + str(self.indexActi) + "(", "OutActivation" + str(self.indexActi)])
                        self.full_source_Conv_cc.append(source_Conv_cc)
                        self.full_source_Conv_hh.append(source_Conv_hh)
                        self.indexActi += 1


                if layer.find("average_pooling2d") >= 0:
                    in_shape = (self.model.layers[i].input.shape[3], self.model.layers[i].input.shape[1],
                                self.model.layers[i].input.shape[2])
                    out_shape = (self.model.layers[i].output.shape[3], self.model.layers[i].output.shape[1],
                                 self.model.layers[i].output.shape[2])
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    if self.type == "fxp" and self.index_P2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    source_Pool_cc = self.fxp_inc + "void average_Pool2D_" + str(
                        self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_AveragePooling[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                        poolSize) + ";\n\tint stride = " + str(
                        strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(out_shape[1]) + "; y++){\n\t\t\t\t" + str(self.type) + " Average = 0;\n\t\t\t\t" + str(self.type) + " Pool_value = 0;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(in_shape[2]) + " * " + "h + " + str(in_shape[2]) + " * stride * z + w + y * stride;\n\t\t\t\t\t\tPool_value += input_AveragePooling[Pool_index];" + "\n\t\t\t\t\t\tAverage = Pool_value / " + str(poolSize*poolSize) + ";\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(out_shape[1]) + " * " + str(out_shape[2]) + " * i + index;\n\t\t\t\toutput_AveragePooling[outIndex] = Average;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                    source_Pool_hh = self.fxp_inc + "void average_Pool2D_" + str(
                        self.index_P2D) + "(" + self.type + " input_AveragePooling[" + str(
                        in_shape[0] * (in_shape[1] + 2)) + "], " + self.type + " output_AveragePooling[" + str(
                        out_shape[0] * out_shape[1]) + "]);\n"
                    self.full_source_Pool_cc.append(source_Pool_cc)
                    self.full_source_Pool_hh.append(source_Pool_hh)

                    if self.config["layers"][i]['config']['padding'] == 'same':
                        self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P2D) + "[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tPadding_Pool2D_" + str(self.index_P2D) + "(", "OutPadPool" + str(self.index_P2D), "",
                             ""])
                    self.call_function += "\t" + self.type + " OutPool" + str(self.index_P2D) + "[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\taverage_Pool2D_" + str(self.index_P2D) + "(", "OutPool" + str(self.index_P2D), "", ""])
                    if self.config["layers"][i]['config']['padding'] == 'same':
                        source_pad_pool_cc = "void Padding_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1] * (in_shape[2] + 2)) + "]){\n\tloop_for_3_channel_pad_" + str(
                            self.index_P2D) + ":\n\tfor (int c = 0; c < " + str(
                            in_shape[1]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                            self.index_P2D) + ":\n\t\tfor (int n = 0; n < " + str(
                            in_shape[1]) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                            self.index_P2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                            in_shape[2] + 2) + "; i++){\n\t\t\t\tif (i < 1 || i >= " + str(
                            in_shape[2] + 2 - 1) + ") output_Pad_Pool[" + str(
                            in_shape[2] + 2) + "*n*c+i]=0; else output_Pad_Pool[" + str(
                            in_shape[1] + 2) + "*n*c+i]=input_Pad_Pool[" + str(
                            in_shape[1]) + "*n*c+i-1];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_pad_pool_hh = "void Padding_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "]);\n"
                        self.full_source_Pool_cc.append(source_pad_pool_cc)
                        self.full_source_Pool_hh.append(source_pad_pool_hh)
                    self.index_P2D += 1


                if layer.find("max_pooling2d") >= 0:
                    in_shape = (self.model.layers[i].input.shape[3], self.model.layers[i].input.shape[1],
                                self.model.layers[i].input.shape[2])
                    out_shape = (self.model.layers[i].output.shape[3], self.model.layers[i].output.shape[1],
                                 self.model.layers[i].output.shape[2])
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    if self.type == "fxp" and self.index_P2D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    source_Pool_cc = self.fxp_inc + "void Max_Pool2D_" + str(
                        self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                        in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_MaxPooling[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "]){\n\tint PoolSize = " + str(
                        poolSize) + ";\n\tint stride = " + str(
                        strides) + ";\n\tint index = 0;\n\tfor (int i = 0; i < " + str(
                        out_shape[0]) + "; i++){\n\t\tindex = 0;\n\t\tfor (int z = 0; z < " + str(
                        out_shape[1]) + "; z++){\n\t\t\tfor (int y = 0; y < " + str(out_shape[
                                                                                        1]) + "; y++){\n\t\t\t\t" + self.type + " max_c = -10;\n\t\t\t\tfor (int h = 0; h < PoolSize; h++){\n\t\t\t\t\tfor (int w = 0; w < PoolSize; w++){\n\t\t\t\t\t\tint Pool_index = " + str(
                        in_shape[1]) + " * " + str(in_shape[2]) + " * i + " + str(in_shape[2]) + " * " + "h + " + str(
                        in_shape[
                            2]) + " * stride * z + w + y * stride;\n\t\t\t\t\t\t" + self.type + " Pool_value = input_MaxPooling[Pool_index];" + "\n\t\t\t\t\t\tif (Pool_value >= max_c) max_c = Pool_value;" + "\n\t\t\t\t\t}\n\t\t\t\t}" + "\n\t\t\t\tint outIndex = " + str(
                        out_shape[1]) + " * " + str(out_shape[
                                                        2]) + " * i + index;\n\t\t\t\toutput_MaxPooling[outIndex] = max_c;\n\t\t\t\tindex++;" + "\n\t\t\t}\n\t\t}\n\t}\n}\n"
                    source_Pool_hh = self.fxp_inc + "void Max_Pool2D_" + str(
                        self.index_P2D) + "(" + self.type + " input_MaxPooling[" + str(
                        in_shape[0] * (in_shape[1] + 2)) + "], " + self.type + " output_MaxPooling[" + str(
                        out_shape[0] * out_shape[1]) + "]);\n"
                    self.full_source_Pool_cc.append(source_Pool_cc)
                    self.full_source_Pool_hh.append(source_Pool_hh)

                    if self.config["layers"][i]['config']['padding'] == 'same':
                        self.call_function += "\t" + self.type + " OutPadPool" + str(self.index_P2D) + "[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "];\n"
                        self.full_source_CNN_cc.append(
                            ["\tPadding_Pool2D_" + str(self.index_P2D) + "(", "OutPadPool" + str(self.index_P2D), "",
                             ""])
                    self.call_function += "\t" + self.type + " OutPool" + str(self.index_P2D) + "[" + str(
                        out_shape[0] * out_shape[1] * out_shape[2]) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tMax_Pool2D_" + str(self.index_P2D) + "(", "OutPool" + str(self.index_P2D), "", ""])
                    if self.config["layers"][i]['config']['padding'] == 'same':
                        source_pad_pool_cc = "void Padding_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "], " + self.type + " output_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1] * (in_shape[2] + 2)) + "]){\n\tloop_for_3_channel_pad_" + str(
                            self.index_P2D) + ":\n\tfor (int c = 0; c < " + str(
                            in_shape[1]) + "; c++){" + "\n\t\tloop_for_channel_pad_" + str(
                            self.index_P2D) + ":\n\t\tfor (int n = 0; n < " + str(
                            in_shape[1]) + "; n++){\n\t\t\tloop_for_weight_pad_" + str(
                            self.index_P2D) + ":\n\t\t\tfor (int i = 0; i < " + str(
                            in_shape[2] + 2) + "; i++){\n\t\t\t\tif (i < 1 || i >= " + str(
                            in_shape[2] + 2 - 1) + ") output_Pad_Pool[" + str(
                            in_shape[2] + 2) + "*n*c+i]=0; else output_Pad_Pool[" + str(
                            in_shape[1] + 2) + "*n*c+i]=input_Pad_Pool[" + str(
                            in_shape[1]) + "*n*c+i-1];\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_pad_pool_hh = "void Padding_Pool2D_" + str(
                            self.index_P2D) + "(" + self.type + " input_Pad_Pool[" + str(
                            in_shape[0] * in_shape[1]) + "], " + self.type + " output_Pad_Pool[" + str(
                            in_shape[0] * (in_shape[1] + 2)) + "]);\n"
                        self.full_source_Pool_cc.append(source_pad_pool_cc)
                        self.full_source_Pool_hh.append(source_pad_pool_hh)
                    self.index_P2D += 1

                # convert max_pooling1d layer into c array that act like an max_pooling1d layer
                if layer.find("max_pooling1d") >= 0:
                    in_shape = (self.model.layers[i].input.shape[2], self.model.layers[i].input.shape[1])
                    out_shape = (self.model.layers[i].output.shape[2], self.model.layers[i].output.shape[1])
                    if self.type == "fxp" and self.index_P == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    source_Pool_cc = self.fxp_inc + "void Max_Pool1D_" + str(self.index_P) + "(" + self.type + " input_MaxPooling[" + str(in_shape[0] * in_shape[1]) + "], " + self.type + " output_MaxPooling[" + str(out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(poolSize) + ";\n\tint stride= " + str(strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(self.index_P) + ":\n\tfor (int z = 0; z < " + str(out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[1]) + "; y++){\n\t\t\t" + self.type + " max = -10;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(in_shape[1]) + " * z + j + y * stride;\n\t\t\t\t" + self.type + " pool_value = input_MaxPooling[pool_index];\n\t\t\t\tif (pool_value > max) max=pool_value;\n\t\t\t}\n\t\t\tint out_index = " + str(out_shape[1]) + " * z + index;\n\t\t\toutput_MaxPooling[out_index]=max;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                    source_Pool_hh = self.fxp_inc + "void Max_Pool1D_" + str(
                        self.index_P) + "(" + self.type + " input_MaxPooling[" + str(
                        in_shape[0] * (in_shape[1])) + "], " + self.type + " output_MaxPooling[" + str(
                        out_shape[0] * out_shape[1]) + "]);\n"
                    self.full_source_Pool_cc.append(source_Pool_cc)
                    self.full_source_Pool_hh.append(source_Pool_hh)

                    self.call_function += "\t" + self.type + " OutPool" + str(self.index_P) + "[" + str(
                        out_shape[0] *
                        out_shape[1]) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tMax_Pool1D_" + str(self.index_P) + "(", "OutPool" + str(self.index_P), "", ""])

                    self.index_P += 1

                if layer.find("average_pooling1d") >= 0:
                    in_shape = (self.model.layers[i].input.shape[2], self.model.layers[i].input.shape[1])
                    out_shape = (self.model.layers[i].output.shape[2], self.model.layers[i].output.shape[1])
                    if self.type == "fxp" and self.index_P == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    strides = self.model.layers[i].strides[0]
                    poolSize = self.model.layers[i].pool_size[0]
                    source_Pool_cc = self.fxp_inc + "void Average_Pool1D_" + str(self.index_P) + "(" + self.type + " input_AveragePooling[" + str(in_shape[0] * in_shape[1]) + "], " + self.type + " output_AveragePooling[" + str(out_shape[0] * out_shape[1]) + "]){\n\tint PoolSize = " + str(poolSize) + ";\n\tint stride= " + str(strides) + ";\n\tint index = 0;\n\tloop_for_channel_pool_" + str(self.index_P) + ":\n\tfor (int z = 0; z < " + str(out_shape[0]) + "; z++){\n\t\tindex = 0;\n\t\tloop_for_weight_pool_" + str(self.index_P) + ":\n\t\tfor (int y = 0; y < " + str(out_shape[1]) + "; y++){\n\t\t\t" + self.type + " Average = 0;\n\t\t\t" + self.type + " pool_value = 0;\n\t\t\tfor (int j = 0; j < PoolSize; j++)\n\t\t\t{\n\t\t\t\tint pool_index = " + str(in_shape[1]) + " * z + j + y * stride;\n\t\t\t\tpool_value += input_AveragePooling[pool_index];\n\t\t\t}\n\t\t\tAverage = pool_value / PoolSize;\n\t\t\tint out_index = " + str(out_shape[1]) + " * z + index;\n\t\t\toutput_AveragePooling[out_index]=Average;\n\t\t\tindex++;\n\t\t}\n\t}\n}\n"
                    source_Pool_hh = self.fxp_inc + "void Average_Pool1D_" + str(
                        self.index_P) + "(" + self.type + " input_AveragePooling[" + str(
                        in_shape[0] * (in_shape[1])) + "], " + self.type + " output_AveragePooling[" + str(
                        out_shape[0] * out_shape[1]) + "]);\n"
                    self.full_source_Pool_cc.append(source_Pool_cc)
                    self.full_source_Pool_hh.append(source_Pool_hh)

                    self.call_function += "\t" + self.type + " OutPool" + str(self.index_P) + "[" + str(
                        out_shape[0] *
                        out_shape[1]) + "];\n"
                    self.full_source_CNN_cc.append(
                        ["\tAverage_Pool1D_" + str(self.index_P) + "(", "OutPool" + str(self.index_P), "", ""])

                    self.index_P += 1

                # convert flatten layer into c array that act like an flatten layer
                if layer.find("flatten") >= 0:
                    # flatten for 2d
                    if len(self.model.layers[i].input.shape) == 3:
                        in_shape = (self.model.layers[i].input.shape[2], self.model.layers[i].input.shape[1])
                        out_shape = self.model.layers[i].output.shape[1]
                        source_Flatten_cc = "void flatten(" + self.type + " input_Flatten[" + str(
                            in_shape[0] * in_shape[1]) + "]," + self.type + " output_Flatten[" + str(
                            out_shape) + "]){\n\tint hs = 0;\n\tloop_for_a_flatten:\n\tfor (int i = 0; i < " + str(
                            in_shape[1]) + "; i++){\n\t\tloop_for_c_flatten:\n\t\tfor (int j = 0; j < " + str(
                            in_shape[0]) + "; j++){\n\t\t\toutput_Flatten[hs] = input_Flatten[" + str(
                            in_shape[1]) + "*j+i];\n\t\t\ths++;\n\t\t}\n\t}\n}\n"
                        source_Flatten_hh = "void flatten(" + self.type + " input_Flatten[" + str(in_shape[0] *
                                                                                                  in_shape[
                                                                                                      1]) + "]," + self.type + " output_Flatten[" + str(
                            out_shape) + "]);\n"
                        self.full_source_Pool_cc.append(source_Flatten_cc)
                        self.full_source_Pool_hh.append(source_Flatten_hh)
                        self.call_function += "\t" + self.type + " OutFlatten[" + str(out_shape) + "];\n"
                        self.full_source_CNN_cc.append(["\tflatten(", "OutFlatten", "", ""])
                    # Flatten for 3d
                    if len(self.model.layers[i].input.shape) == 4:
                        in_shape = (self.model.layers[i].input.shape[3], self.model.layers[i].input.shape[1],
                                    self.model.layers[i].input.shape[2])
                        out_shape = self.model.layers[i].output.shape[1]
                        source_Flatten_cc = "void flatten(" + self.type + " input_Flatten[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_Flatten[" + str(
                            out_shape) + "]){\n\tint hs = 0;\n\tfor (int i = 0; i < " + str(
                            in_shape[1]) + "; i++){\n\t\tfor (int j = 0; j < " + str(
                            in_shape[2]) + "; j++){\n\t\t\tfor (int k = 0; k < " + str(
                            in_shape[0]) + "; k++){\n\t\t\t\toutput_Flatten[hs] = input_Flatten[" + str(
                            in_shape[1]) + " * i + " + str(in_shape[2]) + " * " + str(
                            in_shape[1]) + " * k + j ];\n\t\t\t\ths++;\n\t\t\t}\n\t\t}\n\t}\n}\n"
                        source_Flatten_hh = "void flatten(" + self.type + " input_Flatten[" + str(
                            in_shape[0] * in_shape[1] * in_shape[2]) + "]," + self.type + " output_Flatten[" + str(
                            out_shape) + "]);\n"
                        self.full_source_Pool_cc.append(source_Flatten_cc)
                        self.full_source_Pool_hh.append(source_Flatten_hh)
                        self.call_function += "\t" + self.type + " OutFlatten[" + str(out_shape) + "];\n"
                        self.full_source_CNN_cc.append(["\tflatten(", "OutFlatten", "", ""])

                # convert dense layer into c array that act like an dense layer
                if layer.find("dense") >= 0:
                    weight_shape = self.model.layers[i].get_weights()[0].shape
                    h = self.model.layers[i].get_weights()[0].reshape(weight_shape[0] * weight_shape[1])
                    # print(i - 1)
                    for k in h:
                        self.Weights.append(k)
                    for k in self.model.layers[i].get_weights()[1]:
                        self.Weights.append(k)
                    in_shape = self.model.layers[i].input.shape[1]
                    # if i==len(config["layers"])-1:
                    #     out_shape = 1
                    # else:
                    #     out_shape = self.model.layers[i-1].output_shape[1]
                    out_shape = self.model.layers[i].output.shape[1]
                    activation = self.config["layers"][i]['config']['activation']
                    if self.type == "fxp" and self.index_D == 0:
                        self.fxp_inc = self.fxp_include
                    else:
                        self.fxp_inc = ""
                    if i == len(self.config["layers"]) - 1:
                        self.full_source_CNN_cc.append(["\tDense_" + str(self.index_D) + "(", "OutModel",
                                                        "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                                                        "&Weights[" + str(self.cnt_param) + "]"])
                        if self.choose_only_output:

                            self.out = " &OutModel"
                        else:
                            if out_shape > 1:

                                self.out = " OutModel[" + str(out_shape) + "]"
                            else:

                                self.out = " &OutModel"
                    else:
                        self.full_source_CNN_cc.append(
                            ["\tDense_" + str(self.index_D) + "(", "OutDense" + str(self.index_D),
                             "&Weights[" + str(self.cnt_param + in_shape * out_shape) + "]",
                             "&Weights[" + str(self.cnt_param) + "]"])

                    if i == len(self.config["layers"]) - 1:
                        if self.choose_only_output:
                            self.act_arr = ["\t" + self.type + " out_Dense[" + str(out_shape) + "];\n",
                                            "out_Dense[i]=s+bias[i];"]
                            result_acc = "\t" + self.type + " maxindex=out_Dense[0];\n\toutput_Dense = 0;\n\tloop_detect:\n\tfor (int i=1; i<" + str(
                                out_shape) + "; i++){\n\t\tif (out_Dense[i]>maxindex) {\n\t\t\tmaxindex=out_Dense[i];\n\t\t\toutput_Dense=i;\n\t\t}\n\t}\n"
                            out_dense = self.type + " &output_Dense"

                        else:
                            result_acc = ""
                            if out_shape > 1:
                                out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                                if activation == "relu":
                                    self.act_arr = ["",
                                                    "if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];"]
                                else:
                                    self.act_arr = ["", "output_Dense[i]=s+bias[i];"]
                            else:
                                out_dense = self.type + " &output_Dense"
                                self.act_arr = ["", "output_Dense=s+bias[i];"]
                    else:
                        result_acc = ""
                        out_dense = self.type + " output_Dense[" + str(out_shape) + "]"
                        self.call_function += "\t" + self.type + " OutDense" + str(self.index_D) + "[" + str(
                            out_shape) + "];\n"
                        if activation == "relu":
                            self.act_arr = ["", "if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];"]
                        else:
                            self.act_arr = ["", "output_Dense[i]=s+bias[i];"]
                    source_Dense_cc = self.fxp_inc + "void Dense_" + str(
                        self.index_D) + "(" + self.type + " input_Dense[" + str(
                        in_shape) + "]," + out_dense + "," + self.type + " bias[" + str(
                        out_shape) + "]," + self.type + " weight[" + str(in_shape * out_shape) + "]){\n" + self.act_arr[
                                          0] + "\tloop_for_a_Dense_" + str(
                        self.index_D) + ":\n\tfor (int i = 0; i < " + str(
                        out_shape) + "; i++){\n\t\t" + self.type + " s=0;\n\t\tloop_for_b_Dense_" + str(
                        self.index_D) + ":\n\t\tfor (int j = 0; j < " + str(
                        in_shape) + "; j++){\n\t\t\ts+=input_Dense[j]*weight[j*" + str(
                        out_shape) + "+i];\n\t\t}\n\t\t" + \
                                      self.act_arr[1] + "\n\t}\n" + result_acc + "}\n"
                    source_Dense_hh = self.fxp_inc + "void Dense_" + str(
                        self.index_D) + "(" + self.type + " input_Dense[" + str(
                        in_shape) + "]," + out_dense + "," + self.type + " bias[" + str(
                        out_shape) + "]," + self.type + " weight[" + str(in_shape * out_shape) + "]);\n"
                    self.full_source_Dense_cc.append(source_Dense_cc)
                    self.full_source_Dense_hh.append(source_Dense_hh)
                    self.index_D += 1
                    self.cnt_param += in_shape * out_shape + out_shape

            for i in range(len(self.full_source_CNN_cc)):
                if i == 0:
                    continue
                else:
                    if (len(self.full_source_CNN_cc[i]) > 2):
                        if self.full_source_CNN_cc[i][2] == "":
                            self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][
                                1] + "," + \
                                                  self.full_source_CNN_cc[i][1] + ");\n"
                        else:
                            if (len(self.full_source_CNN_cc[i]) == 6):
                                self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][
                                    1] + "," + \
                                                      self.full_source_CNN_cc[i][1] + "," + self.full_source_CNN_cc[i][
                                                          2] + "," + \
                                                      self.full_source_CNN_cc[i][3] + "," + self.full_source_CNN_cc[i][
                                                          4] + "," + self.full_source_CNN_cc[i][5] + ");\n"
                            if (len(self.full_source_CNN_cc[i]) == 5):
                                self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][
                                    1] + "," + \
                                                      self.full_source_CNN_cc[i][1] + "," + self.full_source_CNN_cc[i][
                                                          2] + "," + \
                                                      self.full_source_CNN_cc[i][3] + "," + self.full_source_CNN_cc[i][
                                                          4] + ");\n"
                            if (len(self.full_source_CNN_cc[i]) == 4):
                                self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][
                                    1] + "," + \
                                                      self.full_source_CNN_cc[i][1] + "," + self.full_source_CNN_cc[i][
                                                          2] + "," + \
                                                      self.full_source_CNN_cc[i][3] + ");\n"
                            if self.resnet :
                                if (len(self.full_source_CNN_cc[i]) == 3):
                                    self.call_function += self.full_source_CNN_cc[i][0] + ", " + self.full_source_CNN_cc[i - 1][1] + ", " + self.full_source_CNN_cc[i][1] + self.full_source_CNN_cc[i - 1][1] + self.full_source_CNN_cc[i][2] + "\n"
                                    self.full_source_CNN_cc[i][1] = self.full_source_CNN_cc[i - 1][1]
                                if (len(self.full_source_CNN_cc[i]) == 7):
                                    self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][1]  + ", " + self.full_source_CNN_cc[i][1] + self.full_source_CNN_cc[i][2] + ", " + self.full_source_CNN_cc[i][3] + self.full_source_CNN_cc[i][4] + ", " + self.full_source_CNN_cc[i][5] + self.full_source_CNN_cc[i][6] + "\n"
                                    self.full_source_CNN_cc[i][1] = self.full_source_CNN_cc[i][3] + self.full_source_CNN_cc[i][4]

                    else:
                        self.call_function += self.full_source_CNN_cc[i][0] + self.full_source_CNN_cc[i - 1][1] + "," + \
                                              self.full_source_CNN_cc[i][1] + ");\n"
            if self.resnet:
                self.source_CNN = "void addArray(" + self.type + "* array1, " + self.type + "* array2, " + self.type + "* result, int size) {\n\tfor (int i = 0; i < size; i++){\n\t\tresult[i] = array1[i] + array2[i];\n\t}\n}\n"

            if len(self.model.layers[1].input.shape) == 4:
                self.source_CNN += "void CNN(" + self.type + " InModel[" + str(self.model.layers[1].input.shape[2] * self.model.layers[1].input.shape[1] *self.model.layers[1].input.shape[3]) + "]," + self.type + self.out + "," + self.type + " Weights[" + str(self.cnt_param) + "]){\n" + self.call_function + "}\n"
                self.source_CNN_hh = "void CNN(" + self.type + " InModel[" + str(
                    self.model.layers[1].input.shape[2] *
                    self.model.layers[1].input.shape[1] * self.model.layers[1].input.shape[
                        3]) + "]," + self.type + self.out + "," + self.type + " Weights[" + str(
                    self.cnt_param) + "]);\n"
            if len(self.model.layers[1].input.shape) == 3:
                self.source_CNN += "void CNN(" + self.type + " InModel[" + str(self.model.layers[1].input_shape[2] * self.model.layers[1].input_shape[1]) + "]," + self.type + self.out + "," + self.type + " Weights[" + str(self.cnt_param) + "]){\n" + self.call_function + "}\n"
                self.source_CNN_hh = "void CNN(" + self.type + " InModel[" + str(
                    self.model.layers[1].input.shape[2] *
                    self.model.layers[1].input.shape[1]) + "]," + self.type + self.out + "," + self.type + " Weights[" + str(
                    self.cnt_param) + "]);\n"



            self.source_CNN_tb = "#include <conio.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <string>\n#include <fstream>\n#include <iostream>\n#include \"CNN.h\"\n#include \"Conv.h\"\n#include \"Pool.h\"\n#include \"Dense.h\"\n#define NumberOfPicture " + "...\n#define d " + "...\n" + self.fxp_inc + "int main(){\n\t" + self.type + " " + self.out.split("&")[-1] + ";\n\t" + self.type + "* Weights = (" + self.type + "*)malloc(" + str(self.cnt_param) + " * sizeof(" + self.type + "));\n\tfloat tmp;\n\tFILE* Weight = fopen(\"Weights.txt\", \"r\");\n\tfor (int i = 0; i < " + str(self.cnt_param) + "; i++){\n\t\tfscanf(Weight, \"%f\", &tmp);\n\t\t*(Weights + i)=tmp;\n\t}\n\tfclose(Weight);" + "\n\t////read Input" + "\n\t" + self.type + "* InModel = (" + self.type + "*)malloc((NumberOfPicture * d * " + str(self.model.layers[1].input.shape[2]) + " * " + str(self.model.layers[1].input.shape[1]) + ") * sizeof(" + self.type + "));\n\tFILE* Input = fopen(\"Input.txt\", \"r\");\n\tfor (int i = 0; i < " + "NumberOfPicture * d * " + str(self.model.layers[1].input.shape[2]) + " * " + str(self.model.layers[1].input.shape[1]) + "; i++){\n\t\tfscanf(Input, \"%f\", &tmp);\n\t\t*(InModel + i)=tmp;\n\t}\n\tfclose(Input);" + "\n\t//Read Label"  + "\n\t" + self.type + "*Label = (" + self.type + "*)malloc((NumberOfPicture) * sizeof(" + self.type + "));" + "\n\tFILE* Output = fopen(\"LabelTXT.txt\", \"r\");\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tfscanf(Output, \"%f\", &tmp);\n\t\t*(Label + i) = tmp;\n\t}\n\tfclose(Output);\n\t" + self.type + " OutArray[NumberOfPicture] = {};\n\t" + self.type + " Image[d * " + str(self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[2]) + "] = {};\n\tfor (int i = 0; i < NumberOfPicture ; i++)\n\t{\n\t\tint startIndex = i * d * " + str(self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[2]) + ";\n\t\tfor (int k = 0; k < d * " + str(self.model.layers[1].input.shape[1]) + " * " + str(self.model.layers[1].input.shape[2]) + "; k++)\n\t\t{\n\t\t\tImage[k] = *(InModel + startIndex + k);\n\t\t}\n\t\tCNN(Image, OutModel, Weights);\n\t\tOutArray[i] = OutModel;\n\t}\n\tfloat countTrue = 0;\n\tfor (int i = 0; i < NumberOfPicture; i++)\n\t{\n\t\tint labelValue = *(Label + i);\n\t\tif (labelValue == OutArray[i])\n\t\t{\n\t\t\tcountTrue = countTrue + 1;\n\t\t}\n\t}\n\tfloat accuracy = (float)((countTrue / NumberOfPicture) * 100);\n\tstd::cout << \"accuracy of Model: \" << accuracy << \"%\\n\";" + "\n\t//std::cout << \"Result: \" <<  OutModel <<  \"\\n\";" + "\n\treturn 0;\n}\n"
            if self.type == "fxp":
                self.fxp_inc = self.fxp_include
            else:
                self.fxp_inc = ""
            print("Successful Converting")

    def WriteCfile(self):
        path = []
        cnt=0
        if len(self.full_source_Conv_cc):
            path.append(self.path_w[0])
            cnt+=1
            with open(self.path_w[0], mode='w') as f:
                for i in range(len(self.full_source_Conv_cc)):
                    for j in range(len(self.full_source_Conv_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Conv_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Conv_hh):
            path.append(self.path_w[1])
            with open(self.path_w[1], mode='w') as f:
                for i in range(len(self.full_source_Conv_hh)):
                    test_j = len(self.full_source_Conv_hh[i].split("\n")) - 1
                    for j in range(len(self.full_source_Conv_hh[i].split("\n")) - 1):
                        test = self.full_source_Conv_hh[i].split("\n")[j] + "\n"
                        f.write(self.full_source_Conv_hh[i].split("\n")[j] + "\n")
        if len(self.full_source_Pool_cc):
            cnt += 1
            path.append(self.path_w[2])
            with open(self.path_w[2], mode='w') as f:
                for i in range(len(self.full_source_Pool_cc)):
                    for j in range(len(self.full_source_Pool_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Pool_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Pool_hh):
            path.append(self.path_w[3])
            with open(self.path_w[3], mode='w') as f:
                for i in range(len(self.full_source_Pool_hh)):
                    for j in range(len(self.full_source_Pool_hh[i].split("\n")) - 1):
                        f.write(self.full_source_Pool_hh[i].split("\n")[j] + "\n")
        if len(self.full_source_Dense_cc):
            cnt += 1
            path.append(self.path_w[4])
            with open(self.path_w[4], mode='w') as f:
                for i in range(len(self.full_source_Dense_cc)):
                    for j in range(len(self.full_source_Dense_cc[i].split("\n")) - 1):
                        f.write(self.full_source_Dense_cc[i].split("\n")[j] + "\n")
        if len(self.full_source_Dense_hh):
            path.append(self.path_w[5])
            with open(self.path_w[5], mode='w') as f:
                for i in range(len(self.full_source_Dense_hh)):
                    for j in range(len(self.full_source_Dense_hh[i].split("\n")) - 1):
                        f.write(self.full_source_Dense_hh[i].split("\n")[j] + "\n")
        if len((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")) - 1 and cnt:
            path.append(self.path_w[6])
            with open(self.path_w[6], mode='w') as f:
                for j in range(len((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")) - 1):
                    f.write((self.CNN_include + self.fxp_inc + self.source_CNN).split("\n")[j] + "\n")
        if len((self.fxp_inc + self.source_CNN_hh).split("\n")) - 1 and cnt:
            path.append(self.path_w[7])
            with open(self.path_w[7], mode='w') as f:
                for j in range(len((self.fxp_inc + self.source_CNN_hh).split("\n")) - 1):
                    f.write((self.fxp_inc + self.source_CNN_hh).split("\n")[j] + "\n")
            self.Weights = np.array(self.Weights)
        if len((self.source_CNN_tb).split("\n")) - 1 and cnt:
            path.append(self.path_w[8])
            with open(self.path_w[8], mode='w') as f:
                for j in range(len((self.source_CNN_tb).split("\n")) - 1):
                    f.write((self.source_CNN_tb).split("\n")[j] + "\n")

        if len(path):
            print("Successful Writing file")
            print("There are ", str(len(path)), " file(s) such as:")
            for name in path:
                print("\t", name)
        else:
            print("Py2C do not support your model!!!")

    def del_one_file(self, name):
        if os.path.exists(name):
            try:
                # Delete the file
                os.remove(name)
                print(f"The file {name} has been deleted.")
            except Exception as e:
                print(f"Unable to delete the file {name}: {e}")
        else:
            print(f"The file {name} does not exist.")

    def del_any_file(self, name_arr):
        for name in name_arr:
            self.del_one_file(name)

    def del_all_file(self):
        for name in self.path_w:
            self.del_one_file(name)

    def Write_Float_Weights_File(self, path="Float_Weights.txt"):
        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        with open(path, mode='w') as f:
            for i in self.Weights:
                f.write(str(i) + " ")
        print("Successful Writing Float Weights file!!!")

    def Write_IEEE754_32bits_Weights_File(self, path="IEEE754_32bits_Weights.txt"):
        def float_to_binary32(f):
            # Chuyển đổi số float thành binary32
            packed = struct.pack('!f', f)

            # Chuyển đổi binary32 thành chuỗi nhị phân
            binary = ''.join(format(byte, '08b') for byte in packed)

            return binary

        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        with open(path, mode='w') as f:
            for i in range(len(self.Weights)):
                # print(i)
                binary_num = float_to_binary32(self.Weights[i])
                decimal_num = int(binary_num, 2)
                f.write(str(decimal_num) + " ")
        print("Successful Writing IEEE754 32bits Weights file!!!")

    def Write_FixedPoint_Weights_File(self, path="FixedPoint_Weights.txt"):
        def float_to_binary32(f):
            packed = struct.pack('!f', f)

            binary = ''.join(format(byte, '08b') for byte in packed)

            return binary

        def binary32_to_fixedpoint(f):
            f32 = int(float_to_binary32(f), 2)
            Nbitbot = self.fxp_para[0] - 2
            Nbittop = self.fxp_para[1] - 2
            Limitbot = 2 ** (23 - Nbitbot)
            M = ((f32 // Limitbot) & (2 ** Nbitbot - 1)) + 2 ** Nbitbot  # and add Nbittop bit(s) 0 state before

            E = int((f32 & 2139095040) / (2 ** 23))
            S = f32 & 2147483648
            Es = E - 127
            F = int(M * (2 ** Es))
            Nbitbotaf = Nbittop + Nbitbot + 1 - self.fxp_para[0] + 1

            G = F // (2 ** Nbitbotaf) & (2 ** (Nbittop + Nbitbot + 1) - 1)
            if S == 1:
                G = 2 ** self.fxp_para[0] - G
            return G

        assert len(self.Weights) != 0, "Converting has not implemented yet!!! Please Run convert2C in Py2C"
        assert self.fxp_para is not None, "Fxp parameter has not set yet!! Please run set_Fxp_Param in Py2C"
        with open(path, mode='w') as f:
            for i in range(len(self.Weights)):
                binary_num = binary32_to_fixedpoint(self.Weights[i])
                f.write(str(binary_num) + " ")
        print("Successful Writing Fixed Point Weights file!!!")
