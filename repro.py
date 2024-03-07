
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config






isolate_fails_code_str = None



# torch version: 2.2.1+cu121
# torch cuda version: 12.1
# torch git version: 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194):
        sub = torch.ops.aten.sub.Tensor(primals_191, primals_189);  primals_191 = None
        div = torch.ops.aten.div.Tensor(sub, primals_190);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(primals_192, primals_189);  primals_192 = None
        div_1 = torch.ops.aten.div.Tensor(sub_1, primals_190);  sub_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convolution = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [4, 4], [0, 0], [1, 1], False, [0, 0], 1)
        permute = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(permute, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_3, [3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = None
        mul = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = None
        permute_1 = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(permute_1, torch.bfloat16)
        convolution_1 = torch.ops.aten.convolution.default(convert_element_type_6, convert_element_type_5, convert_element_type_4, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_2 = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(permute_2, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_7, [3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_7, getitem_3);  convert_element_type_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_31);  mul_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, primals_32);  mul_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view = torch.ops.aten.view.default(convert_element_type_10, [8192, 96]);  convert_element_type_10 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm = torch.ops.aten.addmm.default(convert_element_type_8, view, permute_3)
        view_1 = torch.ops.aten.view.default(addmm, [32, 16, 16, 384])
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_15, [8192, 384]);  convert_element_type_15 = None
        permute_4 = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_1 = torch.ops.aten.addmm.default(convert_element_type_16, view_2, permute_4)
        view_3 = torch.ops.aten.view.default(addmm_1, [32, 16, 16, 96])
        permute_5 = torch.ops.aten.permute.default(view_3, [0, 3, 1, 2]);  view_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(primals_3, permute_5);  permute_5 = None
        add_5 = torch.ops.aten.add.Tensor(mul_7, permute_1);  mul_7 = permute_1 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convolution_2 = torch.ops.aten.convolution.default(convert_element_type_23, convert_element_type_22, convert_element_type_21, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_6 = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(permute_6, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [3], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_5);  convert_element_type_24 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, primals_39);  mul_8 = None
        add_7 = torch.ops.aten.add.Tensor(mul_9, primals_40);  mul_9 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_4 = torch.ops.aten.view.default(convert_element_type_27, [8192, 96]);  convert_element_type_27 = None
        permute_7 = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        addmm_2 = torch.ops.aten.addmm.default(convert_element_type_25, view_4, permute_7)
        view_5 = torch.ops.aten.view.default(addmm_2, [32, 16, 16, 384])
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.5)
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.7071067811865476);  convert_element_type_31 = None
        erf_1 = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_8 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_10, add_8);  mul_10 = add_8 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        view_6 = torch.ops.aten.view.default(convert_element_type_32, [8192, 384]);  convert_element_type_32 = None
        permute_8 = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_3 = torch.ops.aten.addmm.default(convert_element_type_33, view_6, permute_8)
        view_7 = torch.ops.aten.view.default(addmm_3, [32, 16, 16, 96])
        permute_9 = torch.ops.aten.permute.default(view_7, [0, 3, 1, 2]);  view_7 = None
        mul_13 = torch.ops.aten.mul.Tensor(primals_4, permute_9);  permute_9 = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(76, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_75 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt = torch.ops.aten.lt.Scalar(inductor_random_default_75, 0.9941176470588236);  inductor_random_default_75 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(lt, torch.float32)
        div_2 = torch.ops.aten.div.Tensor(convert_element_type_38, 0.9941176470588236);  convert_element_type_38 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, div_2);  div_2 = None
        add_9 = torch.ops.aten.add.Tensor(mul_14, add_5);  mul_14 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16)
        convolution_3 = torch.ops.aten.convolution.default(convert_element_type_41, convert_element_type_40, convert_element_type_39, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_10 = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
        convert_element_type_42 = torch.ops.prims.convert_element_type.default(permute_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_42, [3], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_7);  convert_element_type_42 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, primals_47);  mul_15 = None
        add_11 = torch.ops.aten.add.Tensor(mul_16, primals_48);  mul_16 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        view_8 = torch.ops.aten.view.default(convert_element_type_45, [8192, 96]);  convert_element_type_45 = None
        permute_11 = torch.ops.aten.permute.default(convert_element_type_44, [1, 0]);  convert_element_type_44 = None
        addmm_4 = torch.ops.aten.addmm.default(convert_element_type_43, view_8, permute_11)
        view_9 = torch.ops.aten.view.default(addmm_4, [32, 16, 16, 384])
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.5)
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.7071067811865476);  convert_element_type_49 = None
        erf_2 = torch.ops.aten.erf.default(mul_18);  mul_18 = None
        add_12 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_17, add_12);  mul_17 = add_12 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        view_10 = torch.ops.aten.view.default(convert_element_type_50, [8192, 384]);  convert_element_type_50 = None
        permute_12 = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        addmm_5 = torch.ops.aten.addmm.default(convert_element_type_51, view_10, permute_12)
        view_11 = torch.ops.aten.view.default(addmm_5, [32, 16, 16, 96])
        permute_13 = torch.ops.aten.permute.default(view_11, [0, 3, 1, 2]);  view_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(primals_5, permute_13);  permute_13 = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_74 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(inductor_random_default_74, 0.9882352941176471);  inductor_random_default_74 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(lt_1, torch.float32)
        div_3 = torch.ops.aten.div.Tensor(convert_element_type_56, 0.9882352941176471);  convert_element_type_56 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, div_3);  mul_20 = div_3 = None
        add_13 = torch.ops.aten.add.Tensor(mul_21, add_9);  mul_21 = add_9 = None
        permute_15 = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1])
        var_mean_4 = torch.ops.aten.var_mean.correction(permute_15, [3], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(permute_15, getitem_9);  permute_15 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, primals_6);  mul_22 = None
        add_15 = torch.ops.aten.add.Tensor(mul_23, primals_7);  mul_23 = None
        permute_16 = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(permute_16, torch.bfloat16);  permute_16 = None
        convolution_4 = torch.ops.aten.convolution.default(convert_element_type_59, convert_element_type_58, convert_element_type_57, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convolution_5 = torch.ops.aten.convolution.default(convolution_4, convert_element_type_61, convert_element_type_60, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_17 = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(permute_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [3], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_7 = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_11);  convert_element_type_62 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, primals_57);  mul_24 = None
        add_17 = torch.ops.aten.add.Tensor(mul_25, primals_58);  mul_25 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_12 = torch.ops.aten.view.default(convert_element_type_65, [2048, 192]);  convert_element_type_65 = None
        permute_18 = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        addmm_6 = torch.ops.aten.addmm.default(convert_element_type_63, view_12, permute_18)
        view_13 = torch.ops.aten.view.default(addmm_6, [32, 8, 8, 768])
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_26 = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.5)
        mul_27 = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.7071067811865476);  convert_element_type_69 = None
        erf_3 = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_18 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, add_18);  mul_26 = add_18 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_70, [2048, 768]);  convert_element_type_70 = None
        permute_19 = torch.ops.aten.permute.default(convert_element_type_72, [1, 0]);  convert_element_type_72 = None
        addmm_7 = torch.ops.aten.addmm.default(convert_element_type_71, view_14, permute_19)
        view_15 = torch.ops.aten.view.default(addmm_7, [32, 8, 8, 192])
        permute_20 = torch.ops.aten.permute.default(view_15, [0, 3, 1, 2]);  view_15 = None
        mul_29 = torch.ops.aten.mul.Tensor(primals_8, permute_20);  permute_20 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_73 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        lt_2 = torch.ops.aten.lt.Scalar(inductor_random_default_73, 0.9823529411764705);  inductor_random_default_73 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(lt_2, torch.float32)
        div_4 = torch.ops.aten.div.Tensor(convert_element_type_76, 0.9823529411764705);  convert_element_type_76 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, div_4);  mul_29 = div_4 = None
        add_19 = torch.ops.aten.add.Tensor(mul_30, convolution_4);  mul_30 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_78 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16)
        convolution_6 = torch.ops.aten.convolution.default(convert_element_type_79, convert_element_type_78, convert_element_type_77, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_21 = torch.ops.aten.permute.default(convolution_6, [0, 2, 3, 1]);  convolution_6 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(permute_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_80, [3], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8 = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_13);  convert_element_type_80 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, primals_65);  mul_31 = None
        add_21 = torch.ops.aten.add.Tensor(mul_32, primals_66);  mul_32 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_83, [2048, 192]);  convert_element_type_83 = None
        permute_22 = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        addmm_8 = torch.ops.aten.addmm.default(convert_element_type_81, view_16, permute_22)
        view_17 = torch.ops.aten.view.default(addmm_8, [32, 8, 8, 768])
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.5)
        mul_34 = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.7071067811865476);  convert_element_type_87 = None
        erf_4 = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_22 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_33, add_22);  mul_33 = add_22 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        view_18 = torch.ops.aten.view.default(convert_element_type_88, [2048, 768]);  convert_element_type_88 = None
        permute_23 = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        addmm_9 = torch.ops.aten.addmm.default(convert_element_type_89, view_18, permute_23)
        view_19 = torch.ops.aten.view.default(addmm_9, [32, 8, 8, 192])
        permute_24 = torch.ops.aten.permute.default(view_19, [0, 3, 1, 2]);  view_19 = None
        mul_36 = torch.ops.aten.mul.Tensor(primals_9, permute_24);  permute_24 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_72 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        lt_3 = torch.ops.aten.lt.Scalar(inductor_random_default_72, 0.9764705882352941);  inductor_random_default_72 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(lt_3, torch.float32)
        div_5 = torch.ops.aten.div.Tensor(convert_element_type_94, 0.9764705882352941);  convert_element_type_94 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, div_5);  mul_36 = div_5 = None
        add_23 = torch.ops.aten.add.Tensor(mul_37, add_19);  mul_37 = add_19 = None
        convert_element_type_95 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16)
        convolution_7 = torch.ops.aten.convolution.default(convert_element_type_97, convert_element_type_96, convert_element_type_95, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_25 = torch.ops.aten.permute.default(convolution_7, [0, 2, 3, 1]);  convolution_7 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(permute_25, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_98, [3], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_9 = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_15);  convert_element_type_98 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, primals_73);  mul_38 = None
        add_25 = torch.ops.aten.add.Tensor(mul_39, primals_74);  mul_39 = None
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        view_20 = torch.ops.aten.view.default(convert_element_type_101, [2048, 192]);  convert_element_type_101 = None
        permute_26 = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        addmm_10 = torch.ops.aten.addmm.default(convert_element_type_99, view_20, permute_26)
        view_21 = torch.ops.aten.view.default(addmm_10, [32, 8, 8, 768])
        convert_element_type_105 = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.7071067811865476);  convert_element_type_105 = None
        erf_5 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_26 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_26);  mul_40 = add_26 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_106, [2048, 768]);  convert_element_type_106 = None
        permute_27 = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_11 = torch.ops.aten.addmm.default(convert_element_type_107, view_22, permute_27)
        view_23 = torch.ops.aten.view.default(addmm_11, [32, 8, 8, 192])
        permute_28 = torch.ops.aten.permute.default(view_23, [0, 3, 1, 2]);  view_23 = None
        mul_43 = torch.ops.aten.mul.Tensor(primals_10, permute_28);  permute_28 = None
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_71 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        lt_4 = torch.ops.aten.lt.Scalar(inductor_random_default_71, 0.9705882352941176);  inductor_random_default_71 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(lt_4, torch.float32)
        div_6 = torch.ops.aten.div.Tensor(convert_element_type_112, 0.9705882352941176);  convert_element_type_112 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, div_6);  mul_43 = div_6 = None
        add_27 = torch.ops.aten.add.Tensor(mul_44, add_23);  mul_44 = add_23 = None
        permute_30 = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1])
        var_mean_8 = torch.ops.aten.var_mean.correction(permute_30, [3], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_10 = torch.ops.aten.sub.Tensor(permute_30, getitem_17);  permute_30 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, primals_11);  mul_45 = None
        add_29 = torch.ops.aten.add.Tensor(mul_46, primals_12);  mul_46 = None
        permute_31 = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(permute_31, torch.bfloat16);  permute_31 = None
        convolution_8 = torch.ops.aten.convolution.default(convert_element_type_115, convert_element_type_114, convert_element_type_113, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_117 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convolution_9 = torch.ops.aten.convolution.default(convolution_8, convert_element_type_117, convert_element_type_116, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_32 = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
        convert_element_type_118 = torch.ops.prims.convert_element_type.default(permute_32, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_118, [3], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_11 = torch.ops.aten.sub.Tensor(convert_element_type_118, getitem_19);  convert_element_type_118 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, primals_83);  mul_47 = None
        add_31 = torch.ops.aten.add.Tensor(mul_48, primals_84);  mul_48 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_24 = torch.ops.aten.view.default(convert_element_type_121, [512, 384]);  convert_element_type_121 = None
        permute_33 = torch.ops.aten.permute.default(convert_element_type_120, [1, 0]);  convert_element_type_120 = None
        addmm_12 = torch.ops.aten.addmm.default(convert_element_type_119, view_24, permute_33)
        view_25 = torch.ops.aten.view.default(addmm_12, [32, 4, 4, 1536])
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_125, 0.5)
        mul_50 = torch.ops.aten.mul.Tensor(convert_element_type_125, 0.7071067811865476);  convert_element_type_125 = None
        erf_6 = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_32 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_49, add_32);  mul_49 = add_32 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_128 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        view_26 = torch.ops.aten.view.default(convert_element_type_126, [512, 1536]);  convert_element_type_126 = None
        permute_34 = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        addmm_13 = torch.ops.aten.addmm.default(convert_element_type_127, view_26, permute_34)
        view_27 = torch.ops.aten.view.default(addmm_13, [32, 4, 4, 384])
        permute_35 = torch.ops.aten.permute.default(view_27, [0, 3, 1, 2]);  view_27 = None
        mul_52 = torch.ops.aten.mul.Tensor(primals_13, permute_35);  permute_35 = None
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_70 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        lt_5 = torch.ops.aten.lt.Scalar(inductor_random_default_70, 0.9647058823529412);  inductor_random_default_70 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(lt_5, torch.float32)
        div_7 = torch.ops.aten.div.Tensor(convert_element_type_132, 0.9647058823529412);  convert_element_type_132 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, div_7);  mul_52 = div_7 = None
        add_33 = torch.ops.aten.add.Tensor(mul_53, convolution_8);  mul_53 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        convolution_10 = torch.ops.aten.convolution.default(convert_element_type_135, convert_element_type_134, convert_element_type_133, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_36 = torch.ops.aten.permute.default(convolution_10, [0, 2, 3, 1]);  convolution_10 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(permute_36, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_136, [3], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12 = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_21);  convert_element_type_136 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, primals_91);  mul_54 = None
        add_35 = torch.ops.aten.add.Tensor(mul_55, primals_92);  mul_55 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convert_element_type_138 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        view_28 = torch.ops.aten.view.default(convert_element_type_139, [512, 384]);  convert_element_type_139 = None
        permute_37 = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        addmm_14 = torch.ops.aten.addmm.default(convert_element_type_137, view_28, permute_37)
        view_29 = torch.ops.aten.view.default(addmm_14, [32, 4, 4, 1536])
        convert_element_type_143 = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_143, 0.5)
        mul_57 = torch.ops.aten.mul.Tensor(convert_element_type_143, 0.7071067811865476);  convert_element_type_143 = None
        erf_7 = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_36 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_56, add_36);  mul_56 = add_36 = None
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        view_30 = torch.ops.aten.view.default(convert_element_type_144, [512, 1536]);  convert_element_type_144 = None
        permute_38 = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        addmm_15 = torch.ops.aten.addmm.default(convert_element_type_145, view_30, permute_38)
        view_31 = torch.ops.aten.view.default(addmm_15, [32, 4, 4, 384])
        permute_39 = torch.ops.aten.permute.default(view_31, [0, 3, 1, 2]);  view_31 = None
        mul_59 = torch.ops.aten.mul.Tensor(primals_14, permute_39);  permute_39 = None
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_69 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        lt_6 = torch.ops.aten.lt.Scalar(inductor_random_default_69, 0.9588235294117647);  inductor_random_default_69 = None
        convert_element_type_150 = torch.ops.prims.convert_element_type.default(lt_6, torch.float32)
        div_8 = torch.ops.aten.div.Tensor(convert_element_type_150, 0.9588235294117647);  convert_element_type_150 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, div_8);  mul_59 = div_8 = None
        add_37 = torch.ops.aten.add.Tensor(mul_60, add_33);  mul_60 = add_33 = None
        convert_element_type_151 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        convolution_11 = torch.ops.aten.convolution.default(convert_element_type_153, convert_element_type_152, convert_element_type_151, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_40 = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(permute_40, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_154, [3], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_13 = torch.ops.aten.sub.Tensor(convert_element_type_154, getitem_23);  convert_element_type_154 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, primals_99);  mul_61 = None
        add_39 = torch.ops.aten.add.Tensor(mul_62, primals_100);  mul_62 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_156 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_157 = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_157, [512, 384]);  convert_element_type_157 = None
        permute_41 = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        addmm_16 = torch.ops.aten.addmm.default(convert_element_type_155, view_32, permute_41)
        view_33 = torch.ops.aten.view.default(addmm_16, [32, 4, 4, 1536])
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.5)
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.7071067811865476);  convert_element_type_161 = None
        erf_8 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_63, add_40);  mul_63 = add_40 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_164 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_162, [512, 1536]);  convert_element_type_162 = None
        permute_42 = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_17 = torch.ops.aten.addmm.default(convert_element_type_163, view_34, permute_42)
        view_35 = torch.ops.aten.view.default(addmm_17, [32, 4, 4, 384])
        permute_43 = torch.ops.aten.permute.default(view_35, [0, 3, 1, 2]);  view_35 = None
        mul_66 = torch.ops.aten.mul.Tensor(primals_15, permute_43);  permute_43 = None
        inductor_lookup_seed_default_7 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_68 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        lt_7 = torch.ops.aten.lt.Scalar(inductor_random_default_68, 0.9529411764705882);  inductor_random_default_68 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(lt_7, torch.float32)
        div_9 = torch.ops.aten.div.Tensor(convert_element_type_168, 0.9529411764705882);  convert_element_type_168 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, div_9);  mul_66 = div_9 = None
        add_41 = torch.ops.aten.add.Tensor(mul_67, add_37);  mul_67 = add_37 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_170 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16)
        convolution_12 = torch.ops.aten.convolution.default(convert_element_type_171, convert_element_type_170, convert_element_type_169, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_44 = torch.ops.aten.permute.default(convolution_12, [0, 2, 3, 1]);  convolution_12 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(permute_44, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_172, [3], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_14 = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_25);  convert_element_type_172 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, primals_107);  mul_68 = None
        add_43 = torch.ops.aten.add.Tensor(mul_69, primals_108);  mul_69 = None
        convert_element_type_173 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_174 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_175, [512, 384]);  convert_element_type_175 = None
        permute_45 = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_18 = torch.ops.aten.addmm.default(convert_element_type_173, view_36, permute_45)
        view_37 = torch.ops.aten.view.default(addmm_18, [32, 4, 4, 1536])
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.5)
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.7071067811865476);  convert_element_type_179 = None
        erf_9 = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_44 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_70, add_44);  mul_70 = add_44 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_38 = torch.ops.aten.view.default(convert_element_type_180, [512, 1536]);  convert_element_type_180 = None
        permute_46 = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        addmm_19 = torch.ops.aten.addmm.default(convert_element_type_181, view_38, permute_46)
        view_39 = torch.ops.aten.view.default(addmm_19, [32, 4, 4, 384])
        permute_47 = torch.ops.aten.permute.default(view_39, [0, 3, 1, 2]);  view_39 = None
        mul_73 = torch.ops.aten.mul.Tensor(primals_16, permute_47);  permute_47 = None
        inductor_lookup_seed_default_8 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_67 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        lt_8 = torch.ops.aten.lt.Scalar(inductor_random_default_67, 0.9470588235294117);  inductor_random_default_67 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(lt_8, torch.float32)
        div_10 = torch.ops.aten.div.Tensor(convert_element_type_186, 0.9470588235294117);  convert_element_type_186 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, div_10);  mul_73 = div_10 = None
        add_45 = torch.ops.aten.add.Tensor(mul_74, add_41);  mul_74 = add_41 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16)
        convolution_13 = torch.ops.aten.convolution.default(convert_element_type_189, convert_element_type_188, convert_element_type_187, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_48 = torch.ops.aten.permute.default(convolution_13, [0, 2, 3, 1]);  convolution_13 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(permute_48, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_190, [3], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_15 = torch.ops.aten.sub.Tensor(convert_element_type_190, getitem_27);  convert_element_type_190 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, primals_115);  mul_75 = None
        add_47 = torch.ops.aten.add.Tensor(mul_76, primals_116);  mul_76 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        view_40 = torch.ops.aten.view.default(convert_element_type_193, [512, 384]);  convert_element_type_193 = None
        permute_49 = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        addmm_20 = torch.ops.aten.addmm.default(convert_element_type_191, view_40, permute_49)
        view_41 = torch.ops.aten.view.default(addmm_20, [32, 4, 4, 1536])
        convert_element_type_197 = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_77 = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.5)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.7071067811865476);  convert_element_type_197 = None
        erf_10 = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_48 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_77, add_48);  mul_77 = add_48 = None
        convert_element_type_198 = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        view_42 = torch.ops.aten.view.default(convert_element_type_198, [512, 1536]);  convert_element_type_198 = None
        permute_50 = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_21 = torch.ops.aten.addmm.default(convert_element_type_199, view_42, permute_50)
        view_43 = torch.ops.aten.view.default(addmm_21, [32, 4, 4, 384])
        permute_51 = torch.ops.aten.permute.default(view_43, [0, 3, 1, 2]);  view_43 = None
        mul_80 = torch.ops.aten.mul.Tensor(primals_17, permute_51);  permute_51 = None
        inductor_lookup_seed_default_9 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_66 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        lt_9 = torch.ops.aten.lt.Scalar(inductor_random_default_66, 0.9411764705882353);  inductor_random_default_66 = None
        convert_element_type_204 = torch.ops.prims.convert_element_type.default(lt_9, torch.float32)
        div_11 = torch.ops.aten.div.Tensor(convert_element_type_204, 0.9411764705882353);  convert_element_type_204 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, div_11);  mul_80 = div_11 = None
        add_49 = torch.ops.aten.add.Tensor(mul_81, add_45);  mul_81 = add_45 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16)
        convolution_14 = torch.ops.aten.convolution.default(convert_element_type_207, convert_element_type_206, convert_element_type_205, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_52 = torch.ops.aten.permute.default(convolution_14, [0, 2, 3, 1]);  convolution_14 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(permute_52, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_208, [3], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_16 = torch.ops.aten.sub.Tensor(convert_element_type_208, getitem_29);  convert_element_type_208 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, primals_123);  mul_82 = None
        add_51 = torch.ops.aten.add.Tensor(mul_83, primals_124);  mul_83 = None
        convert_element_type_209 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_211, [512, 384]);  convert_element_type_211 = None
        permute_53 = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        addmm_22 = torch.ops.aten.addmm.default(convert_element_type_209, view_44, permute_53)
        view_45 = torch.ops.aten.view.default(addmm_22, [32, 4, 4, 1536])
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_215, 0.5)
        mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_215, 0.7071067811865476);  convert_element_type_215 = None
        erf_11 = torch.ops.aten.erf.default(mul_85);  mul_85 = None
        add_52 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_84, add_52);  mul_84 = add_52 = None
        convert_element_type_216 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        view_46 = torch.ops.aten.view.default(convert_element_type_216, [512, 1536]);  convert_element_type_216 = None
        permute_54 = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_23 = torch.ops.aten.addmm.default(convert_element_type_217, view_46, permute_54)
        view_47 = torch.ops.aten.view.default(addmm_23, [32, 4, 4, 384])
        permute_55 = torch.ops.aten.permute.default(view_47, [0, 3, 1, 2]);  view_47 = None
        mul_87 = torch.ops.aten.mul.Tensor(primals_18, permute_55);  permute_55 = None
        inductor_lookup_seed_default_10 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_65 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        lt_10 = torch.ops.aten.lt.Scalar(inductor_random_default_65, 0.9352941176470588);  inductor_random_default_65 = None
        convert_element_type_222 = torch.ops.prims.convert_element_type.default(lt_10, torch.float32)
        div_12 = torch.ops.aten.div.Tensor(convert_element_type_222, 0.9352941176470588);  convert_element_type_222 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, div_12);  mul_87 = div_12 = None
        add_53 = torch.ops.aten.add.Tensor(mul_88, add_49);  mul_88 = add_49 = None
        convert_element_type_223 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16)
        convolution_15 = torch.ops.aten.convolution.default(convert_element_type_225, convert_element_type_224, convert_element_type_223, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_56 = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(permute_56, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_226, [3], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_17 = torch.ops.aten.sub.Tensor(convert_element_type_226, getitem_31);  convert_element_type_226 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, primals_131);  mul_89 = None
        add_55 = torch.ops.aten.add.Tensor(mul_90, primals_132);  mul_90 = None
        convert_element_type_227 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_228 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_229, [512, 384]);  convert_element_type_229 = None
        permute_57 = torch.ops.aten.permute.default(convert_element_type_228, [1, 0]);  convert_element_type_228 = None
        addmm_24 = torch.ops.aten.addmm.default(convert_element_type_227, view_48, permute_57)
        view_49 = torch.ops.aten.view.default(addmm_24, [32, 4, 4, 1536])
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        mul_91 = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.5)
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.7071067811865476);  convert_element_type_233 = None
        erf_12 = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_56 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_91, add_56);  mul_91 = add_56 = None
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        view_50 = torch.ops.aten.view.default(convert_element_type_234, [512, 1536]);  convert_element_type_234 = None
        permute_58 = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        addmm_25 = torch.ops.aten.addmm.default(convert_element_type_235, view_50, permute_58)
        view_51 = torch.ops.aten.view.default(addmm_25, [32, 4, 4, 384])
        permute_59 = torch.ops.aten.permute.default(view_51, [0, 3, 1, 2]);  view_51 = None
        mul_94 = torch.ops.aten.mul.Tensor(primals_19, permute_59);  permute_59 = None
        inductor_lookup_seed_default_11 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_64 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        lt_11 = torch.ops.aten.lt.Scalar(inductor_random_default_64, 0.9294117647058824);  inductor_random_default_64 = None
        convert_element_type_240 = torch.ops.prims.convert_element_type.default(lt_11, torch.float32)
        div_13 = torch.ops.aten.div.Tensor(convert_element_type_240, 0.9294117647058824);  convert_element_type_240 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, div_13);  mul_94 = div_13 = None
        add_57 = torch.ops.aten.add.Tensor(mul_95, add_53);  mul_95 = add_53 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_242 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_243 = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        convolution_16 = torch.ops.aten.convolution.default(convert_element_type_243, convert_element_type_242, convert_element_type_241, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_60 = torch.ops.aten.permute.default(convolution_16, [0, 2, 3, 1]);  convolution_16 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(permute_60, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_244, [3], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_18 = torch.ops.aten.sub.Tensor(convert_element_type_244, getitem_33);  convert_element_type_244 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, primals_139);  mul_96 = None
        add_59 = torch.ops.aten.add.Tensor(mul_97, primals_140);  mul_97 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        view_52 = torch.ops.aten.view.default(convert_element_type_247, [512, 384]);  convert_element_type_247 = None
        permute_61 = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_26 = torch.ops.aten.addmm.default(convert_element_type_245, view_52, permute_61)
        view_53 = torch.ops.aten.view.default(addmm_26, [32, 4, 4, 1536])
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        mul_98 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.5)
        mul_99 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476);  convert_element_type_251 = None
        erf_13 = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_60 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_98, add_60);  mul_98 = add_60 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        view_54 = torch.ops.aten.view.default(convert_element_type_252, [512, 1536]);  convert_element_type_252 = None
        permute_62 = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        addmm_27 = torch.ops.aten.addmm.default(convert_element_type_253, view_54, permute_62)
        view_55 = torch.ops.aten.view.default(addmm_27, [32, 4, 4, 384])
        permute_63 = torch.ops.aten.permute.default(view_55, [0, 3, 1, 2]);  view_55 = None
        mul_101 = torch.ops.aten.mul.Tensor(primals_20, permute_63);  permute_63 = None
        inductor_lookup_seed_default_12 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_63 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        lt_12 = torch.ops.aten.lt.Scalar(inductor_random_default_63, 0.9235294117647059);  inductor_random_default_63 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(lt_12, torch.float32)
        div_14 = torch.ops.aten.div.Tensor(convert_element_type_258, 0.9235294117647059);  convert_element_type_258 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, div_14);  mul_101 = div_14 = None
        add_61 = torch.ops.aten.add.Tensor(mul_102, add_57);  mul_102 = add_57 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_260 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convert_element_type_261 = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        convolution_17 = torch.ops.aten.convolution.default(convert_element_type_261, convert_element_type_260, convert_element_type_259, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_64 = torch.ops.aten.permute.default(convolution_17, [0, 2, 3, 1]);  convolution_17 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(permute_64, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_262, [3], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_19 = torch.ops.aten.sub.Tensor(convert_element_type_262, getitem_35);  convert_element_type_262 = None
        mul_103 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_103, primals_147);  mul_103 = None
        add_63 = torch.ops.aten.add.Tensor(mul_104, primals_148);  mul_104 = None
        convert_element_type_263 = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None
        view_56 = torch.ops.aten.view.default(convert_element_type_265, [512, 384]);  convert_element_type_265 = None
        permute_65 = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_28 = torch.ops.aten.addmm.default(convert_element_type_263, view_56, permute_65)
        view_57 = torch.ops.aten.view.default(addmm_28, [32, 4, 4, 1536])
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_105 = torch.ops.aten.mul.Tensor(convert_element_type_269, 0.5)
        mul_106 = torch.ops.aten.mul.Tensor(convert_element_type_269, 0.7071067811865476);  convert_element_type_269 = None
        erf_14 = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_64 = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_105, add_64);  mul_105 = add_64 = None
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convert_element_type_272 = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        view_58 = torch.ops.aten.view.default(convert_element_type_270, [512, 1536]);  convert_element_type_270 = None
        permute_66 = torch.ops.aten.permute.default(convert_element_type_272, [1, 0]);  convert_element_type_272 = None
        addmm_29 = torch.ops.aten.addmm.default(convert_element_type_271, view_58, permute_66)
        view_59 = torch.ops.aten.view.default(addmm_29, [32, 4, 4, 384])
        permute_67 = torch.ops.aten.permute.default(view_59, [0, 3, 1, 2]);  view_59 = None
        mul_108 = torch.ops.aten.mul.Tensor(primals_21, permute_67);  permute_67 = None
        inductor_lookup_seed_default_13 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_62 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        lt_13 = torch.ops.aten.lt.Scalar(inductor_random_default_62, 0.9176470588235294);  inductor_random_default_62 = None
        convert_element_type_276 = torch.ops.prims.convert_element_type.default(lt_13, torch.float32)
        div_15 = torch.ops.aten.div.Tensor(convert_element_type_276, 0.9176470588235294);  convert_element_type_276 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, div_15);  mul_108 = div_15 = None
        add_65 = torch.ops.aten.add.Tensor(mul_109, add_61);  mul_109 = add_61 = None
        permute_69 = torch.ops.aten.permute.default(add_65, [0, 2, 3, 1])
        var_mean_18 = torch.ops.aten.var_mean.correction(permute_69, [3], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_20 = torch.ops.aten.sub.Tensor(permute_69, getitem_37);  permute_69 = None
        mul_110 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, primals_22);  mul_110 = None
        add_67 = torch.ops.aten.add.Tensor(mul_111, primals_23);  mul_111 = None
        permute_70 = torch.ops.aten.permute.default(add_67, [0, 3, 1, 2]);  add_67 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(permute_70, torch.bfloat16);  permute_70 = None
        convolution_18 = torch.ops.aten.convolution.default(convert_element_type_279, convert_element_type_278, convert_element_type_277, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convert_element_type_280 = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convolution_19 = torch.ops.aten.convolution.default(convolution_18, convert_element_type_281, convert_element_type_280, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_71 = torch.ops.aten.permute.default(convolution_19, [0, 2, 3, 1]);  convolution_19 = None
        convert_element_type_282 = torch.ops.prims.convert_element_type.default(permute_71, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_282, [3], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_21 = torch.ops.aten.sub.Tensor(convert_element_type_282, getitem_39);  convert_element_type_282 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, primals_157);  mul_112 = None
        add_69 = torch.ops.aten.add.Tensor(mul_113, primals_158);  mul_113 = None
        convert_element_type_283 = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        view_60 = torch.ops.aten.view.default(convert_element_type_285, [128, 768]);  convert_element_type_285 = None
        permute_72 = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        addmm_30 = torch.ops.aten.addmm.default(convert_element_type_283, view_60, permute_72)
        view_61 = torch.ops.aten.view.default(addmm_30, [32, 2, 2, 3072])
        convert_element_type_289 = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_114 = torch.ops.aten.mul.Tensor(convert_element_type_289, 0.5)
        mul_115 = torch.ops.aten.mul.Tensor(convert_element_type_289, 0.7071067811865476);  convert_element_type_289 = None
        erf_15 = torch.ops.aten.erf.default(mul_115);  mul_115 = None
        add_70 = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_114, add_70);  mul_114 = add_70 = None
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        view_62 = torch.ops.aten.view.default(convert_element_type_290, [128, 3072]);  convert_element_type_290 = None
        permute_73 = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        addmm_31 = torch.ops.aten.addmm.default(convert_element_type_291, view_62, permute_73)
        view_63 = torch.ops.aten.view.default(addmm_31, [32, 2, 2, 768])
        permute_74 = torch.ops.aten.permute.default(view_63, [0, 3, 1, 2]);  view_63 = None
        mul_117 = torch.ops.aten.mul.Tensor(primals_24, permute_74);  permute_74 = None
        inductor_lookup_seed_default_14 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_61 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        lt_14 = torch.ops.aten.lt.Scalar(inductor_random_default_61, 0.9117647058823529);  inductor_random_default_61 = None
        convert_element_type_296 = torch.ops.prims.convert_element_type.default(lt_14, torch.float32)
        div_16 = torch.ops.aten.div.Tensor(convert_element_type_296, 0.9117647058823529);  convert_element_type_296 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_117, div_16);  mul_117 = div_16 = None
        add_71 = torch.ops.aten.add.Tensor(mul_118, convolution_18);  mul_118 = None
        convert_element_type_297 = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16)
        convolution_20 = torch.ops.aten.convolution.default(convert_element_type_299, convert_element_type_298, convert_element_type_297, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_75 = torch.ops.aten.permute.default(convolution_20, [0, 2, 3, 1]);  convolution_20 = None
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(permute_75, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_300, [3], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_22 = torch.ops.aten.sub.Tensor(convert_element_type_300, getitem_41);  convert_element_type_300 = None
        mul_119 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_119, primals_165);  mul_119 = None
        add_73 = torch.ops.aten.add.Tensor(mul_120, primals_166);  mul_120 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        convert_element_type_302 = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convert_element_type_303 = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        view_64 = torch.ops.aten.view.default(convert_element_type_303, [128, 768]);  convert_element_type_303 = None
        permute_76 = torch.ops.aten.permute.default(convert_element_type_302, [1, 0]);  convert_element_type_302 = None
        addmm_32 = torch.ops.aten.addmm.default(convert_element_type_301, view_64, permute_76)
        view_65 = torch.ops.aten.view.default(addmm_32, [32, 2, 2, 3072])
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        mul_121 = torch.ops.aten.mul.Tensor(convert_element_type_307, 0.5)
        mul_122 = torch.ops.aten.mul.Tensor(convert_element_type_307, 0.7071067811865476);  convert_element_type_307 = None
        erf_16 = torch.ops.aten.erf.default(mul_122);  mul_122 = None
        add_74 = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_121, add_74);  mul_121 = add_74 = None
        convert_element_type_308 = torch.ops.prims.convert_element_type.default(mul_123, torch.bfloat16);  mul_123 = None
        convert_element_type_309 = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_310 = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        view_66 = torch.ops.aten.view.default(convert_element_type_308, [128, 3072]);  convert_element_type_308 = None
        permute_77 = torch.ops.aten.permute.default(convert_element_type_310, [1, 0]);  convert_element_type_310 = None
        addmm_33 = torch.ops.aten.addmm.default(convert_element_type_309, view_66, permute_77)
        view_67 = torch.ops.aten.view.default(addmm_33, [32, 2, 2, 768])
        permute_78 = torch.ops.aten.permute.default(view_67, [0, 3, 1, 2]);  view_67 = None
        mul_124 = torch.ops.aten.mul.Tensor(primals_25, permute_78);  permute_78 = None
        inductor_lookup_seed_default_15 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_60 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        lt_15 = torch.ops.aten.lt.Scalar(inductor_random_default_60, 0.9058823529411765);  inductor_random_default_60 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(lt_15, torch.float32)
        div_17 = torch.ops.aten.div.Tensor(convert_element_type_314, 0.9058823529411765);  convert_element_type_314 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, div_17);  mul_124 = div_17 = None
        add_75 = torch.ops.aten.add.Tensor(mul_125, add_71);  mul_125 = add_71 = None
        convert_element_type_315 = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_316 = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        convolution_21 = torch.ops.aten.convolution.default(convert_element_type_317, convert_element_type_316, convert_element_type_315, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_79 = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(permute_79, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_318, [3], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_23 = torch.ops.aten.sub.Tensor(convert_element_type_318, getitem_43);  convert_element_type_318 = None
        mul_126 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_126, primals_173);  mul_126 = None
        add_77 = torch.ops.aten.add.Tensor(mul_127, primals_174);  mul_127 = None
        convert_element_type_319 = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convert_element_type_321 = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_321, [128, 768]);  convert_element_type_321 = None
        permute_80 = torch.ops.aten.permute.default(convert_element_type_320, [1, 0]);  convert_element_type_320 = None
        addmm_34 = torch.ops.aten.addmm.default(convert_element_type_319, view_68, permute_80)
        view_69 = torch.ops.aten.view.default(addmm_34, [32, 2, 2, 3072])
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.5)
        mul_129 = torch.ops.aten.mul.Tensor(convert_element_type_325, 0.7071067811865476);  convert_element_type_325 = None
        erf_17 = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_78 = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_128, add_78);  mul_128 = add_78 = None
        convert_element_type_326 = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        convert_element_type_327 = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        view_70 = torch.ops.aten.view.default(convert_element_type_326, [128, 3072]);  convert_element_type_326 = None
        permute_81 = torch.ops.aten.permute.default(convert_element_type_328, [1, 0]);  convert_element_type_328 = None
        addmm_35 = torch.ops.aten.addmm.default(convert_element_type_327, view_70, permute_81)
        view_71 = torch.ops.aten.view.default(addmm_35, [32, 2, 2, 768])
        permute_82 = torch.ops.aten.permute.default(view_71, [0, 3, 1, 2]);  view_71 = None
        mul_131 = torch.ops.aten.mul.Tensor(primals_26, permute_82);  permute_82 = None
        inductor_lookup_seed_default_16 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_59 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        lt_16 = torch.ops.aten.lt.Scalar(inductor_random_default_59, 0.9);  inductor_random_default_59 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(lt_16, torch.float32)
        div_18 = torch.ops.aten.div.Tensor(convert_element_type_332, 0.9);  convert_element_type_332 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_131, div_18);  mul_131 = div_18 = None
        add_79 = torch.ops.aten.add.Tensor(mul_132, add_75);  mul_132 = add_75 = None
        convert_element_type_335 = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        convolution_22 = torch.ops.aten.convolution.default(convert_element_type_335, convert_element_type_1, convert_element_type, [4, 4], [0, 0], [1, 1], False, [0, 0], 1)
        permute_83 = torch.ops.aten.permute.default(convolution_22, [0, 2, 3, 1]);  convolution_22 = None
        convert_element_type_336 = torch.ops.prims.convert_element_type.default(permute_83, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_336, [3], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_24 = torch.ops.aten.sub.Tensor(convert_element_type_336, getitem_45);  convert_element_type_336 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, primals_1);  mul_133 = None
        add_81 = torch.ops.aten.add.Tensor(mul_134, primals_2);  mul_134 = None
        permute_84 = torch.ops.aten.permute.default(add_81, [0, 3, 1, 2]);  add_81 = None
        convert_element_type_339 = torch.ops.prims.convert_element_type.default(permute_84, torch.bfloat16)
        convolution_23 = torch.ops.aten.convolution.default(convert_element_type_339, convert_element_type_5, convert_element_type_4, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_85 = torch.ops.aten.permute.default(convolution_23, [0, 2, 3, 1]);  convolution_23 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(permute_85, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_340, [3], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_25 = torch.ops.aten.sub.Tensor(convert_element_type_340, getitem_47);  convert_element_type_340 = None
        mul_135 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_135, primals_31);  mul_135 = None
        add_83 = torch.ops.aten.add.Tensor(mul_136, primals_32);  mul_136 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_72 = torch.ops.aten.view.default(convert_element_type_343, [8192, 96]);  convert_element_type_343 = None
        addmm_36 = torch.ops.aten.addmm.default(convert_element_type_8, view_72, permute_3)
        view_73 = torch.ops.aten.view.default(addmm_36, [32, 16, 16, 384])
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        mul_137 = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.5)
        mul_138 = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.7071067811865476);  convert_element_type_347 = None
        erf_18 = torch.ops.aten.erf.default(mul_138);  mul_138 = None
        add_84 = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_137, add_84);  mul_137 = add_84 = None
        convert_element_type_348 = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None
        view_74 = torch.ops.aten.view.default(convert_element_type_348, [8192, 384]);  convert_element_type_348 = None
        addmm_37 = torch.ops.aten.addmm.default(convert_element_type_16, view_74, permute_4)
        view_75 = torch.ops.aten.view.default(addmm_37, [32, 16, 16, 96])
        permute_88 = torch.ops.aten.permute.default(view_75, [0, 3, 1, 2]);  view_75 = None
        mul_140 = torch.ops.aten.mul.Tensor(primals_3, permute_88);  permute_88 = None
        add_85 = torch.ops.aten.add.Tensor(mul_140, permute_84);  mul_140 = permute_84 = None
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        convolution_24 = torch.ops.aten.convolution.default(convert_element_type_356, convert_element_type_22, convert_element_type_21, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_89 = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(permute_89, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_357, [3], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_26 = torch.ops.aten.sub.Tensor(convert_element_type_357, getitem_49);  convert_element_type_357 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_141, primals_39);  mul_141 = None
        add_87 = torch.ops.aten.add.Tensor(mul_142, primals_40);  mul_142 = None
        convert_element_type_360 = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_76 = torch.ops.aten.view.default(convert_element_type_360, [8192, 96]);  convert_element_type_360 = None
        addmm_38 = torch.ops.aten.addmm.default(convert_element_type_25, view_76, permute_7)
        view_77 = torch.ops.aten.view.default(addmm_38, [32, 16, 16, 384])
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_364, 0.5)
        mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_364, 0.7071067811865476);  convert_element_type_364 = None
        erf_19 = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_88 = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_143, add_88);  mul_143 = add_88 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        view_78 = torch.ops.aten.view.default(convert_element_type_365, [8192, 384]);  convert_element_type_365 = None
        addmm_39 = torch.ops.aten.addmm.default(convert_element_type_33, view_78, permute_8)
        view_79 = torch.ops.aten.view.default(addmm_39, [32, 16, 16, 96])
        permute_92 = torch.ops.aten.permute.default(view_79, [0, 3, 1, 2]);  view_79 = None
        mul_146 = torch.ops.aten.mul.Tensor(primals_4, permute_92);  permute_92 = None
        inductor_lookup_seed_default_17 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_58 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        lt_17 = torch.ops.aten.lt.Scalar(inductor_random_default_58, 0.9941176470588236);  inductor_random_default_58 = None
        convert_element_type_371 = torch.ops.prims.convert_element_type.default(lt_17, torch.float32)
        div_19 = torch.ops.aten.div.Tensor(convert_element_type_371, 0.9941176470588236);  convert_element_type_371 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_146, div_19);  mul_146 = div_19 = None
        add_89 = torch.ops.aten.add.Tensor(mul_147, add_85);  mul_147 = add_85 = None
        convert_element_type_374 = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        convolution_25 = torch.ops.aten.convolution.default(convert_element_type_374, convert_element_type_40, convert_element_type_39, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_93 = torch.ops.aten.permute.default(convolution_25, [0, 2, 3, 1]);  convolution_25 = None
        convert_element_type_375 = torch.ops.prims.convert_element_type.default(permute_93, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_375, [3], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_27 = torch.ops.aten.sub.Tensor(convert_element_type_375, getitem_51);  convert_element_type_375 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_25);  sub_27 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, primals_47);  mul_148 = None
        add_91 = torch.ops.aten.add.Tensor(mul_149, primals_48);  mul_149 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        view_80 = torch.ops.aten.view.default(convert_element_type_378, [8192, 96]);  convert_element_type_378 = None
        addmm_40 = torch.ops.aten.addmm.default(convert_element_type_43, view_80, permute_11)
        view_81 = torch.ops.aten.view.default(addmm_40, [32, 16, 16, 384])
        convert_element_type_382 = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_382, 0.5)
        mul_151 = torch.ops.aten.mul.Tensor(convert_element_type_382, 0.7071067811865476);  convert_element_type_382 = None
        erf_20 = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_92 = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_150, add_92);  mul_150 = add_92 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        view_82 = torch.ops.aten.view.default(convert_element_type_383, [8192, 384]);  convert_element_type_383 = None
        addmm_41 = torch.ops.aten.addmm.default(convert_element_type_51, view_82, permute_12)
        view_83 = torch.ops.aten.view.default(addmm_41, [32, 16, 16, 96])
        permute_96 = torch.ops.aten.permute.default(view_83, [0, 3, 1, 2]);  view_83 = None
        mul_153 = torch.ops.aten.mul.Tensor(primals_5, permute_96);  permute_96 = None
        inductor_lookup_seed_default_18 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_57 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        lt_18 = torch.ops.aten.lt.Scalar(inductor_random_default_57, 0.9882352941176471);  inductor_random_default_57 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(lt_18, torch.float32)
        div_20 = torch.ops.aten.div.Tensor(convert_element_type_389, 0.9882352941176471);  convert_element_type_389 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, div_20);  mul_153 = div_20 = None
        add_93 = torch.ops.aten.add.Tensor(mul_154, add_89);  mul_154 = add_89 = None
        permute_98 = torch.ops.aten.permute.default(add_93, [0, 2, 3, 1])
        var_mean_26 = torch.ops.aten.var_mean.correction(permute_98, [3], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_28 = torch.ops.aten.sub.Tensor(permute_98, getitem_53);  permute_98 = None
        mul_155 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_26);  sub_28 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, primals_6);  mul_155 = None
        add_95 = torch.ops.aten.add.Tensor(mul_156, primals_7);  mul_156 = None
        permute_99 = torch.ops.aten.permute.default(add_95, [0, 3, 1, 2]);  add_95 = None
        convert_element_type_392 = torch.ops.prims.convert_element_type.default(permute_99, torch.bfloat16);  permute_99 = None
        convolution_26 = torch.ops.aten.convolution.default(convert_element_type_392, convert_element_type_58, convert_element_type_57, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_27 = torch.ops.aten.convolution.default(convolution_26, convert_element_type_61, convert_element_type_60, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_100 = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None
        convert_element_type_395 = torch.ops.prims.convert_element_type.default(permute_100, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_395, [3], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_29 = torch.ops.aten.sub.Tensor(convert_element_type_395, getitem_55);  convert_element_type_395 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_27);  sub_29 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, primals_57);  mul_157 = None
        add_97 = torch.ops.aten.add.Tensor(mul_158, primals_58);  mul_158 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None
        view_84 = torch.ops.aten.view.default(convert_element_type_398, [2048, 192]);  convert_element_type_398 = None
        addmm_42 = torch.ops.aten.addmm.default(convert_element_type_63, view_84, permute_18)
        view_85 = torch.ops.aten.view.default(addmm_42, [32, 8, 8, 768])
        convert_element_type_402 = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_402, 0.5)
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_402, 0.7071067811865476);  convert_element_type_402 = None
        erf_21 = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_98 = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_98);  mul_159 = add_98 = None
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        view_86 = torch.ops.aten.view.default(convert_element_type_403, [2048, 768]);  convert_element_type_403 = None
        addmm_43 = torch.ops.aten.addmm.default(convert_element_type_71, view_86, permute_19)
        view_87 = torch.ops.aten.view.default(addmm_43, [32, 8, 8, 192])
        permute_103 = torch.ops.aten.permute.default(view_87, [0, 3, 1, 2]);  view_87 = None
        mul_162 = torch.ops.aten.mul.Tensor(primals_8, permute_103);  permute_103 = None
        inductor_lookup_seed_default_19 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_56 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        lt_19 = torch.ops.aten.lt.Scalar(inductor_random_default_56, 0.9823529411764705);  inductor_random_default_56 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(lt_19, torch.float32)
        div_21 = torch.ops.aten.div.Tensor(convert_element_type_409, 0.9823529411764705);  convert_element_type_409 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, div_21);  mul_162 = div_21 = None
        add_99 = torch.ops.aten.add.Tensor(mul_163, convolution_26);  mul_163 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)
        convolution_28 = torch.ops.aten.convolution.default(convert_element_type_412, convert_element_type_78, convert_element_type_77, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_104 = torch.ops.aten.permute.default(convolution_28, [0, 2, 3, 1]);  convolution_28 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(permute_104, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_413, [3], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_100 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_30 = torch.ops.aten.sub.Tensor(convert_element_type_413, getitem_57);  convert_element_type_413 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, primals_65);  mul_164 = None
        add_101 = torch.ops.aten.add.Tensor(mul_165, primals_66);  mul_165 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        view_88 = torch.ops.aten.view.default(convert_element_type_416, [2048, 192]);  convert_element_type_416 = None
        addmm_44 = torch.ops.aten.addmm.default(convert_element_type_81, view_88, permute_22)
        view_89 = torch.ops.aten.view.default(addmm_44, [32, 8, 8, 768])
        convert_element_type_420 = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_420, 0.5)
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_420, 0.7071067811865476);  convert_element_type_420 = None
        erf_22 = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_102 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_166, add_102);  mul_166 = add_102 = None
        convert_element_type_421 = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        view_90 = torch.ops.aten.view.default(convert_element_type_421, [2048, 768]);  convert_element_type_421 = None
        addmm_45 = torch.ops.aten.addmm.default(convert_element_type_89, view_90, permute_23)
        view_91 = torch.ops.aten.view.default(addmm_45, [32, 8, 8, 192])
        permute_107 = torch.ops.aten.permute.default(view_91, [0, 3, 1, 2]);  view_91 = None
        mul_169 = torch.ops.aten.mul.Tensor(primals_9, permute_107);  permute_107 = None
        inductor_lookup_seed_default_20 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_55 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        lt_20 = torch.ops.aten.lt.Scalar(inductor_random_default_55, 0.9764705882352941);  inductor_random_default_55 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(lt_20, torch.float32)
        div_22 = torch.ops.aten.div.Tensor(convert_element_type_427, 0.9764705882352941);  convert_element_type_427 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, div_22);  mul_169 = div_22 = None
        add_103 = torch.ops.aten.add.Tensor(mul_170, add_99);  mul_170 = add_99 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)
        convolution_29 = torch.ops.aten.convolution.default(convert_element_type_430, convert_element_type_96, convert_element_type_95, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_108 = torch.ops.aten.permute.default(convolution_29, [0, 2, 3, 1]);  convolution_29 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(permute_108, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_431, [3], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_104 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_31 = torch.ops.aten.sub.Tensor(convert_element_type_431, getitem_59);  convert_element_type_431 = None
        mul_171 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_29);  sub_31 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, primals_73);  mul_171 = None
        add_105 = torch.ops.aten.add.Tensor(mul_172, primals_74);  mul_172 = None
        convert_element_type_434 = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_434, [2048, 192]);  convert_element_type_434 = None
        addmm_46 = torch.ops.aten.addmm.default(convert_element_type_99, view_92, permute_26)
        view_93 = torch.ops.aten.view.default(addmm_46, [32, 8, 8, 768])
        convert_element_type_438 = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_173 = torch.ops.aten.mul.Tensor(convert_element_type_438, 0.5)
        mul_174 = torch.ops.aten.mul.Tensor(convert_element_type_438, 0.7071067811865476);  convert_element_type_438 = None
        erf_23 = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_106 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_175 = torch.ops.aten.mul.Tensor(mul_173, add_106);  mul_173 = add_106 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        view_94 = torch.ops.aten.view.default(convert_element_type_439, [2048, 768]);  convert_element_type_439 = None
        addmm_47 = torch.ops.aten.addmm.default(convert_element_type_107, view_94, permute_27)
        view_95 = torch.ops.aten.view.default(addmm_47, [32, 8, 8, 192])
        permute_111 = torch.ops.aten.permute.default(view_95, [0, 3, 1, 2]);  view_95 = None
        mul_176 = torch.ops.aten.mul.Tensor(primals_10, permute_111);  permute_111 = None
        inductor_lookup_seed_default_21 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_54 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        lt_21 = torch.ops.aten.lt.Scalar(inductor_random_default_54, 0.9705882352941176);  inductor_random_default_54 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(lt_21, torch.float32)
        div_23 = torch.ops.aten.div.Tensor(convert_element_type_445, 0.9705882352941176);  convert_element_type_445 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, div_23);  mul_176 = div_23 = None
        add_107 = torch.ops.aten.add.Tensor(mul_177, add_103);  mul_177 = add_103 = None
        permute_113 = torch.ops.aten.permute.default(add_107, [0, 2, 3, 1])
        var_mean_30 = torch.ops.aten.var_mean.correction(permute_113, [3], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_108 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_32 = torch.ops.aten.sub.Tensor(permute_113, getitem_61);  permute_113 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_30);  sub_32 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, primals_11);  mul_178 = None
        add_109 = torch.ops.aten.add.Tensor(mul_179, primals_12);  mul_179 = None
        permute_114 = torch.ops.aten.permute.default(add_109, [0, 3, 1, 2]);  add_109 = None
        convert_element_type_448 = torch.ops.prims.convert_element_type.default(permute_114, torch.bfloat16);  permute_114 = None
        convolution_30 = torch.ops.aten.convolution.default(convert_element_type_448, convert_element_type_114, convert_element_type_113, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_31 = torch.ops.aten.convolution.default(convolution_30, convert_element_type_117, convert_element_type_116, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_115 = torch.ops.aten.permute.default(convolution_31, [0, 2, 3, 1]);  convolution_31 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(permute_115, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_451, [3], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_33 = torch.ops.aten.sub.Tensor(convert_element_type_451, getitem_63);  convert_element_type_451 = None
        mul_180 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_31);  sub_33 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, primals_83);  mul_180 = None
        add_111 = torch.ops.aten.add.Tensor(mul_181, primals_84);  mul_181 = None
        convert_element_type_454 = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        view_96 = torch.ops.aten.view.default(convert_element_type_454, [512, 384]);  convert_element_type_454 = None
        addmm_48 = torch.ops.aten.addmm.default(convert_element_type_119, view_96, permute_33)
        view_97 = torch.ops.aten.view.default(addmm_48, [32, 4, 4, 1536])
        convert_element_type_458 = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_458, 0.5)
        mul_183 = torch.ops.aten.mul.Tensor(convert_element_type_458, 0.7071067811865476);  convert_element_type_458 = None
        erf_24 = torch.ops.aten.erf.default(mul_183);  mul_183 = None
        add_112 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_182, add_112);  mul_182 = add_112 = None
        convert_element_type_459 = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        view_98 = torch.ops.aten.view.default(convert_element_type_459, [512, 1536]);  convert_element_type_459 = None
        addmm_49 = torch.ops.aten.addmm.default(convert_element_type_127, view_98, permute_34)
        view_99 = torch.ops.aten.view.default(addmm_49, [32, 4, 4, 384])
        permute_118 = torch.ops.aten.permute.default(view_99, [0, 3, 1, 2]);  view_99 = None
        mul_185 = torch.ops.aten.mul.Tensor(primals_13, permute_118);  permute_118 = None
        inductor_lookup_seed_default_22 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_53 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        lt_22 = torch.ops.aten.lt.Scalar(inductor_random_default_53, 0.9647058823529412);  inductor_random_default_53 = None
        convert_element_type_465 = torch.ops.prims.convert_element_type.default(lt_22, torch.float32)
        div_24 = torch.ops.aten.div.Tensor(convert_element_type_465, 0.9647058823529412);  convert_element_type_465 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, div_24);  mul_185 = div_24 = None
        add_113 = torch.ops.aten.add.Tensor(mul_186, convolution_30);  mul_186 = None
        convert_element_type_468 = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16)
        convolution_32 = torch.ops.aten.convolution.default(convert_element_type_468, convert_element_type_134, convert_element_type_133, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_119 = torch.ops.aten.permute.default(convolution_32, [0, 2, 3, 1]);  convolution_32 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(permute_119, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_469, [3], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_114 = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_34 = torch.ops.aten.sub.Tensor(convert_element_type_469, getitem_65);  convert_element_type_469 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_32);  sub_34 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, primals_91);  mul_187 = None
        add_115 = torch.ops.aten.add.Tensor(mul_188, primals_92);  mul_188 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None
        view_100 = torch.ops.aten.view.default(convert_element_type_472, [512, 384]);  convert_element_type_472 = None
        addmm_50 = torch.ops.aten.addmm.default(convert_element_type_137, view_100, permute_37)
        view_101 = torch.ops.aten.view.default(addmm_50, [32, 4, 4, 1536])
        convert_element_type_476 = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        mul_189 = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.5)
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_476, 0.7071067811865476);  convert_element_type_476 = None
        erf_25 = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_116 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_189, add_116);  mul_189 = add_116 = None
        convert_element_type_477 = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None
        view_102 = torch.ops.aten.view.default(convert_element_type_477, [512, 1536]);  convert_element_type_477 = None
        addmm_51 = torch.ops.aten.addmm.default(convert_element_type_145, view_102, permute_38)
        view_103 = torch.ops.aten.view.default(addmm_51, [32, 4, 4, 384])
        permute_122 = torch.ops.aten.permute.default(view_103, [0, 3, 1, 2]);  view_103 = None
        mul_192 = torch.ops.aten.mul.Tensor(primals_14, permute_122);  permute_122 = None
        inductor_lookup_seed_default_23 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_52 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        lt_23 = torch.ops.aten.lt.Scalar(inductor_random_default_52, 0.9588235294117647);  inductor_random_default_52 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(lt_23, torch.float32)
        div_25 = torch.ops.aten.div.Tensor(convert_element_type_483, 0.9588235294117647);  convert_element_type_483 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, div_25);  mul_192 = div_25 = None
        add_117 = torch.ops.aten.add.Tensor(mul_193, add_113);  mul_193 = add_113 = None
        convert_element_type_486 = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16)
        convolution_33 = torch.ops.aten.convolution.default(convert_element_type_486, convert_element_type_152, convert_element_type_151, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_123 = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None
        convert_element_type_487 = torch.ops.prims.convert_element_type.default(permute_123, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_487, [3], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_35 = torch.ops.aten.sub.Tensor(convert_element_type_487, getitem_67);  convert_element_type_487 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_33);  sub_35 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, primals_99);  mul_194 = None
        add_119 = torch.ops.aten.add.Tensor(mul_195, primals_100);  mul_195 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_490, [512, 384]);  convert_element_type_490 = None
        addmm_52 = torch.ops.aten.addmm.default(convert_element_type_155, view_104, permute_41)
        view_105 = torch.ops.aten.view.default(addmm_52, [32, 4, 4, 1536])
        convert_element_type_494 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_196 = torch.ops.aten.mul.Tensor(convert_element_type_494, 0.5)
        mul_197 = torch.ops.aten.mul.Tensor(convert_element_type_494, 0.7071067811865476);  convert_element_type_494 = None
        erf_26 = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_120 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, add_120);  mul_196 = add_120 = None
        convert_element_type_495 = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_495, [512, 1536]);  convert_element_type_495 = None
        addmm_53 = torch.ops.aten.addmm.default(convert_element_type_163, view_106, permute_42)
        view_107 = torch.ops.aten.view.default(addmm_53, [32, 4, 4, 384])
        permute_126 = torch.ops.aten.permute.default(view_107, [0, 3, 1, 2]);  view_107 = None
        mul_199 = torch.ops.aten.mul.Tensor(primals_15, permute_126);  permute_126 = None
        inductor_lookup_seed_default_24 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_51 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        lt_24 = torch.ops.aten.lt.Scalar(inductor_random_default_51, 0.9529411764705882);  inductor_random_default_51 = None
        convert_element_type_501 = torch.ops.prims.convert_element_type.default(lt_24, torch.float32)
        div_26 = torch.ops.aten.div.Tensor(convert_element_type_501, 0.9529411764705882);  convert_element_type_501 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, div_26);  mul_199 = div_26 = None
        add_121 = torch.ops.aten.add.Tensor(mul_200, add_117);  mul_200 = add_117 = None
        convert_element_type_504 = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16)
        convolution_34 = torch.ops.aten.convolution.default(convert_element_type_504, convert_element_type_170, convert_element_type_169, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_127 = torch.ops.aten.permute.default(convolution_34, [0, 2, 3, 1]);  convolution_34 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(permute_127, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_505, [3], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_36 = torch.ops.aten.sub.Tensor(convert_element_type_505, getitem_69);  convert_element_type_505 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_34);  sub_36 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_201, primals_107);  mul_201 = None
        add_123 = torch.ops.aten.add.Tensor(mul_202, primals_108);  mul_202 = None
        convert_element_type_508 = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_508, [512, 384]);  convert_element_type_508 = None
        addmm_54 = torch.ops.aten.addmm.default(convert_element_type_173, view_108, permute_45)
        view_109 = torch.ops.aten.view.default(addmm_54, [32, 4, 4, 1536])
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_203 = torch.ops.aten.mul.Tensor(convert_element_type_512, 0.5)
        mul_204 = torch.ops.aten.mul.Tensor(convert_element_type_512, 0.7071067811865476);  convert_element_type_512 = None
        erf_27 = torch.ops.aten.erf.default(mul_204);  mul_204 = None
        add_124 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_203, add_124);  mul_203 = add_124 = None
        convert_element_type_513 = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16);  mul_205 = None
        view_110 = torch.ops.aten.view.default(convert_element_type_513, [512, 1536]);  convert_element_type_513 = None
        addmm_55 = torch.ops.aten.addmm.default(convert_element_type_181, view_110, permute_46)
        view_111 = torch.ops.aten.view.default(addmm_55, [32, 4, 4, 384])
        permute_130 = torch.ops.aten.permute.default(view_111, [0, 3, 1, 2]);  view_111 = None
        mul_206 = torch.ops.aten.mul.Tensor(primals_16, permute_130);  permute_130 = None
        inductor_lookup_seed_default_25 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_50 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        lt_25 = torch.ops.aten.lt.Scalar(inductor_random_default_50, 0.9470588235294117);  inductor_random_default_50 = None
        convert_element_type_519 = torch.ops.prims.convert_element_type.default(lt_25, torch.float32)
        div_27 = torch.ops.aten.div.Tensor(convert_element_type_519, 0.9470588235294117);  convert_element_type_519 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, div_27);  mul_206 = div_27 = None
        add_125 = torch.ops.aten.add.Tensor(mul_207, add_121);  mul_207 = add_121 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16)
        convolution_35 = torch.ops.aten.convolution.default(convert_element_type_522, convert_element_type_188, convert_element_type_187, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_131 = torch.ops.aten.permute.default(convolution_35, [0, 2, 3, 1]);  convolution_35 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(permute_131, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_523, [3], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_126 = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_37 = torch.ops.aten.sub.Tensor(convert_element_type_523, getitem_71);  convert_element_type_523 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_35);  sub_37 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, primals_115);  mul_208 = None
        add_127 = torch.ops.aten.add.Tensor(mul_209, primals_116);  mul_209 = None
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None
        view_112 = torch.ops.aten.view.default(convert_element_type_526, [512, 384]);  convert_element_type_526 = None
        addmm_56 = torch.ops.aten.addmm.default(convert_element_type_191, view_112, permute_49)
        view_113 = torch.ops.aten.view.default(addmm_56, [32, 4, 4, 1536])
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        mul_210 = torch.ops.aten.mul.Tensor(convert_element_type_530, 0.5)
        mul_211 = torch.ops.aten.mul.Tensor(convert_element_type_530, 0.7071067811865476);  convert_element_type_530 = None
        erf_28 = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_128 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_210, add_128);  mul_210 = add_128 = None
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        view_114 = torch.ops.aten.view.default(convert_element_type_531, [512, 1536]);  convert_element_type_531 = None
        addmm_57 = torch.ops.aten.addmm.default(convert_element_type_199, view_114, permute_50)
        view_115 = torch.ops.aten.view.default(addmm_57, [32, 4, 4, 384])
        permute_134 = torch.ops.aten.permute.default(view_115, [0, 3, 1, 2]);  view_115 = None
        mul_213 = torch.ops.aten.mul.Tensor(primals_17, permute_134);  permute_134 = None
        inductor_lookup_seed_default_26 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_49 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        lt_26 = torch.ops.aten.lt.Scalar(inductor_random_default_49, 0.9411764705882353);  inductor_random_default_49 = None
        convert_element_type_537 = torch.ops.prims.convert_element_type.default(lt_26, torch.float32)
        div_28 = torch.ops.aten.div.Tensor(convert_element_type_537, 0.9411764705882353);  convert_element_type_537 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, div_28);  mul_213 = div_28 = None
        add_129 = torch.ops.aten.add.Tensor(mul_214, add_125);  mul_214 = add_125 = None
        convert_element_type_540 = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16)
        convolution_36 = torch.ops.aten.convolution.default(convert_element_type_540, convert_element_type_206, convert_element_type_205, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_135 = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None
        convert_element_type_541 = torch.ops.prims.convert_element_type.default(permute_135, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_541, [3], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_38 = torch.ops.aten.sub.Tensor(convert_element_type_541, getitem_73);  convert_element_type_541 = None
        mul_215 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_36);  sub_38 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, primals_123);  mul_215 = None
        add_131 = torch.ops.aten.add.Tensor(mul_216, primals_124);  mul_216 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        view_116 = torch.ops.aten.view.default(convert_element_type_544, [512, 384]);  convert_element_type_544 = None
        addmm_58 = torch.ops.aten.addmm.default(convert_element_type_209, view_116, permute_53)
        view_117 = torch.ops.aten.view.default(addmm_58, [32, 4, 4, 1536])
        convert_element_type_548 = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_217 = torch.ops.aten.mul.Tensor(convert_element_type_548, 0.5)
        mul_218 = torch.ops.aten.mul.Tensor(convert_element_type_548, 0.7071067811865476);  convert_element_type_548 = None
        erf_29 = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_132 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_217, add_132);  mul_217 = add_132 = None
        convert_element_type_549 = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        view_118 = torch.ops.aten.view.default(convert_element_type_549, [512, 1536]);  convert_element_type_549 = None
        addmm_59 = torch.ops.aten.addmm.default(convert_element_type_217, view_118, permute_54)
        view_119 = torch.ops.aten.view.default(addmm_59, [32, 4, 4, 384])
        permute_138 = torch.ops.aten.permute.default(view_119, [0, 3, 1, 2]);  view_119 = None
        mul_220 = torch.ops.aten.mul.Tensor(primals_18, permute_138);  permute_138 = None
        inductor_lookup_seed_default_27 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_48 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        lt_27 = torch.ops.aten.lt.Scalar(inductor_random_default_48, 0.9352941176470588);  inductor_random_default_48 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(lt_27, torch.float32)
        div_29 = torch.ops.aten.div.Tensor(convert_element_type_555, 0.9352941176470588);  convert_element_type_555 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, div_29);  mul_220 = div_29 = None
        add_133 = torch.ops.aten.add.Tensor(mul_221, add_129);  mul_221 = add_129 = None
        convert_element_type_558 = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16)
        convolution_37 = torch.ops.aten.convolution.default(convert_element_type_558, convert_element_type_224, convert_element_type_223, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_139 = torch.ops.aten.permute.default(convolution_37, [0, 2, 3, 1]);  convolution_37 = None
        convert_element_type_559 = torch.ops.prims.convert_element_type.default(permute_139, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_559, [3], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_39 = torch.ops.aten.sub.Tensor(convert_element_type_559, getitem_75);  convert_element_type_559 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_37);  sub_39 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, primals_131);  mul_222 = None
        add_135 = torch.ops.aten.add.Tensor(mul_223, primals_132);  mul_223 = None
        convert_element_type_562 = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None
        view_120 = torch.ops.aten.view.default(convert_element_type_562, [512, 384]);  convert_element_type_562 = None
        addmm_60 = torch.ops.aten.addmm.default(convert_element_type_227, view_120, permute_57)
        view_121 = torch.ops.aten.view.default(addmm_60, [32, 4, 4, 1536])
        convert_element_type_566 = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        mul_224 = torch.ops.aten.mul.Tensor(convert_element_type_566, 0.5)
        mul_225 = torch.ops.aten.mul.Tensor(convert_element_type_566, 0.7071067811865476);  convert_element_type_566 = None
        erf_30 = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_136 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_224, add_136);  mul_224 = add_136 = None
        convert_element_type_567 = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        view_122 = torch.ops.aten.view.default(convert_element_type_567, [512, 1536]);  convert_element_type_567 = None
        addmm_61 = torch.ops.aten.addmm.default(convert_element_type_235, view_122, permute_58)
        view_123 = torch.ops.aten.view.default(addmm_61, [32, 4, 4, 384])
        permute_142 = torch.ops.aten.permute.default(view_123, [0, 3, 1, 2]);  view_123 = None
        mul_227 = torch.ops.aten.mul.Tensor(primals_19, permute_142);  permute_142 = None
        inductor_lookup_seed_default_28 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_47 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        lt_28 = torch.ops.aten.lt.Scalar(inductor_random_default_47, 0.9294117647058824);  inductor_random_default_47 = None
        convert_element_type_573 = torch.ops.prims.convert_element_type.default(lt_28, torch.float32)
        div_30 = torch.ops.aten.div.Tensor(convert_element_type_573, 0.9294117647058824);  convert_element_type_573 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, div_30);  mul_227 = div_30 = None
        add_137 = torch.ops.aten.add.Tensor(mul_228, add_133);  mul_228 = add_133 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16)
        convolution_38 = torch.ops.aten.convolution.default(convert_element_type_576, convert_element_type_242, convert_element_type_241, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_143 = torch.ops.aten.permute.default(convolution_38, [0, 2, 3, 1]);  convolution_38 = None
        convert_element_type_577 = torch.ops.prims.convert_element_type.default(permute_143, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_577, [3], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_40 = torch.ops.aten.sub.Tensor(convert_element_type_577, getitem_77);  convert_element_type_577 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_38);  sub_40 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, primals_139);  mul_229 = None
        add_139 = torch.ops.aten.add.Tensor(mul_230, primals_140);  mul_230 = None
        convert_element_type_580 = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        view_124 = torch.ops.aten.view.default(convert_element_type_580, [512, 384]);  convert_element_type_580 = None
        addmm_62 = torch.ops.aten.addmm.default(convert_element_type_245, view_124, permute_61)
        view_125 = torch.ops.aten.view.default(addmm_62, [32, 4, 4, 1536])
        convert_element_type_584 = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_584, 0.5)
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_584, 0.7071067811865476);  convert_element_type_584 = None
        erf_31 = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_140 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_231, add_140);  mul_231 = add_140 = None
        convert_element_type_585 = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None
        view_126 = torch.ops.aten.view.default(convert_element_type_585, [512, 1536]);  convert_element_type_585 = None
        addmm_63 = torch.ops.aten.addmm.default(convert_element_type_253, view_126, permute_62)
        view_127 = torch.ops.aten.view.default(addmm_63, [32, 4, 4, 384])
        permute_146 = torch.ops.aten.permute.default(view_127, [0, 3, 1, 2]);  view_127 = None
        mul_234 = torch.ops.aten.mul.Tensor(primals_20, permute_146);  permute_146 = None
        inductor_lookup_seed_default_29 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_46 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        lt_29 = torch.ops.aten.lt.Scalar(inductor_random_default_46, 0.9235294117647059);  inductor_random_default_46 = None
        convert_element_type_591 = torch.ops.prims.convert_element_type.default(lt_29, torch.float32)
        div_31 = torch.ops.aten.div.Tensor(convert_element_type_591, 0.9235294117647059);  convert_element_type_591 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, div_31);  mul_234 = div_31 = None
        add_141 = torch.ops.aten.add.Tensor(mul_235, add_137);  mul_235 = add_137 = None
        convert_element_type_594 = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)
        convolution_39 = torch.ops.aten.convolution.default(convert_element_type_594, convert_element_type_260, convert_element_type_259, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_147 = torch.ops.aten.permute.default(convolution_39, [0, 2, 3, 1]);  convolution_39 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(permute_147, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_595, [3], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_41 = torch.ops.aten.sub.Tensor(convert_element_type_595, getitem_79);  convert_element_type_595 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_39);  sub_41 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, primals_147);  mul_236 = None
        add_143 = torch.ops.aten.add.Tensor(mul_237, primals_148);  mul_237 = None
        convert_element_type_598 = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        view_128 = torch.ops.aten.view.default(convert_element_type_598, [512, 384]);  convert_element_type_598 = None
        addmm_64 = torch.ops.aten.addmm.default(convert_element_type_263, view_128, permute_65)
        view_129 = torch.ops.aten.view.default(addmm_64, [32, 4, 4, 1536])
        convert_element_type_602 = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_238 = torch.ops.aten.mul.Tensor(convert_element_type_602, 0.5)
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_602, 0.7071067811865476);  convert_element_type_602 = None
        erf_32 = torch.ops.aten.erf.default(mul_239);  mul_239 = None
        add_144 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_238, add_144);  mul_238 = add_144 = None
        convert_element_type_603 = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        view_130 = torch.ops.aten.view.default(convert_element_type_603, [512, 1536]);  convert_element_type_603 = None
        addmm_65 = torch.ops.aten.addmm.default(convert_element_type_271, view_130, permute_66)
        view_131 = torch.ops.aten.view.default(addmm_65, [32, 4, 4, 384])
        permute_150 = torch.ops.aten.permute.default(view_131, [0, 3, 1, 2]);  view_131 = None
        mul_241 = torch.ops.aten.mul.Tensor(primals_21, permute_150);  permute_150 = None
        inductor_lookup_seed_default_30 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_45 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        lt_30 = torch.ops.aten.lt.Scalar(inductor_random_default_45, 0.9176470588235294);  inductor_random_default_45 = None
        convert_element_type_609 = torch.ops.prims.convert_element_type.default(lt_30, torch.float32)
        div_32 = torch.ops.aten.div.Tensor(convert_element_type_609, 0.9176470588235294);  convert_element_type_609 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, div_32);  mul_241 = div_32 = None
        add_145 = torch.ops.aten.add.Tensor(mul_242, add_141);  mul_242 = add_141 = None
        permute_152 = torch.ops.aten.permute.default(add_145, [0, 2, 3, 1])
        var_mean_40 = torch.ops.aten.var_mean.correction(permute_152, [3], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_42 = torch.ops.aten.sub.Tensor(permute_152, getitem_81);  permute_152 = None
        mul_243 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_40);  sub_42 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_243, primals_22);  mul_243 = None
        add_147 = torch.ops.aten.add.Tensor(mul_244, primals_23);  mul_244 = None
        permute_153 = torch.ops.aten.permute.default(add_147, [0, 3, 1, 2]);  add_147 = None
        convert_element_type_612 = torch.ops.prims.convert_element_type.default(permute_153, torch.bfloat16);  permute_153 = None
        convolution_40 = torch.ops.aten.convolution.default(convert_element_type_612, convert_element_type_278, convert_element_type_277, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_41 = torch.ops.aten.convolution.default(convolution_40, convert_element_type_281, convert_element_type_280, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_154 = torch.ops.aten.permute.default(convolution_41, [0, 2, 3, 1]);  convolution_41 = None
        convert_element_type_615 = torch.ops.prims.convert_element_type.default(permute_154, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_615, [3], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_148 = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_43 = torch.ops.aten.sub.Tensor(convert_element_type_615, getitem_83);  convert_element_type_615 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_41);  sub_43 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, primals_157);  mul_245 = None
        add_149 = torch.ops.aten.add.Tensor(mul_246, primals_158);  mul_246 = None
        convert_element_type_618 = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        view_132 = torch.ops.aten.view.default(convert_element_type_618, [128, 768]);  convert_element_type_618 = None
        addmm_66 = torch.ops.aten.addmm.default(convert_element_type_283, view_132, permute_72)
        view_133 = torch.ops.aten.view.default(addmm_66, [32, 2, 2, 3072])
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_622, 0.5)
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_622, 0.7071067811865476);  convert_element_type_622 = None
        erf_33 = torch.ops.aten.erf.default(mul_248);  mul_248 = None
        add_150 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_247, add_150);  mul_247 = add_150 = None
        convert_element_type_623 = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        view_134 = torch.ops.aten.view.default(convert_element_type_623, [128, 3072]);  convert_element_type_623 = None
        addmm_67 = torch.ops.aten.addmm.default(convert_element_type_291, view_134, permute_73)
        view_135 = torch.ops.aten.view.default(addmm_67, [32, 2, 2, 768])
        permute_157 = torch.ops.aten.permute.default(view_135, [0, 3, 1, 2]);  view_135 = None
        mul_250 = torch.ops.aten.mul.Tensor(primals_24, permute_157);  permute_157 = None
        inductor_lookup_seed_default_31 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_44 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        lt_31 = torch.ops.aten.lt.Scalar(inductor_random_default_44, 0.9117647058823529);  inductor_random_default_44 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(lt_31, torch.float32)
        div_33 = torch.ops.aten.div.Tensor(convert_element_type_629, 0.9117647058823529);  convert_element_type_629 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, div_33);  mul_250 = div_33 = None
        add_151 = torch.ops.aten.add.Tensor(mul_251, convolution_40);  mul_251 = None
        convert_element_type_632 = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16)
        convolution_42 = torch.ops.aten.convolution.default(convert_element_type_632, convert_element_type_298, convert_element_type_297, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_158 = torch.ops.aten.permute.default(convolution_42, [0, 2, 3, 1]);  convolution_42 = None
        convert_element_type_633 = torch.ops.prims.convert_element_type.default(permute_158, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_633, [3], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_152 = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_44 = torch.ops.aten.sub.Tensor(convert_element_type_633, getitem_85);  convert_element_type_633 = None
        mul_252 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_42);  sub_44 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, primals_165);  mul_252 = None
        add_153 = torch.ops.aten.add.Tensor(mul_253, primals_166);  mul_253 = None
        convert_element_type_636 = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None
        view_136 = torch.ops.aten.view.default(convert_element_type_636, [128, 768]);  convert_element_type_636 = None
        addmm_68 = torch.ops.aten.addmm.default(convert_element_type_301, view_136, permute_76)
        view_137 = torch.ops.aten.view.default(addmm_68, [32, 2, 2, 3072])
        convert_element_type_640 = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_640, 0.5)
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_640, 0.7071067811865476);  convert_element_type_640 = None
        erf_34 = torch.ops.aten.erf.default(mul_255);  mul_255 = None
        add_154 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_254, add_154);  mul_254 = add_154 = None
        convert_element_type_641 = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16);  mul_256 = None
        view_138 = torch.ops.aten.view.default(convert_element_type_641, [128, 3072]);  convert_element_type_641 = None
        addmm_69 = torch.ops.aten.addmm.default(convert_element_type_309, view_138, permute_77)
        view_139 = torch.ops.aten.view.default(addmm_69, [32, 2, 2, 768])
        permute_161 = torch.ops.aten.permute.default(view_139, [0, 3, 1, 2]);  view_139 = None
        mul_257 = torch.ops.aten.mul.Tensor(primals_25, permute_161);  permute_161 = None
        inductor_lookup_seed_default_32 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_43 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        lt_32 = torch.ops.aten.lt.Scalar(inductor_random_default_43, 0.9058823529411765);  inductor_random_default_43 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(lt_32, torch.float32)
        div_34 = torch.ops.aten.div.Tensor(convert_element_type_647, 0.9058823529411765);  convert_element_type_647 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_257, div_34);  mul_257 = div_34 = None
        add_155 = torch.ops.aten.add.Tensor(mul_258, add_151);  mul_258 = add_151 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16)
        convolution_43 = torch.ops.aten.convolution.default(convert_element_type_650, convert_element_type_316, convert_element_type_315, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_162 = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None
        convert_element_type_651 = torch.ops.prims.convert_element_type.default(permute_162, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_651, [3], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_156 = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_45 = torch.ops.aten.sub.Tensor(convert_element_type_651, getitem_87);  convert_element_type_651 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_43);  sub_45 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, primals_173);  mul_259 = None
        add_157 = torch.ops.aten.add.Tensor(mul_260, primals_174);  mul_260 = None
        convert_element_type_654 = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        view_140 = torch.ops.aten.view.default(convert_element_type_654, [128, 768]);  convert_element_type_654 = None
        addmm_70 = torch.ops.aten.addmm.default(convert_element_type_319, view_140, permute_80)
        view_141 = torch.ops.aten.view.default(addmm_70, [32, 2, 2, 3072])
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_261 = torch.ops.aten.mul.Tensor(convert_element_type_658, 0.5)
        mul_262 = torch.ops.aten.mul.Tensor(convert_element_type_658, 0.7071067811865476);  convert_element_type_658 = None
        erf_35 = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_158 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_261, add_158);  mul_261 = add_158 = None
        convert_element_type_659 = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        view_142 = torch.ops.aten.view.default(convert_element_type_659, [128, 3072]);  convert_element_type_659 = None
        addmm_71 = torch.ops.aten.addmm.default(convert_element_type_327, view_142, permute_81)
        view_143 = torch.ops.aten.view.default(addmm_71, [32, 2, 2, 768])
        permute_165 = torch.ops.aten.permute.default(view_143, [0, 3, 1, 2]);  view_143 = None
        mul_264 = torch.ops.aten.mul.Tensor(primals_26, permute_165);  permute_165 = None
        inductor_lookup_seed_default_33 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_42 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        lt_33 = torch.ops.aten.lt.Scalar(inductor_random_default_42, 0.9);  inductor_random_default_42 = None
        convert_element_type_665 = torch.ops.prims.convert_element_type.default(lt_33, torch.float32)
        div_35 = torch.ops.aten.div.Tensor(convert_element_type_665, 0.9);  convert_element_type_665 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, div_35);  mul_264 = div_35 = None
        add_159 = torch.ops.aten.add.Tensor(mul_265, add_155);  mul_265 = add_155 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True, dtype = torch.float32);  pow_1 = None
        sqrt = torch.ops.aten.sqrt.default(sum_1);  sum_1 = None
        add_160 = torch.ops.aten.add.Tensor(sqrt, 1e-10)
        div_36 = torch.ops.aten.div.Tensor(add_13, add_160);  add_160 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_93, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_2, [1], True, dtype = torch.float32);  pow_2 = None
        sqrt_1 = torch.ops.aten.sqrt.default(sum_2);  sum_2 = None
        add_161 = torch.ops.aten.add.Tensor(sqrt_1, 1e-10)
        div_37 = torch.ops.aten.div.Tensor(add_93, add_161);  add_161 = None
        sub_46 = torch.ops.aten.sub.Tensor(div_36, div_37);  div_36 = div_37 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sub_46, 2);  sub_46 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        sum_3 = torch.ops.aten.sum.dim_IntList(pow_4, [1], True, dtype = torch.float32);  pow_4 = None
        sqrt_2 = torch.ops.aten.sqrt.default(sum_3);  sum_3 = None
        add_162 = torch.ops.aten.add.Tensor(sqrt_2, 1e-10)
        div_38 = torch.ops.aten.div.Tensor(add_27, add_162);  add_162 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_107, 2)
        sum_4 = torch.ops.aten.sum.dim_IntList(pow_5, [1], True, dtype = torch.float32);  pow_5 = None
        sqrt_3 = torch.ops.aten.sqrt.default(sum_4);  sum_4 = None
        add_163 = torch.ops.aten.add.Tensor(sqrt_3, 1e-10)
        div_39 = torch.ops.aten.div.Tensor(add_107, add_163);  add_163 = None
        sub_47 = torch.ops.aten.sub.Tensor(div_38, div_39);  div_38 = div_39 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sub_47, 2);  sub_47 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
        sum_5 = torch.ops.aten.sum.dim_IntList(pow_7, [1], True, dtype = torch.float32);  pow_7 = None
        sqrt_4 = torch.ops.aten.sqrt.default(sum_5);  sum_5 = None
        add_164 = torch.ops.aten.add.Tensor(sqrt_4, 1e-10)
        div_40 = torch.ops.aten.div.Tensor(add_65, add_164);  add_164 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_145, 2)
        sum_6 = torch.ops.aten.sum.dim_IntList(pow_8, [1], True, dtype = torch.float32);  pow_8 = None
        sqrt_5 = torch.ops.aten.sqrt.default(sum_6);  sum_6 = None
        add_165 = torch.ops.aten.add.Tensor(sqrt_5, 1e-10)
        div_41 = torch.ops.aten.div.Tensor(add_145, add_165);  add_165 = None
        sub_48 = torch.ops.aten.sub.Tensor(div_40, div_41);  div_40 = div_41 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sub_48, 2);  sub_48 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_79, 2)
        sum_7 = torch.ops.aten.sum.dim_IntList(pow_10, [1], True, dtype = torch.float32);  pow_10 = None
        sqrt_6 = torch.ops.aten.sqrt.default(sum_7);  sum_7 = None
        add_166 = torch.ops.aten.add.Tensor(sqrt_6, 1e-10)
        div_42 = torch.ops.aten.div.Tensor(add_79, add_166);  add_166 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(add_159, 2)
        sum_8 = torch.ops.aten.sum.dim_IntList(pow_11, [1], True, dtype = torch.float32);  pow_11 = None
        sqrt_7 = torch.ops.aten.sqrt.default(sum_8);  sum_8 = None
        add_167 = torch.ops.aten.add.Tensor(sqrt_7, 1e-10)
        div_43 = torch.ops.aten.div.Tensor(add_159, add_167);  add_167 = None
        sub_49 = torch.ops.aten.sub.Tensor(div_42, div_43);  div_42 = div_43 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(sub_49, 2);  sub_49 = None
        inductor_lookup_seed_default_34 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_41 = torch.ops.prims.inductor_random.default([32, 96, 16, 16], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_666 = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.float32);  inductor_random_default_41 = None
        clone = torch.ops.aten.clone.default(convert_element_type_666, memory_format = torch.channels_last);  convert_element_type_666 = None
        gt = torch.ops.aten.gt.Scalar(clone, 0.5);  clone = None
        mul_266 = torch.ops.aten.mul.Tensor(gt, pow_3);  pow_3 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, 2.0);  mul_266 = None
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_668 = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None
        convolution_44 = torch.ops.aten.convolution.default(convert_element_type_668, convert_element_type_667, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean = torch.ops.aten.mean.dim(convolution_44, [2, 3], True);  convolution_44 = None
        inductor_lookup_seed_default_35 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_40 = torch.ops.prims.inductor_random.default([32, 192, 8, 8], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_669 = torch.ops.prims.convert_element_type.default(inductor_random_default_40, torch.float32);  inductor_random_default_40 = None
        clone_1 = torch.ops.aten.clone.default(convert_element_type_669, memory_format = torch.channels_last);  convert_element_type_669 = None
        gt_1 = torch.ops.aten.gt.Scalar(clone_1, 0.5);  clone_1 = None
        mul_268 = torch.ops.aten.mul.Tensor(gt_1, pow_6);  pow_6 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, 2.0);  mul_268 = None
        convert_element_type_670 = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        convert_element_type_671 = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        convolution_45 = torch.ops.aten.convolution.default(convert_element_type_671, convert_element_type_670, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_1 = torch.ops.aten.mean.dim(convolution_45, [2, 3], True);  convolution_45 = None
        inductor_lookup_seed_default_36 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_39 = torch.ops.prims.inductor_random.default([32, 384, 4, 4], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_672 = torch.ops.prims.convert_element_type.default(inductor_random_default_39, torch.float32);  inductor_random_default_39 = None
        clone_2 = torch.ops.aten.clone.default(convert_element_type_672, memory_format = torch.channels_last);  convert_element_type_672 = None
        gt_2 = torch.ops.aten.gt.Scalar(clone_2, 0.5);  clone_2 = None
        mul_270 = torch.ops.aten.mul.Tensor(gt_2, pow_9);  pow_9 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, 2.0);  mul_270 = None
        convert_element_type_673 = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None
        convolution_46 = torch.ops.aten.convolution.default(convert_element_type_674, convert_element_type_673, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_2 = torch.ops.aten.mean.dim(convolution_46, [2, 3], True);  convolution_46 = None
        inductor_lookup_seed_default_37 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_38 = torch.ops.prims.inductor_random.default([32, 768, 2, 2], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(inductor_random_default_38, torch.float32);  inductor_random_default_38 = None
        clone_3 = torch.ops.aten.clone.default(convert_element_type_675, memory_format = torch.channels_last);  convert_element_type_675 = None
        gt_3 = torch.ops.aten.gt.Scalar(clone_3, 0.5);  clone_3 = None
        mul_272 = torch.ops.aten.mul.Tensor(gt_3, pow_12);  pow_12 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, 2.0);  mul_272 = None
        convert_element_type_676 = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None
        convolution_47 = torch.ops.aten.convolution.default(convert_element_type_677, convert_element_type_676, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_3 = torch.ops.aten.mean.dim(convolution_47, [2, 3], True);  convolution_47 = None
        add_168 = torch.ops.aten.add.Tensor(mean, 0);  mean = None
        add_169 = torch.ops.aten.add.Tensor(add_168, mean_1);  add_168 = mean_1 = None
        add_170 = torch.ops.aten.add.Tensor(add_169, mean_2);  add_169 = mean_2 = None
        add_171 = torch.ops.aten.add.Tensor(add_170, mean_3);  add_170 = mean_3 = None
        sub_51 = torch.ops.aten.sub.Tensor(primals_193, primals_189);  primals_193 = primals_189 = None
        div_45 = torch.ops.aten.div.Tensor(sub_51, primals_190);  sub_51 = primals_190 = None
        inductor_lookup_seed_default_38 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_37 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        lt_34 = torch.ops.aten.lt.Scalar(inductor_random_default_37, 0.9941176470588236);  inductor_random_default_37 = None
        convert_element_type_716 = torch.ops.prims.convert_element_type.default(lt_34, torch.float32)
        div_46 = torch.ops.aten.div.Tensor(convert_element_type_716, 0.9941176470588236);  convert_element_type_716 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_13, div_46);  mul_13 = div_46 = None
        add_181 = torch.ops.aten.add.Tensor(mul_288, add_5);  mul_288 = add_5 = None
        convert_element_type_719 = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16)
        convolution_51 = torch.ops.aten.convolution.default(convert_element_type_719, convert_element_type_40, convert_element_type_39, [1, 1], [3, 3], [1, 1], False, [0, 0], 96)
        permute_176 = torch.ops.aten.permute.default(convolution_51, [0, 2, 3, 1]);  convolution_51 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(permute_176, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_720, [3], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_182 = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_55 = torch.ops.aten.sub.Tensor(convert_element_type_720, getitem_95);  convert_element_type_720 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_47);  sub_55 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, primals_47);  mul_289 = None
        add_183 = torch.ops.aten.add.Tensor(mul_290, primals_48);  mul_290 = None
        convert_element_type_723 = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        view_152 = torch.ops.aten.view.default(convert_element_type_723, [8192, 96]);  convert_element_type_723 = None
        addmm_76 = torch.ops.aten.addmm.default(convert_element_type_43, view_152, permute_11)
        view_153 = torch.ops.aten.view.default(addmm_76, [32, 16, 16, 384])
        convert_element_type_727 = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        mul_291 = torch.ops.aten.mul.Tensor(convert_element_type_727, 0.5)
        mul_292 = torch.ops.aten.mul.Tensor(convert_element_type_727, 0.7071067811865476);  convert_element_type_727 = None
        erf_38 = torch.ops.aten.erf.default(mul_292);  mul_292 = None
        add_184 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_291, add_184);  mul_291 = add_184 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        view_154 = torch.ops.aten.view.default(convert_element_type_728, [8192, 384]);  convert_element_type_728 = None
        addmm_77 = torch.ops.aten.addmm.default(convert_element_type_51, view_154, permute_12)
        view_155 = torch.ops.aten.view.default(addmm_77, [32, 16, 16, 96])
        permute_179 = torch.ops.aten.permute.default(view_155, [0, 3, 1, 2]);  view_155 = None
        mul_294 = torch.ops.aten.mul.Tensor(primals_5, permute_179);  permute_179 = None
        inductor_lookup_seed_default_39 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_36 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        lt_35 = torch.ops.aten.lt.Scalar(inductor_random_default_36, 0.9882352941176471);  inductor_random_default_36 = None
        convert_element_type_734 = torch.ops.prims.convert_element_type.default(lt_35, torch.float32)
        div_47 = torch.ops.aten.div.Tensor(convert_element_type_734, 0.9882352941176471);  convert_element_type_734 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, div_47);  mul_294 = div_47 = None
        add_185 = torch.ops.aten.add.Tensor(mul_295, add_181);  mul_295 = add_181 = None
        permute_181 = torch.ops.aten.permute.default(add_185, [0, 2, 3, 1])
        var_mean_48 = torch.ops.aten.var_mean.correction(permute_181, [3], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_56 = torch.ops.aten.sub.Tensor(permute_181, getitem_97);  permute_181 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_48);  sub_56 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, primals_6);  mul_296 = None
        add_187 = torch.ops.aten.add.Tensor(mul_297, primals_7);  mul_297 = None
        permute_182 = torch.ops.aten.permute.default(add_187, [0, 3, 1, 2]);  add_187 = None
        convert_element_type_737 = torch.ops.prims.convert_element_type.default(permute_182, torch.bfloat16);  permute_182 = None
        convolution_52 = torch.ops.aten.convolution.default(convert_element_type_737, convert_element_type_58, convert_element_type_57, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_53 = torch.ops.aten.convolution.default(convolution_52, convert_element_type_61, convert_element_type_60, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_183 = torch.ops.aten.permute.default(convolution_53, [0, 2, 3, 1]);  convolution_53 = None
        convert_element_type_740 = torch.ops.prims.convert_element_type.default(permute_183, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_740, [3], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_57 = torch.ops.aten.sub.Tensor(convert_element_type_740, getitem_99);  convert_element_type_740 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_49);  sub_57 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, primals_57);  mul_298 = None
        add_189 = torch.ops.aten.add.Tensor(mul_299, primals_58);  mul_299 = None
        convert_element_type_743 = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        view_156 = torch.ops.aten.view.default(convert_element_type_743, [2048, 192]);  convert_element_type_743 = None
        addmm_78 = torch.ops.aten.addmm.default(convert_element_type_63, view_156, permute_18)
        view_157 = torch.ops.aten.view.default(addmm_78, [32, 8, 8, 768])
        convert_element_type_747 = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_300 = torch.ops.aten.mul.Tensor(convert_element_type_747, 0.5)
        mul_301 = torch.ops.aten.mul.Tensor(convert_element_type_747, 0.7071067811865476);  convert_element_type_747 = None
        erf_39 = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_190 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_300, add_190);  mul_300 = add_190 = None
        convert_element_type_748 = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        view_158 = torch.ops.aten.view.default(convert_element_type_748, [2048, 768]);  convert_element_type_748 = None
        addmm_79 = torch.ops.aten.addmm.default(convert_element_type_71, view_158, permute_19)
        view_159 = torch.ops.aten.view.default(addmm_79, [32, 8, 8, 192])
        permute_186 = torch.ops.aten.permute.default(view_159, [0, 3, 1, 2]);  view_159 = None
        mul_303 = torch.ops.aten.mul.Tensor(primals_8, permute_186);  permute_186 = None
        inductor_lookup_seed_default_40 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_35 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        lt_36 = torch.ops.aten.lt.Scalar(inductor_random_default_35, 0.9823529411764705);  inductor_random_default_35 = None
        convert_element_type_754 = torch.ops.prims.convert_element_type.default(lt_36, torch.float32)
        div_48 = torch.ops.aten.div.Tensor(convert_element_type_754, 0.9823529411764705);  convert_element_type_754 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_303, div_48);  mul_303 = div_48 = None
        add_191 = torch.ops.aten.add.Tensor(mul_304, convolution_52);  mul_304 = None
        convert_element_type_757 = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16)
        convolution_54 = torch.ops.aten.convolution.default(convert_element_type_757, convert_element_type_78, convert_element_type_77, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_187 = torch.ops.aten.permute.default(convolution_54, [0, 2, 3, 1]);  convolution_54 = None
        convert_element_type_758 = torch.ops.prims.convert_element_type.default(permute_187, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_758, [3], correction = 0, keepdim = True)
        getitem_100 = var_mean_50[0]
        getitem_101 = var_mean_50[1];  var_mean_50 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_58 = torch.ops.aten.sub.Tensor(convert_element_type_758, getitem_101);  convert_element_type_758 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_50);  sub_58 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, primals_65);  mul_305 = None
        add_193 = torch.ops.aten.add.Tensor(mul_306, primals_66);  mul_306 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None
        view_160 = torch.ops.aten.view.default(convert_element_type_761, [2048, 192]);  convert_element_type_761 = None
        addmm_80 = torch.ops.aten.addmm.default(convert_element_type_81, view_160, permute_22)
        view_161 = torch.ops.aten.view.default(addmm_80, [32, 8, 8, 768])
        convert_element_type_765 = torch.ops.prims.convert_element_type.default(view_161, torch.float32);  view_161 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_765, 0.5)
        mul_308 = torch.ops.aten.mul.Tensor(convert_element_type_765, 0.7071067811865476);  convert_element_type_765 = None
        erf_40 = torch.ops.aten.erf.default(mul_308);  mul_308 = None
        add_194 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_307, add_194);  mul_307 = add_194 = None
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None
        view_162 = torch.ops.aten.view.default(convert_element_type_766, [2048, 768]);  convert_element_type_766 = None
        addmm_81 = torch.ops.aten.addmm.default(convert_element_type_89, view_162, permute_23)
        view_163 = torch.ops.aten.view.default(addmm_81, [32, 8, 8, 192])
        permute_190 = torch.ops.aten.permute.default(view_163, [0, 3, 1, 2]);  view_163 = None
        mul_310 = torch.ops.aten.mul.Tensor(primals_9, permute_190);  permute_190 = None
        inductor_lookup_seed_default_41 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_34 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        lt_37 = torch.ops.aten.lt.Scalar(inductor_random_default_34, 0.9764705882352941);  inductor_random_default_34 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(lt_37, torch.float32)
        div_49 = torch.ops.aten.div.Tensor(convert_element_type_772, 0.9764705882352941);  convert_element_type_772 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, div_49);  mul_310 = div_49 = None
        add_195 = torch.ops.aten.add.Tensor(mul_311, add_191);  mul_311 = add_191 = None
        convert_element_type_775 = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16)
        convolution_55 = torch.ops.aten.convolution.default(convert_element_type_775, convert_element_type_96, convert_element_type_95, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
        permute_191 = torch.ops.aten.permute.default(convolution_55, [0, 2, 3, 1]);  convolution_55 = None
        convert_element_type_776 = torch.ops.prims.convert_element_type.default(permute_191, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_776, [3], correction = 0, keepdim = True)
        getitem_102 = var_mean_51[0]
        getitem_103 = var_mean_51[1];  var_mean_51 = None
        add_196 = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_59 = torch.ops.aten.sub.Tensor(convert_element_type_776, getitem_103);  convert_element_type_776 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_51);  sub_59 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, primals_73);  mul_312 = None
        add_197 = torch.ops.aten.add.Tensor(mul_313, primals_74);  mul_313 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None
        view_164 = torch.ops.aten.view.default(convert_element_type_779, [2048, 192]);  convert_element_type_779 = None
        addmm_82 = torch.ops.aten.addmm.default(convert_element_type_99, view_164, permute_26)
        view_165 = torch.ops.aten.view.default(addmm_82, [32, 8, 8, 768])
        convert_element_type_783 = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_314 = torch.ops.aten.mul.Tensor(convert_element_type_783, 0.5)
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_783, 0.7071067811865476);  convert_element_type_783 = None
        erf_41 = torch.ops.aten.erf.default(mul_315);  mul_315 = None
        add_198 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_314, add_198);  mul_314 = add_198 = None
        convert_element_type_784 = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16);  mul_316 = None
        view_166 = torch.ops.aten.view.default(convert_element_type_784, [2048, 768]);  convert_element_type_784 = None
        addmm_83 = torch.ops.aten.addmm.default(convert_element_type_107, view_166, permute_27)
        view_167 = torch.ops.aten.view.default(addmm_83, [32, 8, 8, 192])
        permute_194 = torch.ops.aten.permute.default(view_167, [0, 3, 1, 2]);  view_167 = None
        mul_317 = torch.ops.aten.mul.Tensor(primals_10, permute_194);  permute_194 = None
        inductor_lookup_seed_default_42 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_33 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        lt_38 = torch.ops.aten.lt.Scalar(inductor_random_default_33, 0.9705882352941176);  inductor_random_default_33 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(lt_38, torch.float32)
        div_50 = torch.ops.aten.div.Tensor(convert_element_type_790, 0.9705882352941176);  convert_element_type_790 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, div_50);  mul_317 = div_50 = None
        add_199 = torch.ops.aten.add.Tensor(mul_318, add_195);  mul_318 = add_195 = None
        permute_196 = torch.ops.aten.permute.default(add_199, [0, 2, 3, 1])
        var_mean_52 = torch.ops.aten.var_mean.correction(permute_196, [3], correction = 0, keepdim = True)
        getitem_104 = var_mean_52[0]
        getitem_105 = var_mean_52[1];  var_mean_52 = None
        add_200 = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_60 = torch.ops.aten.sub.Tensor(permute_196, getitem_105);  permute_196 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_52);  sub_60 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, primals_11);  mul_319 = None
        add_201 = torch.ops.aten.add.Tensor(mul_320, primals_12);  mul_320 = None
        permute_197 = torch.ops.aten.permute.default(add_201, [0, 3, 1, 2]);  add_201 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(permute_197, torch.bfloat16);  permute_197 = None
        convolution_56 = torch.ops.aten.convolution.default(convert_element_type_793, convert_element_type_114, convert_element_type_113, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_57 = torch.ops.aten.convolution.default(convolution_56, convert_element_type_117, convert_element_type_116, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_198 = torch.ops.aten.permute.default(convolution_57, [0, 2, 3, 1]);  convolution_57 = None
        convert_element_type_796 = torch.ops.prims.convert_element_type.default(permute_198, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_796, [3], correction = 0, keepdim = True)
        getitem_106 = var_mean_53[0]
        getitem_107 = var_mean_53[1];  var_mean_53 = None
        add_202 = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_61 = torch.ops.aten.sub.Tensor(convert_element_type_796, getitem_107);  convert_element_type_796 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_53);  sub_61 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, primals_83);  mul_321 = None
        add_203 = torch.ops.aten.add.Tensor(mul_322, primals_84);  mul_322 = None
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(add_203, torch.bfloat16);  add_203 = None
        view_168 = torch.ops.aten.view.default(convert_element_type_799, [512, 384]);  convert_element_type_799 = None
        addmm_84 = torch.ops.aten.addmm.default(convert_element_type_119, view_168, permute_33)
        view_169 = torch.ops.aten.view.default(addmm_84, [32, 4, 4, 1536])
        convert_element_type_803 = torch.ops.prims.convert_element_type.default(view_169, torch.float32);  view_169 = None
        mul_323 = torch.ops.aten.mul.Tensor(convert_element_type_803, 0.5)
        mul_324 = torch.ops.aten.mul.Tensor(convert_element_type_803, 0.7071067811865476);  convert_element_type_803 = None
        erf_42 = torch.ops.aten.erf.default(mul_324);  mul_324 = None
        add_204 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_323, add_204);  mul_323 = add_204 = None
        convert_element_type_804 = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None
        view_170 = torch.ops.aten.view.default(convert_element_type_804, [512, 1536]);  convert_element_type_804 = None
        addmm_85 = torch.ops.aten.addmm.default(convert_element_type_127, view_170, permute_34)
        view_171 = torch.ops.aten.view.default(addmm_85, [32, 4, 4, 384])
        permute_201 = torch.ops.aten.permute.default(view_171, [0, 3, 1, 2]);  view_171 = None
        mul_326 = torch.ops.aten.mul.Tensor(primals_13, permute_201);  permute_201 = None
        inductor_lookup_seed_default_43 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_32 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        lt_39 = torch.ops.aten.lt.Scalar(inductor_random_default_32, 0.9647058823529412);  inductor_random_default_32 = None
        convert_element_type_810 = torch.ops.prims.convert_element_type.default(lt_39, torch.float32)
        div_51 = torch.ops.aten.div.Tensor(convert_element_type_810, 0.9647058823529412);  convert_element_type_810 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_326, div_51);  mul_326 = div_51 = None
        add_205 = torch.ops.aten.add.Tensor(mul_327, convolution_56);  mul_327 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(add_205, torch.bfloat16)
        convolution_58 = torch.ops.aten.convolution.default(convert_element_type_813, convert_element_type_134, convert_element_type_133, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_202 = torch.ops.aten.permute.default(convolution_58, [0, 2, 3, 1]);  convolution_58 = None
        convert_element_type_814 = torch.ops.prims.convert_element_type.default(permute_202, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_814, [3], correction = 0, keepdim = True)
        getitem_108 = var_mean_54[0]
        getitem_109 = var_mean_54[1];  var_mean_54 = None
        add_206 = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_62 = torch.ops.aten.sub.Tensor(convert_element_type_814, getitem_109);  convert_element_type_814 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_54);  sub_62 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, primals_91);  mul_328 = None
        add_207 = torch.ops.aten.add.Tensor(mul_329, primals_92);  mul_329 = None
        convert_element_type_817 = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16);  add_207 = None
        view_172 = torch.ops.aten.view.default(convert_element_type_817, [512, 384]);  convert_element_type_817 = None
        addmm_86 = torch.ops.aten.addmm.default(convert_element_type_137, view_172, permute_37)
        view_173 = torch.ops.aten.view.default(addmm_86, [32, 4, 4, 1536])
        convert_element_type_821 = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_821, 0.5)
        mul_331 = torch.ops.aten.mul.Tensor(convert_element_type_821, 0.7071067811865476);  convert_element_type_821 = None
        erf_43 = torch.ops.aten.erf.default(mul_331);  mul_331 = None
        add_208 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_330, add_208);  mul_330 = add_208 = None
        convert_element_type_822 = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        view_174 = torch.ops.aten.view.default(convert_element_type_822, [512, 1536]);  convert_element_type_822 = None
        addmm_87 = torch.ops.aten.addmm.default(convert_element_type_145, view_174, permute_38)
        view_175 = torch.ops.aten.view.default(addmm_87, [32, 4, 4, 384])
        permute_205 = torch.ops.aten.permute.default(view_175, [0, 3, 1, 2]);  view_175 = None
        mul_333 = torch.ops.aten.mul.Tensor(primals_14, permute_205);  permute_205 = None
        inductor_lookup_seed_default_44 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_31 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        lt_40 = torch.ops.aten.lt.Scalar(inductor_random_default_31, 0.9588235294117647);  inductor_random_default_31 = None
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(lt_40, torch.float32)
        div_52 = torch.ops.aten.div.Tensor(convert_element_type_828, 0.9588235294117647);  convert_element_type_828 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_333, div_52);  mul_333 = div_52 = None
        add_209 = torch.ops.aten.add.Tensor(mul_334, add_205);  mul_334 = add_205 = None
        convert_element_type_831 = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16)
        convolution_59 = torch.ops.aten.convolution.default(convert_element_type_831, convert_element_type_152, convert_element_type_151, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_206 = torch.ops.aten.permute.default(convolution_59, [0, 2, 3, 1]);  convolution_59 = None
        convert_element_type_832 = torch.ops.prims.convert_element_type.default(permute_206, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_832, [3], correction = 0, keepdim = True)
        getitem_110 = var_mean_55[0]
        getitem_111 = var_mean_55[1];  var_mean_55 = None
        add_210 = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_63 = torch.ops.aten.sub.Tensor(convert_element_type_832, getitem_111);  convert_element_type_832 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_55);  sub_63 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, primals_99);  mul_335 = None
        add_211 = torch.ops.aten.add.Tensor(mul_336, primals_100);  mul_336 = None
        convert_element_type_835 = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16);  add_211 = None
        view_176 = torch.ops.aten.view.default(convert_element_type_835, [512, 384]);  convert_element_type_835 = None
        addmm_88 = torch.ops.aten.addmm.default(convert_element_type_155, view_176, permute_41)
        view_177 = torch.ops.aten.view.default(addmm_88, [32, 4, 4, 1536])
        convert_element_type_839 = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_337 = torch.ops.aten.mul.Tensor(convert_element_type_839, 0.5)
        mul_338 = torch.ops.aten.mul.Tensor(convert_element_type_839, 0.7071067811865476);  convert_element_type_839 = None
        erf_44 = torch.ops.aten.erf.default(mul_338);  mul_338 = None
        add_212 = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_337, add_212);  mul_337 = add_212 = None
        convert_element_type_840 = torch.ops.prims.convert_element_type.default(mul_339, torch.bfloat16);  mul_339 = None
        view_178 = torch.ops.aten.view.default(convert_element_type_840, [512, 1536]);  convert_element_type_840 = None
        addmm_89 = torch.ops.aten.addmm.default(convert_element_type_163, view_178, permute_42)
        view_179 = torch.ops.aten.view.default(addmm_89, [32, 4, 4, 384])
        permute_209 = torch.ops.aten.permute.default(view_179, [0, 3, 1, 2]);  view_179 = None
        mul_340 = torch.ops.aten.mul.Tensor(primals_15, permute_209);  permute_209 = None
        inductor_lookup_seed_default_45 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_30 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        lt_41 = torch.ops.aten.lt.Scalar(inductor_random_default_30, 0.9529411764705882);  inductor_random_default_30 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(lt_41, torch.float32)
        div_53 = torch.ops.aten.div.Tensor(convert_element_type_846, 0.9529411764705882);  convert_element_type_846 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, div_53);  mul_340 = div_53 = None
        add_213 = torch.ops.aten.add.Tensor(mul_341, add_209);  mul_341 = add_209 = None
        convert_element_type_849 = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16)
        convolution_60 = torch.ops.aten.convolution.default(convert_element_type_849, convert_element_type_170, convert_element_type_169, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_210 = torch.ops.aten.permute.default(convolution_60, [0, 2, 3, 1]);  convolution_60 = None
        convert_element_type_850 = torch.ops.prims.convert_element_type.default(permute_210, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_850, [3], correction = 0, keepdim = True)
        getitem_112 = var_mean_56[0]
        getitem_113 = var_mean_56[1];  var_mean_56 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_64 = torch.ops.aten.sub.Tensor(convert_element_type_850, getitem_113);  convert_element_type_850 = None
        mul_342 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_56);  sub_64 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_342, primals_107);  mul_342 = None
        add_215 = torch.ops.aten.add.Tensor(mul_343, primals_108);  mul_343 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None
        view_180 = torch.ops.aten.view.default(convert_element_type_853, [512, 384]);  convert_element_type_853 = None
        addmm_90 = torch.ops.aten.addmm.default(convert_element_type_173, view_180, permute_45)
        view_181 = torch.ops.aten.view.default(addmm_90, [32, 4, 4, 1536])
        convert_element_type_857 = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        mul_344 = torch.ops.aten.mul.Tensor(convert_element_type_857, 0.5)
        mul_345 = torch.ops.aten.mul.Tensor(convert_element_type_857, 0.7071067811865476);  convert_element_type_857 = None
        erf_45 = torch.ops.aten.erf.default(mul_345);  mul_345 = None
        add_216 = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_346 = torch.ops.aten.mul.Tensor(mul_344, add_216);  mul_344 = add_216 = None
        convert_element_type_858 = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        view_182 = torch.ops.aten.view.default(convert_element_type_858, [512, 1536]);  convert_element_type_858 = None
        addmm_91 = torch.ops.aten.addmm.default(convert_element_type_181, view_182, permute_46)
        view_183 = torch.ops.aten.view.default(addmm_91, [32, 4, 4, 384])
        permute_213 = torch.ops.aten.permute.default(view_183, [0, 3, 1, 2]);  view_183 = None
        mul_347 = torch.ops.aten.mul.Tensor(primals_16, permute_213);  permute_213 = None
        inductor_lookup_seed_default_46 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_29 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        lt_42 = torch.ops.aten.lt.Scalar(inductor_random_default_29, 0.9470588235294117);  inductor_random_default_29 = None
        convert_element_type_864 = torch.ops.prims.convert_element_type.default(lt_42, torch.float32)
        div_54 = torch.ops.aten.div.Tensor(convert_element_type_864, 0.9470588235294117);  convert_element_type_864 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, div_54);  mul_347 = div_54 = None
        add_217 = torch.ops.aten.add.Tensor(mul_348, add_213);  mul_348 = add_213 = None
        convert_element_type_867 = torch.ops.prims.convert_element_type.default(add_217, torch.bfloat16)
        convolution_61 = torch.ops.aten.convolution.default(convert_element_type_867, convert_element_type_188, convert_element_type_187, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_214 = torch.ops.aten.permute.default(convolution_61, [0, 2, 3, 1]);  convolution_61 = None
        convert_element_type_868 = torch.ops.prims.convert_element_type.default(permute_214, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_868, [3], correction = 0, keepdim = True)
        getitem_114 = var_mean_57[0]
        getitem_115 = var_mean_57[1];  var_mean_57 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_65 = torch.ops.aten.sub.Tensor(convert_element_type_868, getitem_115);  convert_element_type_868 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_57);  sub_65 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, primals_115);  mul_349 = None
        add_219 = torch.ops.aten.add.Tensor(mul_350, primals_116);  mul_350 = None
        convert_element_type_871 = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        view_184 = torch.ops.aten.view.default(convert_element_type_871, [512, 384]);  convert_element_type_871 = None
        addmm_92 = torch.ops.aten.addmm.default(convert_element_type_191, view_184, permute_49)
        view_185 = torch.ops.aten.view.default(addmm_92, [32, 4, 4, 1536])
        convert_element_type_875 = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None
        mul_351 = torch.ops.aten.mul.Tensor(convert_element_type_875, 0.5)
        mul_352 = torch.ops.aten.mul.Tensor(convert_element_type_875, 0.7071067811865476);  convert_element_type_875 = None
        erf_46 = torch.ops.aten.erf.default(mul_352);  mul_352 = None
        add_220 = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_351, add_220);  mul_351 = add_220 = None
        convert_element_type_876 = torch.ops.prims.convert_element_type.default(mul_353, torch.bfloat16);  mul_353 = None
        view_186 = torch.ops.aten.view.default(convert_element_type_876, [512, 1536]);  convert_element_type_876 = None
        addmm_93 = torch.ops.aten.addmm.default(convert_element_type_199, view_186, permute_50)
        view_187 = torch.ops.aten.view.default(addmm_93, [32, 4, 4, 384])
        permute_217 = torch.ops.aten.permute.default(view_187, [0, 3, 1, 2]);  view_187 = None
        mul_354 = torch.ops.aten.mul.Tensor(primals_17, permute_217);  permute_217 = None
        inductor_lookup_seed_default_47 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_28 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        lt_43 = torch.ops.aten.lt.Scalar(inductor_random_default_28, 0.9411764705882353);  inductor_random_default_28 = None
        convert_element_type_882 = torch.ops.prims.convert_element_type.default(lt_43, torch.float32)
        div_55 = torch.ops.aten.div.Tensor(convert_element_type_882, 0.9411764705882353);  convert_element_type_882 = None
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, div_55);  mul_354 = div_55 = None
        add_221 = torch.ops.aten.add.Tensor(mul_355, add_217);  mul_355 = add_217 = None
        convert_element_type_885 = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16)
        convolution_62 = torch.ops.aten.convolution.default(convert_element_type_885, convert_element_type_206, convert_element_type_205, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_218 = torch.ops.aten.permute.default(convolution_62, [0, 2, 3, 1]);  convolution_62 = None
        convert_element_type_886 = torch.ops.prims.convert_element_type.default(permute_218, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_886, [3], correction = 0, keepdim = True)
        getitem_116 = var_mean_58[0]
        getitem_117 = var_mean_58[1];  var_mean_58 = None
        add_222 = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_66 = torch.ops.aten.sub.Tensor(convert_element_type_886, getitem_117);  convert_element_type_886 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_58);  sub_66 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, primals_123);  mul_356 = None
        add_223 = torch.ops.aten.add.Tensor(mul_357, primals_124);  mul_357 = None
        convert_element_type_889 = torch.ops.prims.convert_element_type.default(add_223, torch.bfloat16);  add_223 = None
        view_188 = torch.ops.aten.view.default(convert_element_type_889, [512, 384]);  convert_element_type_889 = None
        addmm_94 = torch.ops.aten.addmm.default(convert_element_type_209, view_188, permute_53)
        view_189 = torch.ops.aten.view.default(addmm_94, [32, 4, 4, 1536])
        convert_element_type_893 = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        mul_358 = torch.ops.aten.mul.Tensor(convert_element_type_893, 0.5)
        mul_359 = torch.ops.aten.mul.Tensor(convert_element_type_893, 0.7071067811865476);  convert_element_type_893 = None
        erf_47 = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_224 = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_358, add_224);  mul_358 = add_224 = None
        convert_element_type_894 = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        view_190 = torch.ops.aten.view.default(convert_element_type_894, [512, 1536]);  convert_element_type_894 = None
        addmm_95 = torch.ops.aten.addmm.default(convert_element_type_217, view_190, permute_54)
        view_191 = torch.ops.aten.view.default(addmm_95, [32, 4, 4, 384])
        permute_221 = torch.ops.aten.permute.default(view_191, [0, 3, 1, 2]);  view_191 = None
        mul_361 = torch.ops.aten.mul.Tensor(primals_18, permute_221);  permute_221 = None
        inductor_lookup_seed_default_48 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_27 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        lt_44 = torch.ops.aten.lt.Scalar(inductor_random_default_27, 0.9352941176470588);  inductor_random_default_27 = None
        convert_element_type_900 = torch.ops.prims.convert_element_type.default(lt_44, torch.float32)
        div_56 = torch.ops.aten.div.Tensor(convert_element_type_900, 0.9352941176470588);  convert_element_type_900 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, div_56);  mul_361 = div_56 = None
        add_225 = torch.ops.aten.add.Tensor(mul_362, add_221);  mul_362 = add_221 = None
        convert_element_type_903 = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16)
        convolution_63 = torch.ops.aten.convolution.default(convert_element_type_903, convert_element_type_224, convert_element_type_223, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_222 = torch.ops.aten.permute.default(convolution_63, [0, 2, 3, 1]);  convolution_63 = None
        convert_element_type_904 = torch.ops.prims.convert_element_type.default(permute_222, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_904, [3], correction = 0, keepdim = True)
        getitem_118 = var_mean_59[0]
        getitem_119 = var_mean_59[1];  var_mean_59 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_67 = torch.ops.aten.sub.Tensor(convert_element_type_904, getitem_119);  convert_element_type_904 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_59);  sub_67 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, primals_131);  mul_363 = None
        add_227 = torch.ops.aten.add.Tensor(mul_364, primals_132);  mul_364 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(add_227, torch.bfloat16);  add_227 = None
        view_192 = torch.ops.aten.view.default(convert_element_type_907, [512, 384]);  convert_element_type_907 = None
        addmm_96 = torch.ops.aten.addmm.default(convert_element_type_227, view_192, permute_57)
        view_193 = torch.ops.aten.view.default(addmm_96, [32, 4, 4, 1536])
        convert_element_type_911 = torch.ops.prims.convert_element_type.default(view_193, torch.float32);  view_193 = None
        mul_365 = torch.ops.aten.mul.Tensor(convert_element_type_911, 0.5)
        mul_366 = torch.ops.aten.mul.Tensor(convert_element_type_911, 0.7071067811865476);  convert_element_type_911 = None
        erf_48 = torch.ops.aten.erf.default(mul_366);  mul_366 = None
        add_228 = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_365, add_228);  mul_365 = add_228 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16);  mul_367 = None
        view_194 = torch.ops.aten.view.default(convert_element_type_912, [512, 1536]);  convert_element_type_912 = None
        addmm_97 = torch.ops.aten.addmm.default(convert_element_type_235, view_194, permute_58)
        view_195 = torch.ops.aten.view.default(addmm_97, [32, 4, 4, 384])
        permute_225 = torch.ops.aten.permute.default(view_195, [0, 3, 1, 2]);  view_195 = None
        mul_368 = torch.ops.aten.mul.Tensor(primals_19, permute_225);  permute_225 = None
        inductor_lookup_seed_default_49 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_26 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        lt_45 = torch.ops.aten.lt.Scalar(inductor_random_default_26, 0.9294117647058824);  inductor_random_default_26 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(lt_45, torch.float32)
        div_57 = torch.ops.aten.div.Tensor(convert_element_type_918, 0.9294117647058824);  convert_element_type_918 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_368, div_57);  mul_368 = div_57 = None
        add_229 = torch.ops.aten.add.Tensor(mul_369, add_225);  mul_369 = add_225 = None
        convert_element_type_921 = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16)
        convolution_64 = torch.ops.aten.convolution.default(convert_element_type_921, convert_element_type_242, convert_element_type_241, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_226 = torch.ops.aten.permute.default(convolution_64, [0, 2, 3, 1]);  convolution_64 = None
        convert_element_type_922 = torch.ops.prims.convert_element_type.default(permute_226, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_922, [3], correction = 0, keepdim = True)
        getitem_120 = var_mean_60[0]
        getitem_121 = var_mean_60[1];  var_mean_60 = None
        add_230 = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
        sub_68 = torch.ops.aten.sub.Tensor(convert_element_type_922, getitem_121);  convert_element_type_922 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_60);  sub_68 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, primals_139);  mul_370 = None
        add_231 = torch.ops.aten.add.Tensor(mul_371, primals_140);  mul_371 = None
        convert_element_type_925 = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16);  add_231 = None
        view_196 = torch.ops.aten.view.default(convert_element_type_925, [512, 384]);  convert_element_type_925 = None
        addmm_98 = torch.ops.aten.addmm.default(convert_element_type_245, view_196, permute_61)
        view_197 = torch.ops.aten.view.default(addmm_98, [32, 4, 4, 1536])
        convert_element_type_929 = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_372 = torch.ops.aten.mul.Tensor(convert_element_type_929, 0.5)
        mul_373 = torch.ops.aten.mul.Tensor(convert_element_type_929, 0.7071067811865476);  convert_element_type_929 = None
        erf_49 = torch.ops.aten.erf.default(mul_373);  mul_373 = None
        add_232 = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_372, add_232);  mul_372 = add_232 = None
        convert_element_type_930 = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None
        view_198 = torch.ops.aten.view.default(convert_element_type_930, [512, 1536]);  convert_element_type_930 = None
        addmm_99 = torch.ops.aten.addmm.default(convert_element_type_253, view_198, permute_62)
        view_199 = torch.ops.aten.view.default(addmm_99, [32, 4, 4, 384])
        permute_229 = torch.ops.aten.permute.default(view_199, [0, 3, 1, 2]);  view_199 = None
        mul_375 = torch.ops.aten.mul.Tensor(primals_20, permute_229);  permute_229 = None
        inductor_lookup_seed_default_50 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_25 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        lt_46 = torch.ops.aten.lt.Scalar(inductor_random_default_25, 0.9235294117647059);  inductor_random_default_25 = None
        convert_element_type_936 = torch.ops.prims.convert_element_type.default(lt_46, torch.float32)
        div_58 = torch.ops.aten.div.Tensor(convert_element_type_936, 0.9235294117647059);  convert_element_type_936 = None
        mul_376 = torch.ops.aten.mul.Tensor(mul_375, div_58);  mul_375 = div_58 = None
        add_233 = torch.ops.aten.add.Tensor(mul_376, add_229);  mul_376 = add_229 = None
        convert_element_type_939 = torch.ops.prims.convert_element_type.default(add_233, torch.bfloat16)
        convolution_65 = torch.ops.aten.convolution.default(convert_element_type_939, convert_element_type_260, convert_element_type_259, [1, 1], [3, 3], [1, 1], False, [0, 0], 384)
        permute_230 = torch.ops.aten.permute.default(convolution_65, [0, 2, 3, 1]);  convolution_65 = None
        convert_element_type_940 = torch.ops.prims.convert_element_type.default(permute_230, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_940, [3], correction = 0, keepdim = True)
        getitem_122 = var_mean_61[0]
        getitem_123 = var_mean_61[1];  var_mean_61 = None
        add_234 = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_69 = torch.ops.aten.sub.Tensor(convert_element_type_940, getitem_123);  convert_element_type_940 = None
        mul_377 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_61);  sub_69 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_377, primals_147);  mul_377 = None
        add_235 = torch.ops.aten.add.Tensor(mul_378, primals_148);  mul_378 = None
        convert_element_type_943 = torch.ops.prims.convert_element_type.default(add_235, torch.bfloat16);  add_235 = None
        view_200 = torch.ops.aten.view.default(convert_element_type_943, [512, 384]);  convert_element_type_943 = None
        addmm_100 = torch.ops.aten.addmm.default(convert_element_type_263, view_200, permute_65)
        view_201 = torch.ops.aten.view.default(addmm_100, [32, 4, 4, 1536])
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(view_201, torch.float32);  view_201 = None
        mul_379 = torch.ops.aten.mul.Tensor(convert_element_type_947, 0.5)
        mul_380 = torch.ops.aten.mul.Tensor(convert_element_type_947, 0.7071067811865476);  convert_element_type_947 = None
        erf_50 = torch.ops.aten.erf.default(mul_380);  mul_380 = None
        add_236 = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_379, add_236);  mul_379 = add_236 = None
        convert_element_type_948 = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16);  mul_381 = None
        view_202 = torch.ops.aten.view.default(convert_element_type_948, [512, 1536]);  convert_element_type_948 = None
        addmm_101 = torch.ops.aten.addmm.default(convert_element_type_271, view_202, permute_66)
        view_203 = torch.ops.aten.view.default(addmm_101, [32, 4, 4, 384])
        permute_233 = torch.ops.aten.permute.default(view_203, [0, 3, 1, 2]);  view_203 = None
        mul_382 = torch.ops.aten.mul.Tensor(primals_21, permute_233);  permute_233 = None
        inductor_lookup_seed_default_51 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_24 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        lt_47 = torch.ops.aten.lt.Scalar(inductor_random_default_24, 0.9176470588235294);  inductor_random_default_24 = None
        convert_element_type_954 = torch.ops.prims.convert_element_type.default(lt_47, torch.float32)
        div_59 = torch.ops.aten.div.Tensor(convert_element_type_954, 0.9176470588235294);  convert_element_type_954 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, div_59);  mul_382 = div_59 = None
        add_237 = torch.ops.aten.add.Tensor(mul_383, add_233);  mul_383 = add_233 = None
        permute_235 = torch.ops.aten.permute.default(add_237, [0, 2, 3, 1])
        var_mean_62 = torch.ops.aten.var_mean.correction(permute_235, [3], correction = 0, keepdim = True)
        getitem_124 = var_mean_62[0]
        getitem_125 = var_mean_62[1];  var_mean_62 = None
        add_238 = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_70 = torch.ops.aten.sub.Tensor(permute_235, getitem_125);  permute_235 = None
        mul_384 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_62);  sub_70 = None
        mul_385 = torch.ops.aten.mul.Tensor(mul_384, primals_22);  mul_384 = None
        add_239 = torch.ops.aten.add.Tensor(mul_385, primals_23);  mul_385 = None
        permute_236 = torch.ops.aten.permute.default(add_239, [0, 3, 1, 2]);  add_239 = None
        convert_element_type_957 = torch.ops.prims.convert_element_type.default(permute_236, torch.bfloat16);  permute_236 = None
        convolution_66 = torch.ops.aten.convolution.default(convert_element_type_957, convert_element_type_278, convert_element_type_277, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_67 = torch.ops.aten.convolution.default(convolution_66, convert_element_type_281, convert_element_type_280, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_237 = torch.ops.aten.permute.default(convolution_67, [0, 2, 3, 1]);  convolution_67 = None
        convert_element_type_960 = torch.ops.prims.convert_element_type.default(permute_237, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_960, [3], correction = 0, keepdim = True)
        getitem_126 = var_mean_63[0]
        getitem_127 = var_mean_63[1];  var_mean_63 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_71 = torch.ops.aten.sub.Tensor(convert_element_type_960, getitem_127);  convert_element_type_960 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_63);  sub_71 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, primals_157);  mul_386 = None
        add_241 = torch.ops.aten.add.Tensor(mul_387, primals_158);  mul_387 = None
        convert_element_type_963 = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None
        view_204 = torch.ops.aten.view.default(convert_element_type_963, [128, 768]);  convert_element_type_963 = None
        addmm_102 = torch.ops.aten.addmm.default(convert_element_type_283, view_204, permute_72)
        view_205 = torch.ops.aten.view.default(addmm_102, [32, 2, 2, 3072])
        convert_element_type_967 = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        mul_388 = torch.ops.aten.mul.Tensor(convert_element_type_967, 0.5)
        mul_389 = torch.ops.aten.mul.Tensor(convert_element_type_967, 0.7071067811865476);  convert_element_type_967 = None
        erf_51 = torch.ops.aten.erf.default(mul_389);  mul_389 = None
        add_242 = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_388, add_242);  mul_388 = add_242 = None
        convert_element_type_968 = torch.ops.prims.convert_element_type.default(mul_390, torch.bfloat16);  mul_390 = None
        view_206 = torch.ops.aten.view.default(convert_element_type_968, [128, 3072]);  convert_element_type_968 = None
        addmm_103 = torch.ops.aten.addmm.default(convert_element_type_291, view_206, permute_73)
        view_207 = torch.ops.aten.view.default(addmm_103, [32, 2, 2, 768])
        permute_240 = torch.ops.aten.permute.default(view_207, [0, 3, 1, 2]);  view_207 = None
        mul_391 = torch.ops.aten.mul.Tensor(primals_24, permute_240);  permute_240 = None
        inductor_lookup_seed_default_52 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_23 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        lt_48 = torch.ops.aten.lt.Scalar(inductor_random_default_23, 0.9117647058823529);  inductor_random_default_23 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(lt_48, torch.float32)
        div_60 = torch.ops.aten.div.Tensor(convert_element_type_974, 0.9117647058823529);  convert_element_type_974 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, div_60);  mul_391 = div_60 = None
        add_243 = torch.ops.aten.add.Tensor(mul_392, convolution_66);  mul_392 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16)
        convolution_68 = torch.ops.aten.convolution.default(convert_element_type_977, convert_element_type_298, convert_element_type_297, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_241 = torch.ops.aten.permute.default(convolution_68, [0, 2, 3, 1]);  convolution_68 = None
        convert_element_type_978 = torch.ops.prims.convert_element_type.default(permute_241, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_978, [3], correction = 0, keepdim = True)
        getitem_128 = var_mean_64[0]
        getitem_129 = var_mean_64[1];  var_mean_64 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_72 = torch.ops.aten.sub.Tensor(convert_element_type_978, getitem_129);  convert_element_type_978 = None
        mul_393 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_64);  sub_72 = None
        mul_394 = torch.ops.aten.mul.Tensor(mul_393, primals_165);  mul_393 = None
        add_245 = torch.ops.aten.add.Tensor(mul_394, primals_166);  mul_394 = None
        convert_element_type_981 = torch.ops.prims.convert_element_type.default(add_245, torch.bfloat16);  add_245 = None
        view_208 = torch.ops.aten.view.default(convert_element_type_981, [128, 768]);  convert_element_type_981 = None
        addmm_104 = torch.ops.aten.addmm.default(convert_element_type_301, view_208, permute_76)
        view_209 = torch.ops.aten.view.default(addmm_104, [32, 2, 2, 3072])
        convert_element_type_985 = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        mul_395 = torch.ops.aten.mul.Tensor(convert_element_type_985, 0.5)
        mul_396 = torch.ops.aten.mul.Tensor(convert_element_type_985, 0.7071067811865476);  convert_element_type_985 = None
        erf_52 = torch.ops.aten.erf.default(mul_396);  mul_396 = None
        add_246 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_395, add_246);  mul_395 = add_246 = None
        convert_element_type_986 = torch.ops.prims.convert_element_type.default(mul_397, torch.bfloat16);  mul_397 = None
        view_210 = torch.ops.aten.view.default(convert_element_type_986, [128, 3072]);  convert_element_type_986 = None
        addmm_105 = torch.ops.aten.addmm.default(convert_element_type_309, view_210, permute_77)
        view_211 = torch.ops.aten.view.default(addmm_105, [32, 2, 2, 768])
        permute_244 = torch.ops.aten.permute.default(view_211, [0, 3, 1, 2]);  view_211 = None
        mul_398 = torch.ops.aten.mul.Tensor(primals_25, permute_244);  permute_244 = None
        inductor_lookup_seed_default_53 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_22 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        lt_49 = torch.ops.aten.lt.Scalar(inductor_random_default_22, 0.9058823529411765);  inductor_random_default_22 = None
        convert_element_type_992 = torch.ops.prims.convert_element_type.default(lt_49, torch.float32)
        div_61 = torch.ops.aten.div.Tensor(convert_element_type_992, 0.9058823529411765);  convert_element_type_992 = None
        mul_399 = torch.ops.aten.mul.Tensor(mul_398, div_61);  mul_398 = div_61 = None
        add_247 = torch.ops.aten.add.Tensor(mul_399, add_243);  mul_399 = add_243 = None
        convert_element_type_995 = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16)
        convolution_69 = torch.ops.aten.convolution.default(convert_element_type_995, convert_element_type_316, convert_element_type_315, [1, 1], [3, 3], [1, 1], False, [0, 0], 768)
        permute_245 = torch.ops.aten.permute.default(convolution_69, [0, 2, 3, 1]);  convolution_69 = None
        convert_element_type_996 = torch.ops.prims.convert_element_type.default(permute_245, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_996, [3], correction = 0, keepdim = True)
        getitem_130 = var_mean_65[0]
        getitem_131 = var_mean_65[1];  var_mean_65 = None
        add_248 = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        sub_73 = torch.ops.aten.sub.Tensor(convert_element_type_996, getitem_131);  convert_element_type_996 = None
        mul_400 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_65);  sub_73 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, primals_173);  mul_400 = None
        add_249 = torch.ops.aten.add.Tensor(mul_401, primals_174);  mul_401 = None
        convert_element_type_999 = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16);  add_249 = None
        view_212 = torch.ops.aten.view.default(convert_element_type_999, [128, 768]);  convert_element_type_999 = None
        addmm_106 = torch.ops.aten.addmm.default(convert_element_type_319, view_212, permute_80)
        view_213 = torch.ops.aten.view.default(addmm_106, [32, 2, 2, 3072])
        convert_element_type_1003 = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        mul_402 = torch.ops.aten.mul.Tensor(convert_element_type_1003, 0.5)
        mul_403 = torch.ops.aten.mul.Tensor(convert_element_type_1003, 0.7071067811865476);  convert_element_type_1003 = None
        erf_53 = torch.ops.aten.erf.default(mul_403);  mul_403 = None
        add_250 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_402, add_250);  mul_402 = add_250 = None
        convert_element_type_1004 = torch.ops.prims.convert_element_type.default(mul_404, torch.bfloat16);  mul_404 = None
        view_214 = torch.ops.aten.view.default(convert_element_type_1004, [128, 3072]);  convert_element_type_1004 = None
        addmm_107 = torch.ops.aten.addmm.default(convert_element_type_327, view_214, permute_81)
        view_215 = torch.ops.aten.view.default(addmm_107, [32, 2, 2, 768])
        permute_248 = torch.ops.aten.permute.default(view_215, [0, 3, 1, 2]);  view_215 = None
        mul_405 = torch.ops.aten.mul.Tensor(primals_26, permute_248);  permute_248 = None
        inductor_lookup_seed_default_54 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_21 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        lt_50 = torch.ops.aten.lt.Scalar(inductor_random_default_21, 0.9);  inductor_random_default_21 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(lt_50, torch.float32)
        div_62 = torch.ops.aten.div.Tensor(convert_element_type_1010, 0.9);  convert_element_type_1010 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_405, div_62);  mul_405 = div_62 = None
        add_251 = torch.ops.aten.add.Tensor(mul_406, add_247);  mul_406 = add_247 = None
        convert_element_type_1013 = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None
        convolution_70 = torch.ops.aten.convolution.default(convert_element_type_1013, convert_element_type_1, convert_element_type, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type = None
        permute_249 = torch.ops.aten.permute.default(convolution_70, [0, 2, 3, 1]);  convolution_70 = None
        convert_element_type_1014 = torch.ops.prims.convert_element_type.default(permute_249, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_1014, [3], correction = 0, keepdim = True)
        getitem_132 = var_mean_66[0]
        getitem_133 = var_mean_66[1];  var_mean_66 = None
        add_252 = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_74 = torch.ops.aten.sub.Tensor(convert_element_type_1014, getitem_133);  convert_element_type_1014 = None
        mul_407 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_66);  sub_74 = None
        mul_408 = torch.ops.aten.mul.Tensor(mul_407, primals_1);  mul_407 = None
        add_253 = torch.ops.aten.add.Tensor(mul_408, primals_2);  mul_408 = primals_2 = None
        permute_250 = torch.ops.aten.permute.default(add_253, [0, 3, 1, 2]);  add_253 = None
        convert_element_type_1017 = torch.ops.prims.convert_element_type.default(permute_250, torch.bfloat16)
        convolution_71 = torch.ops.aten.convolution.default(convert_element_type_1017, convert_element_type_5, convert_element_type_4, [1, 1], [3, 3], [1, 1], False, [0, 0], 96);  convert_element_type_4 = None
        permute_251 = torch.ops.aten.permute.default(convolution_71, [0, 2, 3, 1]);  convolution_71 = None
        convert_element_type_1018 = torch.ops.prims.convert_element_type.default(permute_251, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_1018, [3], correction = 0, keepdim = True)
        getitem_134 = var_mean_67[0]
        getitem_135 = var_mean_67[1];  var_mean_67 = None
        add_254 = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_75 = torch.ops.aten.sub.Tensor(convert_element_type_1018, getitem_135);  convert_element_type_1018 = None
        mul_409 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_67);  sub_75 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, primals_31);  mul_409 = None
        add_255 = torch.ops.aten.add.Tensor(mul_410, primals_32);  mul_410 = primals_32 = None
        convert_element_type_1021 = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16);  add_255 = None
        view_216 = torch.ops.aten.view.default(convert_element_type_1021, [8192, 96]);  convert_element_type_1021 = None
        addmm_108 = torch.ops.aten.addmm.default(convert_element_type_8, view_216, permute_3);  convert_element_type_8 = None
        view_217 = torch.ops.aten.view.default(addmm_108, [32, 16, 16, 384])
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_411 = torch.ops.aten.mul.Tensor(convert_element_type_1025, 0.5)
        mul_412 = torch.ops.aten.mul.Tensor(convert_element_type_1025, 0.7071067811865476);  convert_element_type_1025 = None
        erf_54 = torch.ops.aten.erf.default(mul_412);  mul_412 = None
        add_256 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_411, add_256);  mul_411 = add_256 = None
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(mul_413, torch.bfloat16);  mul_413 = None
        view_218 = torch.ops.aten.view.default(convert_element_type_1026, [8192, 384]);  convert_element_type_1026 = None
        addmm_109 = torch.ops.aten.addmm.default(convert_element_type_16, view_218, permute_4);  convert_element_type_16 = None
        view_219 = torch.ops.aten.view.default(addmm_109, [32, 16, 16, 96])
        permute_254 = torch.ops.aten.permute.default(view_219, [0, 3, 1, 2]);  view_219 = None
        mul_414 = torch.ops.aten.mul.Tensor(primals_3, permute_254);  permute_254 = None
        add_257 = torch.ops.aten.add.Tensor(mul_414, permute_250);  mul_414 = permute_250 = None
        convert_element_type_1034 = torch.ops.prims.convert_element_type.default(add_257, torch.bfloat16)
        convolution_72 = torch.ops.aten.convolution.default(convert_element_type_1034, convert_element_type_22, convert_element_type_21, [1, 1], [3, 3], [1, 1], False, [0, 0], 96);  convert_element_type_21 = None
        permute_255 = torch.ops.aten.permute.default(convolution_72, [0, 2, 3, 1]);  convolution_72 = None
        convert_element_type_1035 = torch.ops.prims.convert_element_type.default(permute_255, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_1035, [3], correction = 0, keepdim = True)
        getitem_136 = var_mean_68[0]
        getitem_137 = var_mean_68[1];  var_mean_68 = None
        add_258 = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_76 = torch.ops.aten.sub.Tensor(convert_element_type_1035, getitem_137);  convert_element_type_1035 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_68);  sub_76 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, primals_39);  mul_415 = None
        add_259 = torch.ops.aten.add.Tensor(mul_416, primals_40);  mul_416 = primals_40 = None
        convert_element_type_1038 = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        view_220 = torch.ops.aten.view.default(convert_element_type_1038, [8192, 96]);  convert_element_type_1038 = None
        addmm_110 = torch.ops.aten.addmm.default(convert_element_type_25, view_220, permute_7);  convert_element_type_25 = None
        view_221 = torch.ops.aten.view.default(addmm_110, [32, 16, 16, 384])
        convert_element_type_1042 = torch.ops.prims.convert_element_type.default(view_221, torch.float32);  view_221 = None
        mul_417 = torch.ops.aten.mul.Tensor(convert_element_type_1042, 0.5)
        mul_418 = torch.ops.aten.mul.Tensor(convert_element_type_1042, 0.7071067811865476);  convert_element_type_1042 = None
        erf_55 = torch.ops.aten.erf.default(mul_418);  mul_418 = None
        add_260 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_419 = torch.ops.aten.mul.Tensor(mul_417, add_260);  mul_417 = add_260 = None
        convert_element_type_1043 = torch.ops.prims.convert_element_type.default(mul_419, torch.bfloat16);  mul_419 = None
        view_222 = torch.ops.aten.view.default(convert_element_type_1043, [8192, 384]);  convert_element_type_1043 = None
        addmm_111 = torch.ops.aten.addmm.default(convert_element_type_33, view_222, permute_8);  convert_element_type_33 = None
        view_223 = torch.ops.aten.view.default(addmm_111, [32, 16, 16, 96])
        permute_258 = torch.ops.aten.permute.default(view_223, [0, 3, 1, 2]);  view_223 = None
        mul_420 = torch.ops.aten.mul.Tensor(primals_4, permute_258);  permute_258 = None
        inductor_lookup_seed_default_55 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_20 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        lt_51 = torch.ops.aten.lt.Scalar(inductor_random_default_20, 0.9941176470588236);  inductor_random_default_20 = None
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(lt_51, torch.float32)
        div_63 = torch.ops.aten.div.Tensor(convert_element_type_1049, 0.9941176470588236);  convert_element_type_1049 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_420, div_63);  mul_420 = div_63 = None
        add_261 = torch.ops.aten.add.Tensor(mul_421, add_257);  mul_421 = add_257 = None
        convert_element_type_1052 = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16)
        convolution_73 = torch.ops.aten.convolution.default(convert_element_type_1052, convert_element_type_40, convert_element_type_39, [1, 1], [3, 3], [1, 1], False, [0, 0], 96);  convert_element_type_39 = None
        permute_259 = torch.ops.aten.permute.default(convolution_73, [0, 2, 3, 1]);  convolution_73 = None
        convert_element_type_1053 = torch.ops.prims.convert_element_type.default(permute_259, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_1053, [3], correction = 0, keepdim = True)
        getitem_138 = var_mean_69[0]
        getitem_139 = var_mean_69[1];  var_mean_69 = None
        add_262 = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_77 = torch.ops.aten.sub.Tensor(convert_element_type_1053, getitem_139);  convert_element_type_1053 = None
        mul_422 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_69);  sub_77 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_422, primals_47);  mul_422 = None
        add_263 = torch.ops.aten.add.Tensor(mul_423, primals_48);  mul_423 = primals_48 = None
        convert_element_type_1056 = torch.ops.prims.convert_element_type.default(add_263, torch.bfloat16);  add_263 = None
        view_224 = torch.ops.aten.view.default(convert_element_type_1056, [8192, 96]);  convert_element_type_1056 = None
        addmm_112 = torch.ops.aten.addmm.default(convert_element_type_43, view_224, permute_11);  convert_element_type_43 = None
        view_225 = torch.ops.aten.view.default(addmm_112, [32, 16, 16, 384])
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        mul_424 = torch.ops.aten.mul.Tensor(convert_element_type_1060, 0.5)
        mul_425 = torch.ops.aten.mul.Tensor(convert_element_type_1060, 0.7071067811865476);  convert_element_type_1060 = None
        erf_56 = torch.ops.aten.erf.default(mul_425);  mul_425 = None
        add_264 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_426 = torch.ops.aten.mul.Tensor(mul_424, add_264);  mul_424 = add_264 = None
        convert_element_type_1061 = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        view_226 = torch.ops.aten.view.default(convert_element_type_1061, [8192, 384]);  convert_element_type_1061 = None
        addmm_113 = torch.ops.aten.addmm.default(convert_element_type_51, view_226, permute_12);  convert_element_type_51 = None
        view_227 = torch.ops.aten.view.default(addmm_113, [32, 16, 16, 96])
        permute_262 = torch.ops.aten.permute.default(view_227, [0, 3, 1, 2]);  view_227 = None
        mul_427 = torch.ops.aten.mul.Tensor(primals_5, permute_262);  permute_262 = None
        inductor_lookup_seed_default_56 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_19 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        lt_52 = torch.ops.aten.lt.Scalar(inductor_random_default_19, 0.9882352941176471);  inductor_random_default_19 = None
        convert_element_type_1067 = torch.ops.prims.convert_element_type.default(lt_52, torch.float32)
        div_64 = torch.ops.aten.div.Tensor(convert_element_type_1067, 0.9882352941176471);  convert_element_type_1067 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_427, div_64);  mul_427 = div_64 = None
        add_265 = torch.ops.aten.add.Tensor(mul_428, add_261);  mul_428 = add_261 = None
        permute_264 = torch.ops.aten.permute.default(add_265, [0, 2, 3, 1])
        var_mean_70 = torch.ops.aten.var_mean.correction(permute_264, [3], correction = 0, keepdim = True)
        getitem_140 = var_mean_70[0]
        getitem_141 = var_mean_70[1];  var_mean_70 = None
        add_266 = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_78 = torch.ops.aten.sub.Tensor(permute_264, getitem_141);  permute_264 = None
        mul_429 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_70);  sub_78 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_429, primals_6);  mul_429 = None
        add_267 = torch.ops.aten.add.Tensor(mul_430, primals_7);  mul_430 = primals_7 = None
        permute_265 = torch.ops.aten.permute.default(add_267, [0, 3, 1, 2]);  add_267 = None
        convert_element_type_1070 = torch.ops.prims.convert_element_type.default(permute_265, torch.bfloat16);  permute_265 = None
        convolution_74 = torch.ops.aten.convolution.default(convert_element_type_1070, convert_element_type_58, convert_element_type_57, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_57 = None
        convolution_75 = torch.ops.aten.convolution.default(convolution_74, convert_element_type_61, convert_element_type_60, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  convert_element_type_60 = None
        permute_266 = torch.ops.aten.permute.default(convolution_75, [0, 2, 3, 1]);  convolution_75 = None
        convert_element_type_1073 = torch.ops.prims.convert_element_type.default(permute_266, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_1073, [3], correction = 0, keepdim = True)
        getitem_142 = var_mean_71[0]
        getitem_143 = var_mean_71[1];  var_mean_71 = None
        add_268 = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_79 = torch.ops.aten.sub.Tensor(convert_element_type_1073, getitem_143);  convert_element_type_1073 = None
        mul_431 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_71);  sub_79 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_431, primals_57);  mul_431 = None
        add_269 = torch.ops.aten.add.Tensor(mul_432, primals_58);  mul_432 = primals_58 = None
        convert_element_type_1076 = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None
        view_228 = torch.ops.aten.view.default(convert_element_type_1076, [2048, 192]);  convert_element_type_1076 = None
        addmm_114 = torch.ops.aten.addmm.default(convert_element_type_63, view_228, permute_18);  convert_element_type_63 = None
        view_229 = torch.ops.aten.view.default(addmm_114, [32, 8, 8, 768])
        convert_element_type_1080 = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None
        mul_433 = torch.ops.aten.mul.Tensor(convert_element_type_1080, 0.5)
        mul_434 = torch.ops.aten.mul.Tensor(convert_element_type_1080, 0.7071067811865476);  convert_element_type_1080 = None
        erf_57 = torch.ops.aten.erf.default(mul_434);  mul_434 = None
        add_270 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_433, add_270);  mul_433 = add_270 = None
        convert_element_type_1081 = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None
        view_230 = torch.ops.aten.view.default(convert_element_type_1081, [2048, 768]);  convert_element_type_1081 = None
        addmm_115 = torch.ops.aten.addmm.default(convert_element_type_71, view_230, permute_19);  convert_element_type_71 = None
        view_231 = torch.ops.aten.view.default(addmm_115, [32, 8, 8, 192])
        permute_269 = torch.ops.aten.permute.default(view_231, [0, 3, 1, 2]);  view_231 = None
        mul_436 = torch.ops.aten.mul.Tensor(primals_8, permute_269);  permute_269 = None
        inductor_lookup_seed_default_57 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_18 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        lt_53 = torch.ops.aten.lt.Scalar(inductor_random_default_18, 0.9823529411764705);  inductor_random_default_18 = None
        convert_element_type_1087 = torch.ops.prims.convert_element_type.default(lt_53, torch.float32)
        div_65 = torch.ops.aten.div.Tensor(convert_element_type_1087, 0.9823529411764705);  convert_element_type_1087 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_436, div_65);  mul_436 = div_65 = None
        add_271 = torch.ops.aten.add.Tensor(mul_437, convolution_74);  mul_437 = None
        convert_element_type_1090 = torch.ops.prims.convert_element_type.default(add_271, torch.bfloat16)
        convolution_76 = torch.ops.aten.convolution.default(convert_element_type_1090, convert_element_type_78, convert_element_type_77, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  convert_element_type_77 = None
        permute_270 = torch.ops.aten.permute.default(convolution_76, [0, 2, 3, 1]);  convolution_76 = None
        convert_element_type_1091 = torch.ops.prims.convert_element_type.default(permute_270, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_1091, [3], correction = 0, keepdim = True)
        getitem_144 = var_mean_72[0]
        getitem_145 = var_mean_72[1];  var_mean_72 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_80 = torch.ops.aten.sub.Tensor(convert_element_type_1091, getitem_145);  convert_element_type_1091 = None
        mul_438 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_72);  sub_80 = None
        mul_439 = torch.ops.aten.mul.Tensor(mul_438, primals_65);  mul_438 = None
        add_273 = torch.ops.aten.add.Tensor(mul_439, primals_66);  mul_439 = primals_66 = None
        convert_element_type_1094 = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16);  add_273 = None
        view_232 = torch.ops.aten.view.default(convert_element_type_1094, [2048, 192]);  convert_element_type_1094 = None
        addmm_116 = torch.ops.aten.addmm.default(convert_element_type_81, view_232, permute_22);  convert_element_type_81 = None
        view_233 = torch.ops.aten.view.default(addmm_116, [32, 8, 8, 768])
        convert_element_type_1098 = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        mul_440 = torch.ops.aten.mul.Tensor(convert_element_type_1098, 0.5)
        mul_441 = torch.ops.aten.mul.Tensor(convert_element_type_1098, 0.7071067811865476);  convert_element_type_1098 = None
        erf_58 = torch.ops.aten.erf.default(mul_441);  mul_441 = None
        add_274 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_442 = torch.ops.aten.mul.Tensor(mul_440, add_274);  mul_440 = add_274 = None
        convert_element_type_1099 = torch.ops.prims.convert_element_type.default(mul_442, torch.bfloat16);  mul_442 = None
        view_234 = torch.ops.aten.view.default(convert_element_type_1099, [2048, 768]);  convert_element_type_1099 = None
        addmm_117 = torch.ops.aten.addmm.default(convert_element_type_89, view_234, permute_23);  convert_element_type_89 = None
        view_235 = torch.ops.aten.view.default(addmm_117, [32, 8, 8, 192])
        permute_273 = torch.ops.aten.permute.default(view_235, [0, 3, 1, 2]);  view_235 = None
        mul_443 = torch.ops.aten.mul.Tensor(primals_9, permute_273);  permute_273 = None
        inductor_lookup_seed_default_58 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_17 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        lt_54 = torch.ops.aten.lt.Scalar(inductor_random_default_17, 0.9764705882352941);  inductor_random_default_17 = None
        convert_element_type_1105 = torch.ops.prims.convert_element_type.default(lt_54, torch.float32)
        div_66 = torch.ops.aten.div.Tensor(convert_element_type_1105, 0.9764705882352941);  convert_element_type_1105 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_443, div_66);  mul_443 = div_66 = None
        add_275 = torch.ops.aten.add.Tensor(mul_444, add_271);  mul_444 = add_271 = None
        convert_element_type_1108 = torch.ops.prims.convert_element_type.default(add_275, torch.bfloat16)
        convolution_77 = torch.ops.aten.convolution.default(convert_element_type_1108, convert_element_type_96, convert_element_type_95, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  convert_element_type_95 = None
        permute_274 = torch.ops.aten.permute.default(convolution_77, [0, 2, 3, 1]);  convolution_77 = None
        convert_element_type_1109 = torch.ops.prims.convert_element_type.default(permute_274, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_1109, [3], correction = 0, keepdim = True)
        getitem_146 = var_mean_73[0]
        getitem_147 = var_mean_73[1];  var_mean_73 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_81 = torch.ops.aten.sub.Tensor(convert_element_type_1109, getitem_147);  convert_element_type_1109 = None
        mul_445 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_73);  sub_81 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, primals_73);  mul_445 = None
        add_277 = torch.ops.aten.add.Tensor(mul_446, primals_74);  mul_446 = primals_74 = None
        convert_element_type_1112 = torch.ops.prims.convert_element_type.default(add_277, torch.bfloat16);  add_277 = None
        view_236 = torch.ops.aten.view.default(convert_element_type_1112, [2048, 192]);  convert_element_type_1112 = None
        addmm_118 = torch.ops.aten.addmm.default(convert_element_type_99, view_236, permute_26);  convert_element_type_99 = None
        view_237 = torch.ops.aten.view.default(addmm_118, [32, 8, 8, 768])
        convert_element_type_1116 = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        mul_447 = torch.ops.aten.mul.Tensor(convert_element_type_1116, 0.5)
        mul_448 = torch.ops.aten.mul.Tensor(convert_element_type_1116, 0.7071067811865476);  convert_element_type_1116 = None
        erf_59 = torch.ops.aten.erf.default(mul_448);  mul_448 = None
        add_278 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_447, add_278);  mul_447 = add_278 = None
        convert_element_type_1117 = torch.ops.prims.convert_element_type.default(mul_449, torch.bfloat16);  mul_449 = None
        view_238 = torch.ops.aten.view.default(convert_element_type_1117, [2048, 768]);  convert_element_type_1117 = None
        addmm_119 = torch.ops.aten.addmm.default(convert_element_type_107, view_238, permute_27);  convert_element_type_107 = None
        view_239 = torch.ops.aten.view.default(addmm_119, [32, 8, 8, 192])
        permute_277 = torch.ops.aten.permute.default(view_239, [0, 3, 1, 2]);  view_239 = None
        mul_450 = torch.ops.aten.mul.Tensor(primals_10, permute_277);  permute_277 = None
        inductor_lookup_seed_default_59 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_16 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        lt_55 = torch.ops.aten.lt.Scalar(inductor_random_default_16, 0.9705882352941176);  inductor_random_default_16 = None
        convert_element_type_1123 = torch.ops.prims.convert_element_type.default(lt_55, torch.float32)
        div_67 = torch.ops.aten.div.Tensor(convert_element_type_1123, 0.9705882352941176);  convert_element_type_1123 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_450, div_67);  mul_450 = div_67 = None
        add_279 = torch.ops.aten.add.Tensor(mul_451, add_275);  mul_451 = add_275 = None
        permute_279 = torch.ops.aten.permute.default(add_279, [0, 2, 3, 1])
        var_mean_74 = torch.ops.aten.var_mean.correction(permute_279, [3], correction = 0, keepdim = True)
        getitem_148 = var_mean_74[0]
        getitem_149 = var_mean_74[1];  var_mean_74 = None
        add_280 = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_82 = torch.ops.aten.sub.Tensor(permute_279, getitem_149);  permute_279 = None
        mul_452 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_74);  sub_82 = None
        mul_453 = torch.ops.aten.mul.Tensor(mul_452, primals_11);  mul_452 = None
        add_281 = torch.ops.aten.add.Tensor(mul_453, primals_12);  mul_453 = primals_12 = None
        permute_280 = torch.ops.aten.permute.default(add_281, [0, 3, 1, 2]);  add_281 = None
        convert_element_type_1126 = torch.ops.prims.convert_element_type.default(permute_280, torch.bfloat16);  permute_280 = None
        convolution_78 = torch.ops.aten.convolution.default(convert_element_type_1126, convert_element_type_114, convert_element_type_113, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_113 = None
        convolution_79 = torch.ops.aten.convolution.default(convolution_78, convert_element_type_117, convert_element_type_116, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_116 = None
        permute_281 = torch.ops.aten.permute.default(convolution_79, [0, 2, 3, 1]);  convolution_79 = None
        convert_element_type_1129 = torch.ops.prims.convert_element_type.default(permute_281, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_1129, [3], correction = 0, keepdim = True)
        getitem_150 = var_mean_75[0]
        getitem_151 = var_mean_75[1];  var_mean_75 = None
        add_282 = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_83 = torch.ops.aten.sub.Tensor(convert_element_type_1129, getitem_151);  convert_element_type_1129 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_75);  sub_83 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, primals_83);  mul_454 = None
        add_283 = torch.ops.aten.add.Tensor(mul_455, primals_84);  mul_455 = primals_84 = None
        convert_element_type_1132 = torch.ops.prims.convert_element_type.default(add_283, torch.bfloat16);  add_283 = None
        view_240 = torch.ops.aten.view.default(convert_element_type_1132, [512, 384]);  convert_element_type_1132 = None
        addmm_120 = torch.ops.aten.addmm.default(convert_element_type_119, view_240, permute_33);  convert_element_type_119 = None
        view_241 = torch.ops.aten.view.default(addmm_120, [32, 4, 4, 1536])
        convert_element_type_1136 = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        mul_456 = torch.ops.aten.mul.Tensor(convert_element_type_1136, 0.5)
        mul_457 = torch.ops.aten.mul.Tensor(convert_element_type_1136, 0.7071067811865476);  convert_element_type_1136 = None
        erf_60 = torch.ops.aten.erf.default(mul_457);  mul_457 = None
        add_284 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_456, add_284);  mul_456 = add_284 = None
        convert_element_type_1137 = torch.ops.prims.convert_element_type.default(mul_458, torch.bfloat16);  mul_458 = None
        view_242 = torch.ops.aten.view.default(convert_element_type_1137, [512, 1536]);  convert_element_type_1137 = None
        addmm_121 = torch.ops.aten.addmm.default(convert_element_type_127, view_242, permute_34);  convert_element_type_127 = None
        view_243 = torch.ops.aten.view.default(addmm_121, [32, 4, 4, 384])
        permute_284 = torch.ops.aten.permute.default(view_243, [0, 3, 1, 2]);  view_243 = None
        mul_459 = torch.ops.aten.mul.Tensor(primals_13, permute_284);  permute_284 = None
        inductor_lookup_seed_default_60 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_15 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        lt_56 = torch.ops.aten.lt.Scalar(inductor_random_default_15, 0.9647058823529412);  inductor_random_default_15 = None
        convert_element_type_1143 = torch.ops.prims.convert_element_type.default(lt_56, torch.float32)
        div_68 = torch.ops.aten.div.Tensor(convert_element_type_1143, 0.9647058823529412);  convert_element_type_1143 = None
        mul_460 = torch.ops.aten.mul.Tensor(mul_459, div_68);  mul_459 = div_68 = None
        add_285 = torch.ops.aten.add.Tensor(mul_460, convolution_78);  mul_460 = None
        convert_element_type_1146 = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16)
        convolution_80 = torch.ops.aten.convolution.default(convert_element_type_1146, convert_element_type_134, convert_element_type_133, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_133 = None
        permute_285 = torch.ops.aten.permute.default(convolution_80, [0, 2, 3, 1]);  convolution_80 = None
        convert_element_type_1147 = torch.ops.prims.convert_element_type.default(permute_285, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_1147, [3], correction = 0, keepdim = True)
        getitem_152 = var_mean_76[0]
        getitem_153 = var_mean_76[1];  var_mean_76 = None
        add_286 = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_84 = torch.ops.aten.sub.Tensor(convert_element_type_1147, getitem_153);  convert_element_type_1147 = None
        mul_461 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_76);  sub_84 = None
        mul_462 = torch.ops.aten.mul.Tensor(mul_461, primals_91);  mul_461 = None
        add_287 = torch.ops.aten.add.Tensor(mul_462, primals_92);  mul_462 = primals_92 = None
        convert_element_type_1150 = torch.ops.prims.convert_element_type.default(add_287, torch.bfloat16);  add_287 = None
        view_244 = torch.ops.aten.view.default(convert_element_type_1150, [512, 384]);  convert_element_type_1150 = None
        addmm_122 = torch.ops.aten.addmm.default(convert_element_type_137, view_244, permute_37);  convert_element_type_137 = None
        view_245 = torch.ops.aten.view.default(addmm_122, [32, 4, 4, 1536])
        convert_element_type_1154 = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        mul_463 = torch.ops.aten.mul.Tensor(convert_element_type_1154, 0.5)
        mul_464 = torch.ops.aten.mul.Tensor(convert_element_type_1154, 0.7071067811865476);  convert_element_type_1154 = None
        erf_61 = torch.ops.aten.erf.default(mul_464);  mul_464 = None
        add_288 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_463, add_288);  mul_463 = add_288 = None
        convert_element_type_1155 = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16);  mul_465 = None
        view_246 = torch.ops.aten.view.default(convert_element_type_1155, [512, 1536]);  convert_element_type_1155 = None
        addmm_123 = torch.ops.aten.addmm.default(convert_element_type_145, view_246, permute_38);  convert_element_type_145 = None
        view_247 = torch.ops.aten.view.default(addmm_123, [32, 4, 4, 384])
        permute_288 = torch.ops.aten.permute.default(view_247, [0, 3, 1, 2]);  view_247 = None
        mul_466 = torch.ops.aten.mul.Tensor(primals_14, permute_288);  permute_288 = None
        inductor_lookup_seed_default_61 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_14 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        lt_57 = torch.ops.aten.lt.Scalar(inductor_random_default_14, 0.9588235294117647);  inductor_random_default_14 = None
        convert_element_type_1161 = torch.ops.prims.convert_element_type.default(lt_57, torch.float32)
        div_69 = torch.ops.aten.div.Tensor(convert_element_type_1161, 0.9588235294117647);  convert_element_type_1161 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, div_69);  mul_466 = div_69 = None
        add_289 = torch.ops.aten.add.Tensor(mul_467, add_285);  mul_467 = add_285 = None
        convert_element_type_1164 = torch.ops.prims.convert_element_type.default(add_289, torch.bfloat16)
        convolution_81 = torch.ops.aten.convolution.default(convert_element_type_1164, convert_element_type_152, convert_element_type_151, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_151 = None
        permute_289 = torch.ops.aten.permute.default(convolution_81, [0, 2, 3, 1]);  convolution_81 = None
        convert_element_type_1165 = torch.ops.prims.convert_element_type.default(permute_289, torch.float32)
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_1165, [3], correction = 0, keepdim = True)
        getitem_154 = var_mean_77[0]
        getitem_155 = var_mean_77[1];  var_mean_77 = None
        add_290 = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_85 = torch.ops.aten.sub.Tensor(convert_element_type_1165, getitem_155);  convert_element_type_1165 = None
        mul_468 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_77);  sub_85 = None
        mul_469 = torch.ops.aten.mul.Tensor(mul_468, primals_99);  mul_468 = None
        add_291 = torch.ops.aten.add.Tensor(mul_469, primals_100);  mul_469 = primals_100 = None
        convert_element_type_1168 = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16);  add_291 = None
        view_248 = torch.ops.aten.view.default(convert_element_type_1168, [512, 384]);  convert_element_type_1168 = None
        addmm_124 = torch.ops.aten.addmm.default(convert_element_type_155, view_248, permute_41);  convert_element_type_155 = None
        view_249 = torch.ops.aten.view.default(addmm_124, [32, 4, 4, 1536])
        convert_element_type_1172 = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_470 = torch.ops.aten.mul.Tensor(convert_element_type_1172, 0.5)
        mul_471 = torch.ops.aten.mul.Tensor(convert_element_type_1172, 0.7071067811865476);  convert_element_type_1172 = None
        erf_62 = torch.ops.aten.erf.default(mul_471);  mul_471 = None
        add_292 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_470, add_292);  mul_470 = add_292 = None
        convert_element_type_1173 = torch.ops.prims.convert_element_type.default(mul_472, torch.bfloat16);  mul_472 = None
        view_250 = torch.ops.aten.view.default(convert_element_type_1173, [512, 1536]);  convert_element_type_1173 = None
        addmm_125 = torch.ops.aten.addmm.default(convert_element_type_163, view_250, permute_42);  convert_element_type_163 = None
        view_251 = torch.ops.aten.view.default(addmm_125, [32, 4, 4, 384])
        permute_292 = torch.ops.aten.permute.default(view_251, [0, 3, 1, 2]);  view_251 = None
        mul_473 = torch.ops.aten.mul.Tensor(primals_15, permute_292);  permute_292 = None
        inductor_lookup_seed_default_62 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_13 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        lt_58 = torch.ops.aten.lt.Scalar(inductor_random_default_13, 0.9529411764705882);  inductor_random_default_13 = None
        convert_element_type_1179 = torch.ops.prims.convert_element_type.default(lt_58, torch.float32)
        div_70 = torch.ops.aten.div.Tensor(convert_element_type_1179, 0.9529411764705882);  convert_element_type_1179 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, div_70);  mul_473 = div_70 = None
        add_293 = torch.ops.aten.add.Tensor(mul_474, add_289);  mul_474 = add_289 = None
        convert_element_type_1182 = torch.ops.prims.convert_element_type.default(add_293, torch.bfloat16)
        convolution_82 = torch.ops.aten.convolution.default(convert_element_type_1182, convert_element_type_170, convert_element_type_169, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_169 = None
        permute_293 = torch.ops.aten.permute.default(convolution_82, [0, 2, 3, 1]);  convolution_82 = None
        convert_element_type_1183 = torch.ops.prims.convert_element_type.default(permute_293, torch.float32)
        var_mean_78 = torch.ops.aten.var_mean.correction(convert_element_type_1183, [3], correction = 0, keepdim = True)
        getitem_156 = var_mean_78[0]
        getitem_157 = var_mean_78[1];  var_mean_78 = None
        add_294 = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        sub_86 = torch.ops.aten.sub.Tensor(convert_element_type_1183, getitem_157);  convert_element_type_1183 = None
        mul_475 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_78);  sub_86 = None
        mul_476 = torch.ops.aten.mul.Tensor(mul_475, primals_107);  mul_475 = None
        add_295 = torch.ops.aten.add.Tensor(mul_476, primals_108);  mul_476 = primals_108 = None
        convert_element_type_1186 = torch.ops.prims.convert_element_type.default(add_295, torch.bfloat16);  add_295 = None
        view_252 = torch.ops.aten.view.default(convert_element_type_1186, [512, 384]);  convert_element_type_1186 = None
        addmm_126 = torch.ops.aten.addmm.default(convert_element_type_173, view_252, permute_45);  convert_element_type_173 = None
        view_253 = torch.ops.aten.view.default(addmm_126, [32, 4, 4, 1536])
        convert_element_type_1190 = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_477 = torch.ops.aten.mul.Tensor(convert_element_type_1190, 0.5)
        mul_478 = torch.ops.aten.mul.Tensor(convert_element_type_1190, 0.7071067811865476);  convert_element_type_1190 = None
        erf_63 = torch.ops.aten.erf.default(mul_478);  mul_478 = None
        add_296 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_477, add_296);  mul_477 = add_296 = None
        convert_element_type_1191 = torch.ops.prims.convert_element_type.default(mul_479, torch.bfloat16);  mul_479 = None
        view_254 = torch.ops.aten.view.default(convert_element_type_1191, [512, 1536]);  convert_element_type_1191 = None
        addmm_127 = torch.ops.aten.addmm.default(convert_element_type_181, view_254, permute_46);  convert_element_type_181 = None
        view_255 = torch.ops.aten.view.default(addmm_127, [32, 4, 4, 384])
        permute_296 = torch.ops.aten.permute.default(view_255, [0, 3, 1, 2]);  view_255 = None
        mul_480 = torch.ops.aten.mul.Tensor(primals_16, permute_296);  permute_296 = None
        inductor_lookup_seed_default_63 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_12 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        lt_59 = torch.ops.aten.lt.Scalar(inductor_random_default_12, 0.9470588235294117);  inductor_random_default_12 = None
        convert_element_type_1197 = torch.ops.prims.convert_element_type.default(lt_59, torch.float32)
        div_71 = torch.ops.aten.div.Tensor(convert_element_type_1197, 0.9470588235294117);  convert_element_type_1197 = None
        mul_481 = torch.ops.aten.mul.Tensor(mul_480, div_71);  mul_480 = div_71 = None
        add_297 = torch.ops.aten.add.Tensor(mul_481, add_293);  mul_481 = add_293 = None
        convert_element_type_1200 = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16)
        convolution_83 = torch.ops.aten.convolution.default(convert_element_type_1200, convert_element_type_188, convert_element_type_187, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_187 = None
        permute_297 = torch.ops.aten.permute.default(convolution_83, [0, 2, 3, 1]);  convolution_83 = None
        convert_element_type_1201 = torch.ops.prims.convert_element_type.default(permute_297, torch.float32)
        var_mean_79 = torch.ops.aten.var_mean.correction(convert_element_type_1201, [3], correction = 0, keepdim = True)
        getitem_158 = var_mean_79[0]
        getitem_159 = var_mean_79[1];  var_mean_79 = None
        add_298 = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_87 = torch.ops.aten.sub.Tensor(convert_element_type_1201, getitem_159);  convert_element_type_1201 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_79);  sub_87 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_482, primals_115);  mul_482 = None
        add_299 = torch.ops.aten.add.Tensor(mul_483, primals_116);  mul_483 = primals_116 = None
        convert_element_type_1204 = torch.ops.prims.convert_element_type.default(add_299, torch.bfloat16);  add_299 = None
        view_256 = torch.ops.aten.view.default(convert_element_type_1204, [512, 384]);  convert_element_type_1204 = None
        addmm_128 = torch.ops.aten.addmm.default(convert_element_type_191, view_256, permute_49);  convert_element_type_191 = None
        view_257 = torch.ops.aten.view.default(addmm_128, [32, 4, 4, 1536])
        convert_element_type_1208 = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        mul_484 = torch.ops.aten.mul.Tensor(convert_element_type_1208, 0.5)
        mul_485 = torch.ops.aten.mul.Tensor(convert_element_type_1208, 0.7071067811865476);  convert_element_type_1208 = None
        erf_64 = torch.ops.aten.erf.default(mul_485);  mul_485 = None
        add_300 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_486 = torch.ops.aten.mul.Tensor(mul_484, add_300);  mul_484 = add_300 = None
        convert_element_type_1209 = torch.ops.prims.convert_element_type.default(mul_486, torch.bfloat16);  mul_486 = None
        view_258 = torch.ops.aten.view.default(convert_element_type_1209, [512, 1536]);  convert_element_type_1209 = None
        addmm_129 = torch.ops.aten.addmm.default(convert_element_type_199, view_258, permute_50);  convert_element_type_199 = None
        view_259 = torch.ops.aten.view.default(addmm_129, [32, 4, 4, 384])
        permute_300 = torch.ops.aten.permute.default(view_259, [0, 3, 1, 2]);  view_259 = None
        mul_487 = torch.ops.aten.mul.Tensor(primals_17, permute_300);  permute_300 = None
        inductor_lookup_seed_default_64 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_11 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        lt_60 = torch.ops.aten.lt.Scalar(inductor_random_default_11, 0.9411764705882353);  inductor_random_default_11 = None
        convert_element_type_1215 = torch.ops.prims.convert_element_type.default(lt_60, torch.float32)
        div_72 = torch.ops.aten.div.Tensor(convert_element_type_1215, 0.9411764705882353);  convert_element_type_1215 = None
        mul_488 = torch.ops.aten.mul.Tensor(mul_487, div_72);  mul_487 = div_72 = None
        add_301 = torch.ops.aten.add.Tensor(mul_488, add_297);  mul_488 = add_297 = None
        convert_element_type_1218 = torch.ops.prims.convert_element_type.default(add_301, torch.bfloat16)
        convolution_84 = torch.ops.aten.convolution.default(convert_element_type_1218, convert_element_type_206, convert_element_type_205, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_205 = None
        permute_301 = torch.ops.aten.permute.default(convolution_84, [0, 2, 3, 1]);  convolution_84 = None
        convert_element_type_1219 = torch.ops.prims.convert_element_type.default(permute_301, torch.float32)
        var_mean_80 = torch.ops.aten.var_mean.correction(convert_element_type_1219, [3], correction = 0, keepdim = True)
        getitem_160 = var_mean_80[0]
        getitem_161 = var_mean_80[1];  var_mean_80 = None
        add_302 = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        sub_88 = torch.ops.aten.sub.Tensor(convert_element_type_1219, getitem_161);  convert_element_type_1219 = None
        mul_489 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_80);  sub_88 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_489, primals_123);  mul_489 = None
        add_303 = torch.ops.aten.add.Tensor(mul_490, primals_124);  mul_490 = primals_124 = None
        convert_element_type_1222 = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16);  add_303 = None
        view_260 = torch.ops.aten.view.default(convert_element_type_1222, [512, 384]);  convert_element_type_1222 = None
        addmm_130 = torch.ops.aten.addmm.default(convert_element_type_209, view_260, permute_53);  convert_element_type_209 = None
        view_261 = torch.ops.aten.view.default(addmm_130, [32, 4, 4, 1536])
        convert_element_type_1226 = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_491 = torch.ops.aten.mul.Tensor(convert_element_type_1226, 0.5)
        mul_492 = torch.ops.aten.mul.Tensor(convert_element_type_1226, 0.7071067811865476);  convert_element_type_1226 = None
        erf_65 = torch.ops.aten.erf.default(mul_492);  mul_492 = None
        add_304 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_493 = torch.ops.aten.mul.Tensor(mul_491, add_304);  mul_491 = add_304 = None
        convert_element_type_1227 = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None
        view_262 = torch.ops.aten.view.default(convert_element_type_1227, [512, 1536]);  convert_element_type_1227 = None
        addmm_131 = torch.ops.aten.addmm.default(convert_element_type_217, view_262, permute_54);  convert_element_type_217 = None
        view_263 = torch.ops.aten.view.default(addmm_131, [32, 4, 4, 384])
        permute_304 = torch.ops.aten.permute.default(view_263, [0, 3, 1, 2]);  view_263 = None
        mul_494 = torch.ops.aten.mul.Tensor(primals_18, permute_304);  permute_304 = None
        inductor_lookup_seed_default_65 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_10 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        lt_61 = torch.ops.aten.lt.Scalar(inductor_random_default_10, 0.9352941176470588);  inductor_random_default_10 = None
        convert_element_type_1233 = torch.ops.prims.convert_element_type.default(lt_61, torch.float32)
        div_73 = torch.ops.aten.div.Tensor(convert_element_type_1233, 0.9352941176470588);  convert_element_type_1233 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_494, div_73);  mul_494 = div_73 = None
        add_305 = torch.ops.aten.add.Tensor(mul_495, add_301);  mul_495 = add_301 = None
        convert_element_type_1236 = torch.ops.prims.convert_element_type.default(add_305, torch.bfloat16)
        convolution_85 = torch.ops.aten.convolution.default(convert_element_type_1236, convert_element_type_224, convert_element_type_223, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_223 = None
        permute_305 = torch.ops.aten.permute.default(convolution_85, [0, 2, 3, 1]);  convolution_85 = None
        convert_element_type_1237 = torch.ops.prims.convert_element_type.default(permute_305, torch.float32)
        var_mean_81 = torch.ops.aten.var_mean.correction(convert_element_type_1237, [3], correction = 0, keepdim = True)
        getitem_162 = var_mean_81[0]
        getitem_163 = var_mean_81[1];  var_mean_81 = None
        add_306 = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_89 = torch.ops.aten.sub.Tensor(convert_element_type_1237, getitem_163);  convert_element_type_1237 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_81);  sub_89 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, primals_131);  mul_496 = None
        add_307 = torch.ops.aten.add.Tensor(mul_497, primals_132);  mul_497 = primals_132 = None
        convert_element_type_1240 = torch.ops.prims.convert_element_type.default(add_307, torch.bfloat16);  add_307 = None
        view_264 = torch.ops.aten.view.default(convert_element_type_1240, [512, 384]);  convert_element_type_1240 = None
        addmm_132 = torch.ops.aten.addmm.default(convert_element_type_227, view_264, permute_57);  convert_element_type_227 = None
        view_265 = torch.ops.aten.view.default(addmm_132, [32, 4, 4, 1536])
        convert_element_type_1244 = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_498 = torch.ops.aten.mul.Tensor(convert_element_type_1244, 0.5)
        mul_499 = torch.ops.aten.mul.Tensor(convert_element_type_1244, 0.7071067811865476);  convert_element_type_1244 = None
        erf_66 = torch.ops.aten.erf.default(mul_499);  mul_499 = None
        add_308 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_498, add_308);  mul_498 = add_308 = None
        convert_element_type_1245 = torch.ops.prims.convert_element_type.default(mul_500, torch.bfloat16);  mul_500 = None
        view_266 = torch.ops.aten.view.default(convert_element_type_1245, [512, 1536]);  convert_element_type_1245 = None
        addmm_133 = torch.ops.aten.addmm.default(convert_element_type_235, view_266, permute_58);  convert_element_type_235 = None
        view_267 = torch.ops.aten.view.default(addmm_133, [32, 4, 4, 384])
        permute_308 = torch.ops.aten.permute.default(view_267, [0, 3, 1, 2]);  view_267 = None
        mul_501 = torch.ops.aten.mul.Tensor(primals_19, permute_308);  permute_308 = None
        inductor_lookup_seed_default_66 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_9 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        lt_62 = torch.ops.aten.lt.Scalar(inductor_random_default_9, 0.9294117647058824);  inductor_random_default_9 = None
        convert_element_type_1251 = torch.ops.prims.convert_element_type.default(lt_62, torch.float32)
        div_74 = torch.ops.aten.div.Tensor(convert_element_type_1251, 0.9294117647058824);  convert_element_type_1251 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_501, div_74);  mul_501 = div_74 = None
        add_309 = torch.ops.aten.add.Tensor(mul_502, add_305);  mul_502 = add_305 = None
        convert_element_type_1254 = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16)
        convolution_86 = torch.ops.aten.convolution.default(convert_element_type_1254, convert_element_type_242, convert_element_type_241, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_241 = None
        permute_309 = torch.ops.aten.permute.default(convolution_86, [0, 2, 3, 1]);  convolution_86 = None
        convert_element_type_1255 = torch.ops.prims.convert_element_type.default(permute_309, torch.float32)
        var_mean_82 = torch.ops.aten.var_mean.correction(convert_element_type_1255, [3], correction = 0, keepdim = True)
        getitem_164 = var_mean_82[0]
        getitem_165 = var_mean_82[1];  var_mean_82 = None
        add_310 = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_90 = torch.ops.aten.sub.Tensor(convert_element_type_1255, getitem_165);  convert_element_type_1255 = None
        mul_503 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_82);  sub_90 = None
        mul_504 = torch.ops.aten.mul.Tensor(mul_503, primals_139);  mul_503 = None
        add_311 = torch.ops.aten.add.Tensor(mul_504, primals_140);  mul_504 = primals_140 = None
        convert_element_type_1258 = torch.ops.prims.convert_element_type.default(add_311, torch.bfloat16);  add_311 = None
        view_268 = torch.ops.aten.view.default(convert_element_type_1258, [512, 384]);  convert_element_type_1258 = None
        addmm_134 = torch.ops.aten.addmm.default(convert_element_type_245, view_268, permute_61);  convert_element_type_245 = None
        view_269 = torch.ops.aten.view.default(addmm_134, [32, 4, 4, 1536])
        convert_element_type_1262 = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        mul_505 = torch.ops.aten.mul.Tensor(convert_element_type_1262, 0.5)
        mul_506 = torch.ops.aten.mul.Tensor(convert_element_type_1262, 0.7071067811865476);  convert_element_type_1262 = None
        erf_67 = torch.ops.aten.erf.default(mul_506);  mul_506 = None
        add_312 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_507 = torch.ops.aten.mul.Tensor(mul_505, add_312);  mul_505 = add_312 = None
        convert_element_type_1263 = torch.ops.prims.convert_element_type.default(mul_507, torch.bfloat16);  mul_507 = None
        view_270 = torch.ops.aten.view.default(convert_element_type_1263, [512, 1536]);  convert_element_type_1263 = None
        addmm_135 = torch.ops.aten.addmm.default(convert_element_type_253, view_270, permute_62);  convert_element_type_253 = None
        view_271 = torch.ops.aten.view.default(addmm_135, [32, 4, 4, 384])
        permute_312 = torch.ops.aten.permute.default(view_271, [0, 3, 1, 2]);  view_271 = None
        mul_508 = torch.ops.aten.mul.Tensor(primals_20, permute_312);  permute_312 = None
        inductor_lookup_seed_default_67 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_8 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        lt_63 = torch.ops.aten.lt.Scalar(inductor_random_default_8, 0.9235294117647059);  inductor_random_default_8 = None
        convert_element_type_1269 = torch.ops.prims.convert_element_type.default(lt_63, torch.float32)
        div_75 = torch.ops.aten.div.Tensor(convert_element_type_1269, 0.9235294117647059);  convert_element_type_1269 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_508, div_75);  mul_508 = div_75 = None
        add_313 = torch.ops.aten.add.Tensor(mul_509, add_309);  mul_509 = add_309 = None
        convert_element_type_1272 = torch.ops.prims.convert_element_type.default(add_313, torch.bfloat16)
        convolution_87 = torch.ops.aten.convolution.default(convert_element_type_1272, convert_element_type_260, convert_element_type_259, [1, 1], [3, 3], [1, 1], False, [0, 0], 384);  convert_element_type_259 = None
        permute_313 = torch.ops.aten.permute.default(convolution_87, [0, 2, 3, 1]);  convolution_87 = None
        convert_element_type_1273 = torch.ops.prims.convert_element_type.default(permute_313, torch.float32)
        var_mean_83 = torch.ops.aten.var_mean.correction(convert_element_type_1273, [3], correction = 0, keepdim = True)
        getitem_166 = var_mean_83[0]
        getitem_167 = var_mean_83[1];  var_mean_83 = None
        add_314 = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_91 = torch.ops.aten.sub.Tensor(convert_element_type_1273, getitem_167);  convert_element_type_1273 = None
        mul_510 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_83);  sub_91 = None
        mul_511 = torch.ops.aten.mul.Tensor(mul_510, primals_147);  mul_510 = None
        add_315 = torch.ops.aten.add.Tensor(mul_511, primals_148);  mul_511 = primals_148 = None
        convert_element_type_1276 = torch.ops.prims.convert_element_type.default(add_315, torch.bfloat16);  add_315 = None
        view_272 = torch.ops.aten.view.default(convert_element_type_1276, [512, 384]);  convert_element_type_1276 = None
        addmm_136 = torch.ops.aten.addmm.default(convert_element_type_263, view_272, permute_65);  convert_element_type_263 = None
        view_273 = torch.ops.aten.view.default(addmm_136, [32, 4, 4, 1536])
        convert_element_type_1280 = torch.ops.prims.convert_element_type.default(view_273, torch.float32);  view_273 = None
        mul_512 = torch.ops.aten.mul.Tensor(convert_element_type_1280, 0.5)
        mul_513 = torch.ops.aten.mul.Tensor(convert_element_type_1280, 0.7071067811865476);  convert_element_type_1280 = None
        erf_68 = torch.ops.aten.erf.default(mul_513);  mul_513 = None
        add_316 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_514 = torch.ops.aten.mul.Tensor(mul_512, add_316);  mul_512 = add_316 = None
        convert_element_type_1281 = torch.ops.prims.convert_element_type.default(mul_514, torch.bfloat16);  mul_514 = None
        view_274 = torch.ops.aten.view.default(convert_element_type_1281, [512, 1536]);  convert_element_type_1281 = None
        addmm_137 = torch.ops.aten.addmm.default(convert_element_type_271, view_274, permute_66);  convert_element_type_271 = None
        view_275 = torch.ops.aten.view.default(addmm_137, [32, 4, 4, 384])
        permute_316 = torch.ops.aten.permute.default(view_275, [0, 3, 1, 2]);  view_275 = None
        mul_515 = torch.ops.aten.mul.Tensor(primals_21, permute_316);  permute_316 = None
        inductor_lookup_seed_default_68 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_7 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        lt_64 = torch.ops.aten.lt.Scalar(inductor_random_default_7, 0.9176470588235294);  inductor_random_default_7 = None
        convert_element_type_1287 = torch.ops.prims.convert_element_type.default(lt_64, torch.float32)
        div_76 = torch.ops.aten.div.Tensor(convert_element_type_1287, 0.9176470588235294);  convert_element_type_1287 = None
        mul_516 = torch.ops.aten.mul.Tensor(mul_515, div_76);  mul_515 = div_76 = None
        add_317 = torch.ops.aten.add.Tensor(mul_516, add_313);  mul_516 = add_313 = None
        permute_318 = torch.ops.aten.permute.default(add_317, [0, 2, 3, 1])
        var_mean_84 = torch.ops.aten.var_mean.correction(permute_318, [3], correction = 0, keepdim = True)
        getitem_168 = var_mean_84[0]
        getitem_169 = var_mean_84[1];  var_mean_84 = None
        add_318 = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_318);  add_318 = None
        sub_92 = torch.ops.aten.sub.Tensor(permute_318, getitem_169);  permute_318 = None
        mul_517 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_84);  sub_92 = None
        mul_518 = torch.ops.aten.mul.Tensor(mul_517, primals_22);  mul_517 = None
        add_319 = torch.ops.aten.add.Tensor(mul_518, primals_23);  mul_518 = primals_23 = None
        permute_319 = torch.ops.aten.permute.default(add_319, [0, 3, 1, 2]);  add_319 = None
        convert_element_type_1290 = torch.ops.prims.convert_element_type.default(permute_319, torch.bfloat16);  permute_319 = None
        convolution_88 = torch.ops.aten.convolution.default(convert_element_type_1290, convert_element_type_278, convert_element_type_277, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_277 = None
        convolution_89 = torch.ops.aten.convolution.default(convolution_88, convert_element_type_281, convert_element_type_280, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  convert_element_type_280 = None
        permute_320 = torch.ops.aten.permute.default(convolution_89, [0, 2, 3, 1]);  convolution_89 = None
        convert_element_type_1293 = torch.ops.prims.convert_element_type.default(permute_320, torch.float32)
        var_mean_85 = torch.ops.aten.var_mean.correction(convert_element_type_1293, [3], correction = 0, keepdim = True)
        getitem_170 = var_mean_85[0]
        getitem_171 = var_mean_85[1];  var_mean_85 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_93 = torch.ops.aten.sub.Tensor(convert_element_type_1293, getitem_171);  convert_element_type_1293 = None
        mul_519 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_85);  sub_93 = None
        mul_520 = torch.ops.aten.mul.Tensor(mul_519, primals_157);  mul_519 = None
        add_321 = torch.ops.aten.add.Tensor(mul_520, primals_158);  mul_520 = primals_158 = None
        convert_element_type_1296 = torch.ops.prims.convert_element_type.default(add_321, torch.bfloat16);  add_321 = None
        view_276 = torch.ops.aten.view.default(convert_element_type_1296, [128, 768]);  convert_element_type_1296 = None
        addmm_138 = torch.ops.aten.addmm.default(convert_element_type_283, view_276, permute_72);  convert_element_type_283 = None
        view_277 = torch.ops.aten.view.default(addmm_138, [32, 2, 2, 3072])
        convert_element_type_1300 = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        mul_521 = torch.ops.aten.mul.Tensor(convert_element_type_1300, 0.5)
        mul_522 = torch.ops.aten.mul.Tensor(convert_element_type_1300, 0.7071067811865476);  convert_element_type_1300 = None
        erf_69 = torch.ops.aten.erf.default(mul_522);  mul_522 = None
        add_322 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_523 = torch.ops.aten.mul.Tensor(mul_521, add_322);  mul_521 = add_322 = None
        convert_element_type_1301 = torch.ops.prims.convert_element_type.default(mul_523, torch.bfloat16);  mul_523 = None
        view_278 = torch.ops.aten.view.default(convert_element_type_1301, [128, 3072]);  convert_element_type_1301 = None
        addmm_139 = torch.ops.aten.addmm.default(convert_element_type_291, view_278, permute_73);  convert_element_type_291 = None
        view_279 = torch.ops.aten.view.default(addmm_139, [32, 2, 2, 768])
        permute_323 = torch.ops.aten.permute.default(view_279, [0, 3, 1, 2]);  view_279 = None
        mul_524 = torch.ops.aten.mul.Tensor(primals_24, permute_323);  permute_323 = None
        inductor_lookup_seed_default_69 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_6 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        lt_65 = torch.ops.aten.lt.Scalar(inductor_random_default_6, 0.9117647058823529);  inductor_random_default_6 = None
        convert_element_type_1307 = torch.ops.prims.convert_element_type.default(lt_65, torch.float32)
        div_77 = torch.ops.aten.div.Tensor(convert_element_type_1307, 0.9117647058823529);  convert_element_type_1307 = None
        mul_525 = torch.ops.aten.mul.Tensor(mul_524, div_77);  mul_524 = div_77 = None
        add_323 = torch.ops.aten.add.Tensor(mul_525, convolution_88);  mul_525 = None
        convert_element_type_1310 = torch.ops.prims.convert_element_type.default(add_323, torch.bfloat16)
        convolution_90 = torch.ops.aten.convolution.default(convert_element_type_1310, convert_element_type_298, convert_element_type_297, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  convert_element_type_297 = None
        permute_324 = torch.ops.aten.permute.default(convolution_90, [0, 2, 3, 1]);  convolution_90 = None
        convert_element_type_1311 = torch.ops.prims.convert_element_type.default(permute_324, torch.float32)
        var_mean_86 = torch.ops.aten.var_mean.correction(convert_element_type_1311, [3], correction = 0, keepdim = True)
        getitem_172 = var_mean_86[0]
        getitem_173 = var_mean_86[1];  var_mean_86 = None
        add_324 = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_94 = torch.ops.aten.sub.Tensor(convert_element_type_1311, getitem_173);  convert_element_type_1311 = None
        mul_526 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_86);  sub_94 = None
        mul_527 = torch.ops.aten.mul.Tensor(mul_526, primals_165);  mul_526 = None
        add_325 = torch.ops.aten.add.Tensor(mul_527, primals_166);  mul_527 = primals_166 = None
        convert_element_type_1314 = torch.ops.prims.convert_element_type.default(add_325, torch.bfloat16);  add_325 = None
        view_280 = torch.ops.aten.view.default(convert_element_type_1314, [128, 768]);  convert_element_type_1314 = None
        addmm_140 = torch.ops.aten.addmm.default(convert_element_type_301, view_280, permute_76);  convert_element_type_301 = None
        view_281 = torch.ops.aten.view.default(addmm_140, [32, 2, 2, 3072])
        convert_element_type_1318 = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None
        mul_528 = torch.ops.aten.mul.Tensor(convert_element_type_1318, 0.5)
        mul_529 = torch.ops.aten.mul.Tensor(convert_element_type_1318, 0.7071067811865476);  convert_element_type_1318 = None
        erf_70 = torch.ops.aten.erf.default(mul_529);  mul_529 = None
        add_326 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_530 = torch.ops.aten.mul.Tensor(mul_528, add_326);  mul_528 = add_326 = None
        convert_element_type_1319 = torch.ops.prims.convert_element_type.default(mul_530, torch.bfloat16);  mul_530 = None
        view_282 = torch.ops.aten.view.default(convert_element_type_1319, [128, 3072]);  convert_element_type_1319 = None
        addmm_141 = torch.ops.aten.addmm.default(convert_element_type_309, view_282, permute_77);  convert_element_type_309 = None
        view_283 = torch.ops.aten.view.default(addmm_141, [32, 2, 2, 768])
        permute_327 = torch.ops.aten.permute.default(view_283, [0, 3, 1, 2]);  view_283 = None
        mul_531 = torch.ops.aten.mul.Tensor(primals_25, permute_327);  permute_327 = None
        inductor_lookup_seed_default_70 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_5 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        lt_66 = torch.ops.aten.lt.Scalar(inductor_random_default_5, 0.9058823529411765);  inductor_random_default_5 = None
        convert_element_type_1325 = torch.ops.prims.convert_element_type.default(lt_66, torch.float32)
        div_78 = torch.ops.aten.div.Tensor(convert_element_type_1325, 0.9058823529411765);  convert_element_type_1325 = None
        mul_532 = torch.ops.aten.mul.Tensor(mul_531, div_78);  mul_531 = div_78 = None
        add_327 = torch.ops.aten.add.Tensor(mul_532, add_323);  mul_532 = add_323 = None
        convert_element_type_1328 = torch.ops.prims.convert_element_type.default(add_327, torch.bfloat16)
        convolution_91 = torch.ops.aten.convolution.default(convert_element_type_1328, convert_element_type_316, convert_element_type_315, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  convert_element_type_315 = None
        permute_328 = torch.ops.aten.permute.default(convolution_91, [0, 2, 3, 1]);  convolution_91 = None
        convert_element_type_1329 = torch.ops.prims.convert_element_type.default(permute_328, torch.float32)
        var_mean_87 = torch.ops.aten.var_mean.correction(convert_element_type_1329, [3], correction = 0, keepdim = True)
        getitem_174 = var_mean_87[0]
        getitem_175 = var_mean_87[1];  var_mean_87 = None
        add_328 = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
        sub_95 = torch.ops.aten.sub.Tensor(convert_element_type_1329, getitem_175);  convert_element_type_1329 = None
        mul_533 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_87);  sub_95 = None
        mul_534 = torch.ops.aten.mul.Tensor(mul_533, primals_173);  mul_533 = None
        add_329 = torch.ops.aten.add.Tensor(mul_534, primals_174);  mul_534 = primals_174 = None
        convert_element_type_1332 = torch.ops.prims.convert_element_type.default(add_329, torch.bfloat16);  add_329 = None
        view_284 = torch.ops.aten.view.default(convert_element_type_1332, [128, 768]);  convert_element_type_1332 = None
        addmm_142 = torch.ops.aten.addmm.default(convert_element_type_319, view_284, permute_80);  convert_element_type_319 = None
        view_285 = torch.ops.aten.view.default(addmm_142, [32, 2, 2, 3072])
        convert_element_type_1336 = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_535 = torch.ops.aten.mul.Tensor(convert_element_type_1336, 0.5)
        mul_536 = torch.ops.aten.mul.Tensor(convert_element_type_1336, 0.7071067811865476);  convert_element_type_1336 = None
        erf_71 = torch.ops.aten.erf.default(mul_536);  mul_536 = None
        add_330 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_537 = torch.ops.aten.mul.Tensor(mul_535, add_330);  mul_535 = add_330 = None
        convert_element_type_1337 = torch.ops.prims.convert_element_type.default(mul_537, torch.bfloat16);  mul_537 = None
        view_286 = torch.ops.aten.view.default(convert_element_type_1337, [128, 3072]);  convert_element_type_1337 = None
        addmm_143 = torch.ops.aten.addmm.default(convert_element_type_327, view_286, permute_81);  convert_element_type_327 = None
        view_287 = torch.ops.aten.view.default(addmm_143, [32, 2, 2, 768])
        permute_331 = torch.ops.aten.permute.default(view_287, [0, 3, 1, 2]);  view_287 = None
        mul_538 = torch.ops.aten.mul.Tensor(primals_26, permute_331);  permute_331 = None
        inductor_lookup_seed_default_71 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([32, 1, 1, 1], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        lt_67 = torch.ops.aten.lt.Scalar(inductor_random_default_4, 0.9);  inductor_random_default_4 = None
        convert_element_type_1343 = torch.ops.prims.convert_element_type.default(lt_67, torch.float32)
        div_79 = torch.ops.aten.div.Tensor(convert_element_type_1343, 0.9);  convert_element_type_1343 = None
        mul_539 = torch.ops.aten.mul.Tensor(mul_538, div_79);  mul_538 = div_79 = None
        add_331 = torch.ops.aten.add.Tensor(mul_539, add_327);  mul_539 = add_327 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(add_185, 2)
        sum_9 = torch.ops.aten.sum.dim_IntList(pow_13, [1], True, dtype = torch.float32);  pow_13 = None
        sqrt_8 = torch.ops.aten.sqrt.default(sum_9);  sum_9 = None
        add_332 = torch.ops.aten.add.Tensor(sqrt_8, 1e-10)
        div_80 = torch.ops.aten.div.Tensor(add_185, add_332);  add_332 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(add_265, 2)
        sum_10 = torch.ops.aten.sum.dim_IntList(pow_14, [1], True, dtype = torch.float32);  pow_14 = None
        sqrt_9 = torch.ops.aten.sqrt.default(sum_10);  sum_10 = None
        add_333 = torch.ops.aten.add.Tensor(sqrt_9, 1e-10)
        div_81 = torch.ops.aten.div.Tensor(add_265, add_333);  add_333 = None
        sub_96 = torch.ops.aten.sub.Tensor(div_80, div_81);  div_80 = div_81 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(sub_96, 2);  sub_96 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_199, 2)
        sum_11 = torch.ops.aten.sum.dim_IntList(pow_16, [1], True, dtype = torch.float32);  pow_16 = None
        sqrt_10 = torch.ops.aten.sqrt.default(sum_11);  sum_11 = None
        add_334 = torch.ops.aten.add.Tensor(sqrt_10, 1e-10)
        div_82 = torch.ops.aten.div.Tensor(add_199, add_334);  add_334 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(add_279, 2)
        sum_12 = torch.ops.aten.sum.dim_IntList(pow_17, [1], True, dtype = torch.float32);  pow_17 = None
        sqrt_11 = torch.ops.aten.sqrt.default(sum_12);  sum_12 = None
        add_335 = torch.ops.aten.add.Tensor(sqrt_11, 1e-10)
        div_83 = torch.ops.aten.div.Tensor(add_279, add_335);  add_335 = None
        sub_97 = torch.ops.aten.sub.Tensor(div_82, div_83);  div_82 = div_83 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(sub_97, 2);  sub_97 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(add_237, 2)
        sum_13 = torch.ops.aten.sum.dim_IntList(pow_19, [1], True, dtype = torch.float32);  pow_19 = None
        sqrt_12 = torch.ops.aten.sqrt.default(sum_13);  sum_13 = None
        add_336 = torch.ops.aten.add.Tensor(sqrt_12, 1e-10)
        div_84 = torch.ops.aten.div.Tensor(add_237, add_336);  add_336 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_317, 2)
        sum_14 = torch.ops.aten.sum.dim_IntList(pow_20, [1], True, dtype = torch.float32);  pow_20 = None
        sqrt_13 = torch.ops.aten.sqrt.default(sum_14);  sum_14 = None
        add_337 = torch.ops.aten.add.Tensor(sqrt_13, 1e-10)
        div_85 = torch.ops.aten.div.Tensor(add_317, add_337);  add_337 = None
        sub_98 = torch.ops.aten.sub.Tensor(div_84, div_85);  div_84 = div_85 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(sub_98, 2);  sub_98 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_251, 2)
        sum_15 = torch.ops.aten.sum.dim_IntList(pow_22, [1], True, dtype = torch.float32);  pow_22 = None
        sqrt_14 = torch.ops.aten.sqrt.default(sum_15);  sum_15 = None
        add_338 = torch.ops.aten.add.Tensor(sqrt_14, 1e-10)
        div_86 = torch.ops.aten.div.Tensor(add_251, add_338);  add_338 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(add_331, 2)
        sum_16 = torch.ops.aten.sum.dim_IntList(pow_23, [1], True, dtype = torch.float32);  pow_23 = None
        sqrt_15 = torch.ops.aten.sqrt.default(sum_16);  sum_16 = None
        add_339 = torch.ops.aten.add.Tensor(sqrt_15, 1e-10)
        div_87 = torch.ops.aten.div.Tensor(add_331, add_339);  add_339 = None
        sub_99 = torch.ops.aten.sub.Tensor(div_86, div_87);  div_86 = div_87 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(sub_99, 2);  sub_99 = None
        inductor_lookup_seed_default_72 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([32, 96, 16, 16], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        convert_element_type_1344 = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.float32);  inductor_random_default_3 = None
        clone_4 = torch.ops.aten.clone.default(convert_element_type_1344, memory_format = torch.channels_last);  convert_element_type_1344 = None
        gt_4 = torch.ops.aten.gt.Scalar(clone_4, 0.5);  clone_4 = None
        mul_540 = torch.ops.aten.mul.Tensor(gt_4, pow_15);  pow_15 = None
        mul_541 = torch.ops.aten.mul.Tensor(mul_540, 2.0);  mul_540 = None
        convert_element_type_1346 = torch.ops.prims.convert_element_type.default(mul_541, torch.bfloat16);  mul_541 = None
        convolution_92 = torch.ops.aten.convolution.default(convert_element_type_1346, convert_element_type_667, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_4 = torch.ops.aten.mean.dim(convolution_92, [2, 3], True);  convolution_92 = None
        inductor_lookup_seed_default_73 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 73)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([32, 192, 8, 8], inductor_lookup_seed_default_73, 'rand');  inductor_lookup_seed_default_73 = None
        convert_element_type_1347 = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float32);  inductor_random_default_2 = None
        clone_5 = torch.ops.aten.clone.default(convert_element_type_1347, memory_format = torch.channels_last);  convert_element_type_1347 = None
        gt_5 = torch.ops.aten.gt.Scalar(clone_5, 0.5);  clone_5 = None
        mul_542 = torch.ops.aten.mul.Tensor(gt_5, pow_18);  pow_18 = None
        mul_543 = torch.ops.aten.mul.Tensor(mul_542, 2.0);  mul_542 = None
        convert_element_type_1349 = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None
        convolution_93 = torch.ops.aten.convolution.default(convert_element_type_1349, convert_element_type_670, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_5 = torch.ops.aten.mean.dim(convolution_93, [2, 3], True);  convolution_93 = None
        inductor_lookup_seed_default_74 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 74)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([32, 384, 4, 4], inductor_lookup_seed_default_74, 'rand');  inductor_lookup_seed_default_74 = None
        convert_element_type_1350 = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float32);  inductor_random_default_1 = None
        clone_6 = torch.ops.aten.clone.default(convert_element_type_1350, memory_format = torch.channels_last);  convert_element_type_1350 = None
        gt_6 = torch.ops.aten.gt.Scalar(clone_6, 0.5);  clone_6 = None
        mul_544 = torch.ops.aten.mul.Tensor(gt_6, pow_21);  pow_21 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_544, 2.0);  mul_544 = None
        convert_element_type_1352 = torch.ops.prims.convert_element_type.default(mul_545, torch.bfloat16);  mul_545 = None
        convolution_94 = torch.ops.aten.convolution.default(convert_element_type_1352, convert_element_type_673, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_6 = torch.ops.aten.mean.dim(convolution_94, [2, 3], True);  convolution_94 = None
        inductor_lookup_seed_default_75 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 75);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([32, 768, 2, 2], inductor_lookup_seed_default_75, 'rand');  inductor_lookup_seed_default_75 = None
        convert_element_type_1353 = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float32);  inductor_random_default = None
        clone_7 = torch.ops.aten.clone.default(convert_element_type_1353, memory_format = torch.channels_last);  convert_element_type_1353 = None
        gt_7 = torch.ops.aten.gt.Scalar(clone_7, 0.5);  clone_7 = None
        mul_546 = torch.ops.aten.mul.Tensor(gt_7, pow_24);  pow_24 = None
        mul_547 = torch.ops.aten.mul.Tensor(mul_546, 2.0);  mul_546 = None
        convert_element_type_1355 = torch.ops.prims.convert_element_type.default(mul_547, torch.bfloat16);  mul_547 = None
        convolution_95 = torch.ops.aten.convolution.default(convert_element_type_1355, convert_element_type_676, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_7 = torch.ops.aten.mean.dim(convolution_95, [2, 3], True);  convolution_95 = None
        add_340 = torch.ops.aten.add.Tensor(mean_4, 0);  mean_4 = None
        add_341 = torch.ops.aten.add.Tensor(add_340, mean_5);  add_340 = mean_5 = None
        add_342 = torch.ops.aten.add.Tensor(add_341, mean_6);  add_341 = mean_6 = None
        add_343 = torch.ops.aten.add.Tensor(add_342, mean_7);  add_342 = mean_7 = None
        mul_548 = torch.ops.aten.mul.Tensor(primals_194, 2.0)
        sub_100 = torch.ops.aten.sub.Tensor(mul_548, 1.0);  mul_548 = None
        add_344 = torch.ops.aten.add.Tensor(sub_100, 1.0);  sub_100 = None
        div_88 = torch.ops.aten.div.Tensor(add_344, 2.0);  add_344 = None
        sub_101 = torch.ops.aten.sub.Tensor(add_171, add_343)
        add_345 = torch.ops.aten.add.Tensor(add_343, 0.1)
        div_89 = torch.ops.aten.div.Tensor(add_171, add_345);  add_345 = None
        add_346 = torch.ops.aten.add.Tensor(add_171, 0.1)
        div_90 = torch.ops.aten.div.Tensor(add_343, add_346);  add_346 = None
        cat = torch.ops.aten.cat.default([add_171, add_343, sub_101, div_89, div_90], 1);  sub_101 = div_89 = div_90 = None
        convert_element_type_1356 = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convert_element_type_1357 = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convolution_96 = torch.ops.aten.convolution.default(cat, convert_element_type_1357, convert_element_type_1356, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_1356 = None
        convert_element_type_1358 = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32);  convolution_96 = None
        gt_8 = torch.ops.aten.gt.Scalar(convert_element_type_1358, 0)
        mul_549 = torch.ops.aten.mul.Tensor(convert_element_type_1358, 0.2)
        where = torch.ops.aten.where.self(gt_8, convert_element_type_1358, mul_549);  gt_8 = convert_element_type_1358 = mul_549 = None
        convert_element_type_1359 = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        convert_element_type_1360 = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        convert_element_type_1361 = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convolution_97 = torch.ops.aten.convolution.default(convert_element_type_1359, convert_element_type_1361, convert_element_type_1360, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_1360 = None
        convert_element_type_1362 = torch.ops.prims.convert_element_type.default(convolution_97, torch.float32);  convolution_97 = None
        gt_9 = torch.ops.aten.gt.Scalar(convert_element_type_1362, 0)
        mul_550 = torch.ops.aten.mul.Tensor(convert_element_type_1362, 0.2)
        where_1 = torch.ops.aten.where.self(gt_9, convert_element_type_1362, mul_550);  gt_9 = convert_element_type_1362 = mul_550 = None
        convert_element_type_1363 = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convert_element_type_1364 = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_1365 = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        convolution_98 = torch.ops.aten.convolution.default(convert_element_type_1363, convert_element_type_1365, convert_element_type_1364, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_1364 = None
        convert_element_type_1366 = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32)
        neg = torch.ops.aten.neg.default(convert_element_type_1366)
        clamp_min = torch.ops.aten.clamp_min.default(neg, 0)
        sub_102 = torch.ops.aten.sub.Tensor(1, div_88)
        mul_551 = torch.ops.aten.mul.Tensor(sub_102, convert_element_type_1366);  sub_102 = None
        add_347 = torch.ops.aten.add.Tensor(mul_551, clamp_min);  mul_551 = None
        neg_1 = torch.ops.aten.neg.default(clamp_min)
        exp = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        sub_103 = torch.ops.aten.sub.Tensor(neg, clamp_min);  neg = clamp_min = None
        exp_1 = torch.ops.aten.exp.default(sub_103);  sub_103 = None
        add_348 = torch.ops.aten.add.Tensor(exp, exp_1);  exp = exp_1 = None
        log = torch.ops.aten.log.default(add_348);  add_348 = None
        add_349 = torch.ops.aten.add.Tensor(add_347, log);  add_347 = log = None
        mean_8 = torch.ops.aten.mean.default(add_349);  add_349 = None
        lt_68 = torch.ops.aten.lt.Tensor(add_343, add_171)
        alias_18 = torch.ops.aten.alias.default(lt_68);  lt_68 = None
        alias_19 = torch.ops.aten.alias.default(alias_18);  alias_18 = None
        view_288 = torch.ops.aten.view.default(alias_19, [32]);  alias_19 = None
        view_289 = torch.ops.aten.view.default(primals_194, [32]);  primals_194 = None
        mul_552 = torch.ops.aten.mul.Tensor(view_288, view_289)
        bitwise_not = torch.ops.aten.bitwise_not.default(view_288);  view_288 = None
        sub_104 = torch.ops.aten.sub.Tensor(1, view_289);  view_289 = None
        mul_553 = torch.ops.aten.mul.Tensor(bitwise_not, sub_104);  bitwise_not = sub_104 = None
        add_350 = torch.ops.aten.add.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_1366);  convert_element_type_1366 = None
        sub_105 = torch.ops.aten.sub.Tensor(sigmoid, div_88);  sigmoid = div_88 = None
        permute_333 = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_337 = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        permute_343 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_347 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_353 = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        permute_357 = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        permute_365 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_369 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_375 = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        permute_379 = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        permute_385 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_389 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_395 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_399 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_405 = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        permute_409 = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_415 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_419 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_425 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_429 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_435 = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_439 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_445 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_449 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_457 = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_461 = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_467 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_471 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_477 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_481 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        permute_489 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_493 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_499 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_503 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_509 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_513 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        return [mean_8, add_350, convolution_98, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_31, primals_39, primals_47, primals_57, primals_65, primals_73, primals_83, primals_91, primals_99, primals_107, primals_115, primals_123, primals_131, primals_139, primals_147, primals_157, primals_165, primals_173, convert_element_type_1, convert_element_type_2, permute, getitem_1, rsqrt, convert_element_type_5, convert_element_type_6, permute_2, getitem_3, rsqrt_1, view, addmm, view_2, addmm_1, convert_element_type_22, convert_element_type_23, permute_6, getitem_5, rsqrt_2, view_4, addmm_2, view_6, addmm_3, lt, convert_element_type_40, convert_element_type_41, permute_10, getitem_7, rsqrt_3, view_8, addmm_4, view_10, addmm_5, lt_1, add_13, getitem_9, rsqrt_4, convert_element_type_58, convert_element_type_59, convolution_4, convert_element_type_61, permute_17, getitem_11, rsqrt_5, view_12, addmm_6, view_14, addmm_7, lt_2, convert_element_type_78, convert_element_type_79, permute_21, getitem_13, rsqrt_6, view_16, addmm_8, view_18, addmm_9, lt_3, convert_element_type_96, convert_element_type_97, permute_25, getitem_15, rsqrt_7, view_20, addmm_10, view_22, addmm_11, lt_4, add_27, getitem_17, rsqrt_8, convert_element_type_114, convert_element_type_115, convolution_8, convert_element_type_117, permute_32, getitem_19, rsqrt_9, view_24, addmm_12, view_26, addmm_13, lt_5, convert_element_type_134, convert_element_type_135, permute_36, getitem_21, rsqrt_10, view_28, addmm_14, view_30, addmm_15, lt_6, convert_element_type_152, convert_element_type_153, permute_40, getitem_23, rsqrt_11, view_32, addmm_16, view_34, addmm_17, lt_7, convert_element_type_170, convert_element_type_171, permute_44, getitem_25, rsqrt_12, view_36, addmm_18, view_38, addmm_19, lt_8, convert_element_type_188, convert_element_type_189, permute_48, getitem_27, rsqrt_13, view_40, addmm_20, view_42, addmm_21, lt_9, convert_element_type_206, convert_element_type_207, permute_52, getitem_29, rsqrt_14, view_44, addmm_22, view_46, addmm_23, lt_10, convert_element_type_224, convert_element_type_225, permute_56, getitem_31, rsqrt_15, view_48, addmm_24, view_50, addmm_25, lt_11, convert_element_type_242, convert_element_type_243, permute_60, getitem_33, rsqrt_16, view_52, addmm_26, view_54, addmm_27, lt_12, convert_element_type_260, convert_element_type_261, permute_64, getitem_35, rsqrt_17, view_56, addmm_28, view_58, addmm_29, lt_13, add_65, getitem_37, rsqrt_18, convert_element_type_278, convert_element_type_279, convolution_18, convert_element_type_281, permute_71, getitem_39, rsqrt_19, view_60, addmm_30, view_62, addmm_31, lt_14, convert_element_type_298, convert_element_type_299, permute_75, getitem_41, rsqrt_20, view_64, addmm_32, view_66, addmm_33, lt_15, convert_element_type_316, convert_element_type_317, permute_79, getitem_43, rsqrt_21, view_68, addmm_34, view_70, addmm_35, lt_16, add_79, convert_element_type_335, permute_83, getitem_45, rsqrt_22, convert_element_type_339, permute_85, getitem_47, rsqrt_23, view_72, addmm_36, view_74, addmm_37, convert_element_type_356, permute_89, getitem_49, rsqrt_24, view_76, addmm_38, view_78, addmm_39, lt_17, convert_element_type_374, permute_93, getitem_51, rsqrt_25, view_80, addmm_40, view_82, addmm_41, lt_18, add_93, getitem_53, rsqrt_26, convert_element_type_392, convolution_26, permute_100, getitem_55, rsqrt_27, view_84, addmm_42, view_86, addmm_43, lt_19, convert_element_type_412, permute_104, getitem_57, rsqrt_28, view_88, addmm_44, view_90, addmm_45, lt_20, convert_element_type_430, permute_108, getitem_59, rsqrt_29, view_92, addmm_46, view_94, addmm_47, lt_21, add_107, getitem_61, rsqrt_30, convert_element_type_448, convolution_30, permute_115, getitem_63, rsqrt_31, view_96, addmm_48, view_98, addmm_49, lt_22, convert_element_type_468, permute_119, getitem_65, rsqrt_32, view_100, addmm_50, view_102, addmm_51, lt_23, convert_element_type_486, permute_123, getitem_67, rsqrt_33, view_104, addmm_52, view_106, addmm_53, lt_24, convert_element_type_504, permute_127, getitem_69, rsqrt_34, view_108, addmm_54, view_110, addmm_55, lt_25, convert_element_type_522, permute_131, getitem_71, rsqrt_35, view_112, addmm_56, view_114, addmm_57, lt_26, convert_element_type_540, permute_135, getitem_73, rsqrt_36, view_116, addmm_58, view_118, addmm_59, lt_27, convert_element_type_558, permute_139, getitem_75, rsqrt_37, view_120, addmm_60, view_122, addmm_61, lt_28, convert_element_type_576, permute_143, getitem_77, rsqrt_38, view_124, addmm_62, view_126, addmm_63, lt_29, convert_element_type_594, permute_147, getitem_79, rsqrt_39, view_128, addmm_64, view_130, addmm_65, lt_30, add_145, getitem_81, rsqrt_40, convert_element_type_612, convolution_40, permute_154, getitem_83, rsqrt_41, view_132, addmm_66, view_134, addmm_67, lt_31, convert_element_type_632, permute_158, getitem_85, rsqrt_42, view_136, addmm_68, view_138, addmm_69, lt_32, convert_element_type_650, permute_162, getitem_87, rsqrt_43, view_140, addmm_70, view_142, addmm_71, lt_33, add_159, sqrt, sqrt_1, sqrt_2, sqrt_3, sqrt_4, sqrt_5, sqrt_6, sqrt_7, gt, convert_element_type_667, convert_element_type_668, gt_1, convert_element_type_670, convert_element_type_671, gt_2, convert_element_type_673, convert_element_type_674, gt_3, convert_element_type_676, convert_element_type_677, add_171, lt_34, convert_element_type_719, permute_176, getitem_95, rsqrt_47, view_152, addmm_76, view_154, addmm_77, lt_35, add_185, getitem_97, rsqrt_48, convert_element_type_737, convolution_52, permute_183, getitem_99, rsqrt_49, view_156, addmm_78, view_158, addmm_79, lt_36, convert_element_type_757, permute_187, getitem_101, rsqrt_50, view_160, addmm_80, view_162, addmm_81, lt_37, convert_element_type_775, permute_191, getitem_103, rsqrt_51, view_164, addmm_82, view_166, addmm_83, lt_38, add_199, getitem_105, rsqrt_52, convert_element_type_793, convolution_56, permute_198, getitem_107, rsqrt_53, view_168, addmm_84, view_170, addmm_85, lt_39, convert_element_type_813, permute_202, getitem_109, rsqrt_54, view_172, addmm_86, view_174, addmm_87, lt_40, convert_element_type_831, permute_206, getitem_111, rsqrt_55, view_176, addmm_88, view_178, addmm_89, lt_41, convert_element_type_849, permute_210, getitem_113, rsqrt_56, view_180, addmm_90, view_182, addmm_91, lt_42, convert_element_type_867, permute_214, getitem_115, rsqrt_57, view_184, addmm_92, view_186, addmm_93, lt_43, convert_element_type_885, permute_218, getitem_117, rsqrt_58, view_188, addmm_94, view_190, addmm_95, lt_44, convert_element_type_903, permute_222, getitem_119, rsqrt_59, view_192, addmm_96, view_194, addmm_97, lt_45, convert_element_type_921, permute_226, getitem_121, rsqrt_60, view_196, addmm_98, view_198, addmm_99, lt_46, convert_element_type_939, permute_230, getitem_123, rsqrt_61, view_200, addmm_100, view_202, addmm_101, lt_47, add_237, getitem_125, rsqrt_62, convert_element_type_957, convolution_66, permute_237, getitem_127, rsqrt_63, view_204, addmm_102, view_206, addmm_103, lt_48, convert_element_type_977, permute_241, getitem_129, rsqrt_64, view_208, addmm_104, view_210, addmm_105, lt_49, convert_element_type_995, permute_245, getitem_131, rsqrt_65, view_212, addmm_106, view_214, addmm_107, lt_50, add_251, convert_element_type_1013, permute_249, getitem_133, rsqrt_66, convert_element_type_1017, permute_251, getitem_135, rsqrt_67, view_216, addmm_108, view_218, addmm_109, convert_element_type_1034, permute_255, getitem_137, rsqrt_68, view_220, addmm_110, view_222, addmm_111, lt_51, convert_element_type_1052, permute_259, getitem_139, rsqrt_69, view_224, addmm_112, view_226, addmm_113, lt_52, add_265, getitem_141, rsqrt_70, convert_element_type_1070, convolution_74, permute_266, getitem_143, rsqrt_71, view_228, addmm_114, view_230, addmm_115, lt_53, convert_element_type_1090, permute_270, getitem_145, rsqrt_72, view_232, addmm_116, view_234, addmm_117, lt_54, convert_element_type_1108, permute_274, getitem_147, rsqrt_73, view_236, addmm_118, view_238, addmm_119, lt_55, add_279, getitem_149, rsqrt_74, convert_element_type_1126, convolution_78, permute_281, getitem_151, rsqrt_75, view_240, addmm_120, view_242, addmm_121, lt_56, convert_element_type_1146, permute_285, getitem_153, rsqrt_76, view_244, addmm_122, view_246, addmm_123, lt_57, convert_element_type_1164, permute_289, getitem_155, rsqrt_77, view_248, addmm_124, view_250, addmm_125, lt_58, convert_element_type_1182, permute_293, getitem_157, rsqrt_78, view_252, addmm_126, view_254, addmm_127, lt_59, convert_element_type_1200, permute_297, getitem_159, rsqrt_79, view_256, addmm_128, view_258, addmm_129, lt_60, convert_element_type_1218, permute_301, getitem_161, rsqrt_80, view_260, addmm_130, view_262, addmm_131, lt_61, convert_element_type_1236, permute_305, getitem_163, rsqrt_81, view_264, addmm_132, view_266, addmm_133, lt_62, convert_element_type_1254, permute_309, getitem_165, rsqrt_82, view_268, addmm_134, view_270, addmm_135, lt_63, convert_element_type_1272, permute_313, getitem_167, rsqrt_83, view_272, addmm_136, view_274, addmm_137, lt_64, add_317, getitem_169, rsqrt_84, convert_element_type_1290, convolution_88, permute_320, getitem_171, rsqrt_85, view_276, addmm_138, view_278, addmm_139, lt_65, convert_element_type_1310, permute_324, getitem_173, rsqrt_86, view_280, addmm_140, view_282, addmm_141, lt_66, convert_element_type_1328, permute_328, getitem_175, rsqrt_87, view_284, addmm_142, view_286, addmm_143, lt_67, add_331, sqrt_8, sqrt_9, sqrt_10, sqrt_11, sqrt_12, sqrt_13, sqrt_14, sqrt_15, gt_4, convert_element_type_1346, gt_5, convert_element_type_1349, gt_6, convert_element_type_1352, gt_7, convert_element_type_1355, add_343, cat, convert_element_type_1357, convert_element_type_1359, convert_element_type_1361, convert_element_type_1363, convert_element_type_1365, sub_105, permute_333, permute_337, permute_343, permute_347, permute_353, permute_357, permute_365, permute_369, permute_375, permute_379, permute_385, permute_389, permute_395, permute_399, permute_405, permute_409, permute_415, permute_419, permute_425, permute_429, permute_435, permute_439, permute_445, permute_449, permute_457, permute_461, permute_467, permute_471, permute_477, permute_481, permute_489, permute_493, permute_499, permute_503, permute_509, permute_513]
        
def load_args(reader):
    buf0 = reader.storage('2f840126d1ca3e192263f5760634d48528b006e0', 384, device=device(type='cuda', index=0))
    reader.tensor(buf0, (96,), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage('c0920fae9fd7250702c44fc1aa3910e8c666ea55', 384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (96,), requires_grad=True, is_leaf=True)  # primals_2
    buf2 = reader.storage('a13bcc891b58ac684cb136d018634baee75db26a', 384, device=device(type='cuda', index=0))
    reader.tensor(buf2, (96, 1, 1), requires_grad=True, is_leaf=True)  # primals_3
    buf3 = reader.storage('31dbc634d0f2f146e141f747bedf869598dcc1cc', 384, device=device(type='cuda', index=0))
    reader.tensor(buf3, (96, 1, 1), requires_grad=True, is_leaf=True)  # primals_4
    buf4 = reader.storage('bcb95e6aab5ef564664bf5fdbc6bcb6cacb12118', 384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (96, 1, 1), requires_grad=True, is_leaf=True)  # primals_5
    buf5 = reader.storage('2940a97002293db28f51a03dfef3123f151da1dc', 384, device=device(type='cuda', index=0))
    reader.tensor(buf5, (96,), requires_grad=True, is_leaf=True)  # primals_6
    buf6 = reader.storage('4a8b601c5f815e04d377d8ea11f145a391d8bdc1', 384, device=device(type='cuda', index=0))
    reader.tensor(buf6, (96,), requires_grad=True, is_leaf=True)  # primals_7
    buf7 = reader.storage('44677d09d928c3d94f3a19af8fa23af7132673a9', 768, device=device(type='cuda', index=0))
    reader.tensor(buf7, (192, 1, 1), requires_grad=True, is_leaf=True)  # primals_8
    buf8 = reader.storage('a76be55d07240ff55829e8ea4e0c4bfc10df99d6', 768, device=device(type='cuda', index=0))
    reader.tensor(buf8, (192, 1, 1), requires_grad=True, is_leaf=True)  # primals_9
    buf9 = reader.storage('dae791bd5a869adcd5390ccabed30b58a2808d0c', 768, device=device(type='cuda', index=0))
    reader.tensor(buf9, (192, 1, 1), requires_grad=True, is_leaf=True)  # primals_10
    buf10 = reader.storage('f5ef667f8ea471ed52d0855c42ddf502bf84f891', 768, device=device(type='cuda', index=0))
    reader.tensor(buf10, (192,), requires_grad=True, is_leaf=True)  # primals_11
    buf11 = reader.storage('5e84dedcadf48c1187f4eafa05e601de49ce8efc', 768, device=device(type='cuda', index=0))
    reader.tensor(buf11, (192,), requires_grad=True, is_leaf=True)  # primals_12
    buf12 = reader.storage('979455bbdc23e0e70d6b71c497b314fffffedf48', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf12, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_13
    buf13 = reader.storage('a6b8adfd11b1cb7214c8c762ceb61091e43ba0cf', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf13, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_14
    buf14 = reader.storage('5f18f38358950a369cc3ca4fdfe5a148e3e9ae15', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_15
    buf15 = reader.storage('7e3287e2bd3c807a49f083dd7636ee33250496df', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_16
    buf16 = reader.storage('10b7cae2a638f1b8b7e81ef7240d588432882a51', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_17
    buf17 = reader.storage('76fd8992856c700d28e9e00c93ea109014ebf904', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_18
    buf18 = reader.storage('839312adb1ab8636053ed2cca409c83cd5be59ab', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_19
    buf19 = reader.storage('eede75ab2195e323a035dec373c618b4535049e8', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_20
    buf20 = reader.storage('ea6046de0df5e024cf2b78f943078b81ee7c70b1', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf20, (384, 1, 1), requires_grad=True, is_leaf=True)  # primals_21
    buf21 = reader.storage('ca2c5e15f7703014653d3d6c0e754a98fcba8790', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (384,), requires_grad=True, is_leaf=True)  # primals_22
    buf22 = reader.storage('2f6a675c331b808d873bb8824c8e2ce85699675f', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf22, (384,), requires_grad=True, is_leaf=True)  # primals_23
    buf23 = reader.storage('3b6120624c665d6070cdd7cd4a107899a1ec340e', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768, 1, 1), requires_grad=True, is_leaf=True)  # primals_24
    buf24 = reader.storage('96ca1a284ae57c09aa699d2e6befcf2914e7ecee', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 1, 1), requires_grad=True, is_leaf=True)  # primals_25
    buf25 = reader.storage('87e09e5ba53cc7000cc4e109744581166413a581', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 1, 1), requires_grad=True, is_leaf=True)  # primals_26
    buf26 = reader.storage('6d1cb1ec1abbcb89e085365d013a52ee1ce5a9e0', 18432, device=device(type='cuda', index=0))
    reader.tensor(buf26, (96, 3, 4, 4), (48, 1, 12, 3), requires_grad=True, is_leaf=True)  # primals_27
    buf27 = reader.storage('0e615d783c9c6ae36635ec394c45a0f60111aa08', 384, device=device(type='cuda', index=0))
    reader.tensor(buf27, (96,), requires_grad=True, is_leaf=True)  # primals_28
    buf28 = reader.storage('3216ac46cf71bc604df81782f48df5fb62d3bae1', 18816, device=device(type='cuda', index=0))
    reader.tensor(buf28, (96, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_29
    buf29 = reader.storage('ebed068f5e53f280a5d79038b7d2cc02ef2aff9b', 384, device=device(type='cuda', index=0))
    reader.tensor(buf29, (96,), requires_grad=True, is_leaf=True)  # primals_30
    buf30 = reader.storage('8158e3f0091e6b2d4f82e1f9def9439155267b3d', 384, device=device(type='cuda', index=0))
    reader.tensor(buf30, (96,), requires_grad=True, is_leaf=True)  # primals_31
    buf31 = reader.storage('a4ce9fed64216328137a9b0614b14ed484cde458', 384, device=device(type='cuda', index=0))
    reader.tensor(buf31, (96,), requires_grad=True, is_leaf=True)  # primals_32
    buf32 = reader.storage('448e300ce48885a6b8d1764c12bcb584f886e882', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384, 96), requires_grad=True, is_leaf=True)  # primals_33
    buf33 = reader.storage('5d79dfbd21d4b0f831051fc9157abe7a2f7a7e0a', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (384,), requires_grad=True, is_leaf=True)  # primals_34
    buf34 = reader.storage('96be1157963a7f3af4fece7a30d46fbeea04169b', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf34, (96, 384), requires_grad=True, is_leaf=True)  # primals_35
    buf35 = reader.storage('4d445e8174e763cc08cb0a47d382c63264237502', 384, device=device(type='cuda', index=0))
    reader.tensor(buf35, (96,), requires_grad=True, is_leaf=True)  # primals_36
    buf36 = reader.storage('f35b9c7d79f7316ddf55fd823f0f240e2a5ff745', 18816, device=device(type='cuda', index=0))
    reader.tensor(buf36, (96, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_37
    buf37 = reader.storage('f653e265af8ce9a7e759d3aebbbb7b121238fae9', 384, device=device(type='cuda', index=0))
    reader.tensor(buf37, (96,), requires_grad=True, is_leaf=True)  # primals_38
    buf38 = reader.storage('ccfb33076030e91593c5e261005ac2aeb46fb400', 384, device=device(type='cuda', index=0))
    reader.tensor(buf38, (96,), requires_grad=True, is_leaf=True)  # primals_39
    buf39 = reader.storage('f9a0cb6e0a35d9721ce91b07cb6209d77ab7eaba', 384, device=device(type='cuda', index=0))
    reader.tensor(buf39, (96,), requires_grad=True, is_leaf=True)  # primals_40
    buf40 = reader.storage('b6aa5e67011430857402bdd86ac8c96f0ea689b7', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf40, (384, 96), requires_grad=True, is_leaf=True)  # primals_41
    buf41 = reader.storage('9cb89c2753d24c98e40344143ee0f812cf040773', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (384,), requires_grad=True, is_leaf=True)  # primals_42
    buf42 = reader.storage('645eb381436bc197d17310f073e8a78b8d66a155', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf42, (96, 384), requires_grad=True, is_leaf=True)  # primals_43
    buf43 = reader.storage('d65f77bb6795908b73474437424b3a42a7d89be1', 384, device=device(type='cuda', index=0))
    reader.tensor(buf43, (96,), requires_grad=True, is_leaf=True)  # primals_44
    buf44 = reader.storage('23eee3882e370cfc263a71f50a0888680afb73e1', 18816, device=device(type='cuda', index=0))
    reader.tensor(buf44, (96, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_45
    buf45 = reader.storage('ad68e15fac30f7e2aad7eadd2732b903407a6eca', 384, device=device(type='cuda', index=0))
    reader.tensor(buf45, (96,), requires_grad=True, is_leaf=True)  # primals_46
    buf46 = reader.storage('cff3be0580b040d2588655d202465d71869aa3b5', 384, device=device(type='cuda', index=0))
    reader.tensor(buf46, (96,), requires_grad=True, is_leaf=True)  # primals_47
    buf47 = reader.storage('dea83938e0d05f97409f188286bb23f7f7223618', 384, device=device(type='cuda', index=0))
    reader.tensor(buf47, (96,), requires_grad=True, is_leaf=True)  # primals_48
    buf48 = reader.storage('ae16dfc9286d97b358bb8d9a384d4be5147ad0c6', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf48, (384, 96), requires_grad=True, is_leaf=True)  # primals_49
    buf49 = reader.storage('573f59f7dd98f12de011095e023464c504e53ea4', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf49, (384,), requires_grad=True, is_leaf=True)  # primals_50
    buf50 = reader.storage('9bdf4a125d3b8585cd6911b52ed924b0d0d136d8', 147456, device=device(type='cuda', index=0))
    reader.tensor(buf50, (96, 384), requires_grad=True, is_leaf=True)  # primals_51
    buf51 = reader.storage('5523131130850effce45fce508a1d62b2cb71141', 384, device=device(type='cuda', index=0))
    reader.tensor(buf51, (96,), requires_grad=True, is_leaf=True)  # primals_52
    buf52 = reader.storage('2ee453a9acc33630eb7672d968fb1da41fa2f370', 294912, device=device(type='cuda', index=0))
    reader.tensor(buf52, (192, 96, 2, 2), (384, 1, 192, 96), requires_grad=True, is_leaf=True)  # primals_53
    buf53 = reader.storage('7058e8d61e112e773fba2ffb55902c487c7c4f4e', 768, device=device(type='cuda', index=0))
    reader.tensor(buf53, (192,), requires_grad=True, is_leaf=True)  # primals_54
    buf54 = reader.storage('b0047a7552bc525c47c92c4cbd644db3285a6cdb', 37632, device=device(type='cuda', index=0))
    reader.tensor(buf54, (192, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_55
    buf55 = reader.storage('a18469774717a45643bb54d092fa9431161f6f4b', 768, device=device(type='cuda', index=0))
    reader.tensor(buf55, (192,), requires_grad=True, is_leaf=True)  # primals_56
    buf56 = reader.storage('a95b0736c0be947427ac11f9bb75ffa44907d8b9', 768, device=device(type='cuda', index=0))
    reader.tensor(buf56, (192,), requires_grad=True, is_leaf=True)  # primals_57
    buf57 = reader.storage('d522d5b7a4b25bd4ca9b2ee44f826c3736918c22', 768, device=device(type='cuda', index=0))
    reader.tensor(buf57, (192,), requires_grad=True, is_leaf=True)  # primals_58
    buf58 = reader.storage('add7bcb80495a3f43c078f215bad1531bca6b1a4', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768, 192), requires_grad=True, is_leaf=True)  # primals_59
    buf59 = reader.storage('f096f99610d526fabfe2878558f7921dd8bb97c9', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), requires_grad=True, is_leaf=True)  # primals_60
    buf60 = reader.storage('7fa8575f778eb3b72dac3c966b4eb95883209484', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf60, (192, 768), requires_grad=True, is_leaf=True)  # primals_61
    buf61 = reader.storage('94dae3d8c20db4efc990c3df44039aeb0c35b28a', 768, device=device(type='cuda', index=0))
    reader.tensor(buf61, (192,), requires_grad=True, is_leaf=True)  # primals_62
    buf62 = reader.storage('073d129eb071c9d97775192a9d76c598550c3725', 37632, device=device(type='cuda', index=0))
    reader.tensor(buf62, (192, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_63
    buf63 = reader.storage('738fe9e8c801be227f485583d52acffffb1a07ca', 768, device=device(type='cuda', index=0))
    reader.tensor(buf63, (192,), requires_grad=True, is_leaf=True)  # primals_64
    buf64 = reader.storage('b70491353df731979f8c9946bdeb432395701f1c', 768, device=device(type='cuda', index=0))
    reader.tensor(buf64, (192,), requires_grad=True, is_leaf=True)  # primals_65
    buf65 = reader.storage('0801281c7c2647d88abeea773eab70d33c4f532a', 768, device=device(type='cuda', index=0))
    reader.tensor(buf65, (192,), requires_grad=True, is_leaf=True)  # primals_66
    buf66 = reader.storage('8a66733ad012cb47a330422c0e398bc22de9c68f', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 192), requires_grad=True, is_leaf=True)  # primals_67
    buf67 = reader.storage('f3d45abc4ddbf3ea8bf158d88670919fb74230ef', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), requires_grad=True, is_leaf=True)  # primals_68
    buf68 = reader.storage('667c92f54ed8f4ffec949b13347bba67253ce370', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf68, (192, 768), requires_grad=True, is_leaf=True)  # primals_69
    buf69 = reader.storage('f9a58cbe4eff2cd73b313de8cf1fab23b7c23e82', 768, device=device(type='cuda', index=0))
    reader.tensor(buf69, (192,), requires_grad=True, is_leaf=True)  # primals_70
    buf70 = reader.storage('8b270e44f8eb7e9eebea33c2236426081a75f6ca', 37632, device=device(type='cuda', index=0))
    reader.tensor(buf70, (192, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_71
    buf71 = reader.storage('a72d5aada42c519d3d52981d7c1c5ed4028bc790', 768, device=device(type='cuda', index=0))
    reader.tensor(buf71, (192,), requires_grad=True, is_leaf=True)  # primals_72
    buf72 = reader.storage('3188251682bfd1db226c95aeb08826ebd1a23e15', 768, device=device(type='cuda', index=0))
    reader.tensor(buf72, (192,), requires_grad=True, is_leaf=True)  # primals_73
    buf73 = reader.storage('31a9d83351d037ba2dd64d57eab2e9ac08b69aac', 768, device=device(type='cuda', index=0))
    reader.tensor(buf73, (192,), requires_grad=True, is_leaf=True)  # primals_74
    buf74 = reader.storage('750652f9da3ba087bf36c5f9547927296b071971', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 192), requires_grad=True, is_leaf=True)  # primals_75
    buf75 = reader.storage('191f50ccc1ab5c0172d79c52cff5a99509c2e684', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), requires_grad=True, is_leaf=True)  # primals_76
    buf76 = reader.storage('d16f4d3b7ce6930e23a46b6a95c3492de921e924', 589824, device=device(type='cuda', index=0))
    reader.tensor(buf76, (192, 768), requires_grad=True, is_leaf=True)  # primals_77
    buf77 = reader.storage('bd2ff6408b957d5bf9e1a76282deffa533044962', 768, device=device(type='cuda', index=0))
    reader.tensor(buf77, (192,), requires_grad=True, is_leaf=True)  # primals_78
    buf78 = reader.storage('39c95aab8c3a48d70e5e4eb09edf150fd71208c1', 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384, 192, 2, 2), (768, 1, 384, 192), requires_grad=True, is_leaf=True)  # primals_79
    buf79 = reader.storage('2d3793cba4d4b06bd1ee4fdb9d8e40f5842c4f50', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf79, (384,), requires_grad=True, is_leaf=True)  # primals_80
    buf80 = reader.storage('5ea87aa1a21eba9a8308c3dfd62528194489ae7d', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf80, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_81
    buf81 = reader.storage('aa046e8b470be763857f319360e1d9bafd7b136d', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (384,), requires_grad=True, is_leaf=True)  # primals_82
    buf82 = reader.storage('25abc1dd4e1b95b4061150fac4f6eb033c1382ef', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384,), requires_grad=True, is_leaf=True)  # primals_83
    buf83 = reader.storage('75187bf63f163b398047e7ac24c52534dda720b7', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf83, (384,), requires_grad=True, is_leaf=True)  # primals_84
    buf84 = reader.storage('c2eb70ca11d85cd80d55e8f50e1b105f4791fffc', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536, 384), requires_grad=True, is_leaf=True)  # primals_85
    buf85 = reader.storage('a9d51b8eaf6c01fe4ae1054e728c7b8118e27afa', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1536,), requires_grad=True, is_leaf=True)  # primals_86
    buf86 = reader.storage('bfbc6f46c9b442dd29f2548db3d3f6c547a0ea4e', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (384, 1536), requires_grad=True, is_leaf=True)  # primals_87
    buf87 = reader.storage('35f2bcea8fb6979dc7d088b524d2a2e218df64de', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384,), requires_grad=True, is_leaf=True)  # primals_88
    buf88 = reader.storage('a3db9684724070dac3d1e69e165b322c13634e7b', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_89
    buf89 = reader.storage('c47bbe2b573e18424a871a4c44e3d2b719e18ba1', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384,), requires_grad=True, is_leaf=True)  # primals_90
    buf90 = reader.storage('613ba675aa878b18a675ab7ebd20ba0baf389d35', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), requires_grad=True, is_leaf=True)  # primals_91
    buf91 = reader.storage('bcb5918b72ab87570611913353c32803faecd073', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf91, (384,), requires_grad=True, is_leaf=True)  # primals_92
    buf92 = reader.storage('e765bb331235a7bbfdc815b3a2f07def8c264fed', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1536, 384), requires_grad=True, is_leaf=True)  # primals_93
    buf93 = reader.storage('8ee13b537b5f60e68cf4f42f73f8421106f66f87', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1536,), requires_grad=True, is_leaf=True)  # primals_94
    buf94 = reader.storage('e3df7642561ca76bf9f01bcac9772ae372085362', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384, 1536), requires_grad=True, is_leaf=True)  # primals_95
    buf95 = reader.storage('cd639b719a3b0b686eb24508984b9e3e59ff6ab4', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (384,), requires_grad=True, is_leaf=True)  # primals_96
    buf96 = reader.storage('b431432517813d2ccbfe5b116dfbc870f2dd641f', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf96, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_97
    buf97 = reader.storage('093dccd422d51703b8068f5efe0251b86f30ac11', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf97, (384,), requires_grad=True, is_leaf=True)  # primals_98
    buf98 = reader.storage('3a997b2499a58369dd66e31ae9c51b65f3158ab4', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384,), requires_grad=True, is_leaf=True)  # primals_99
    buf99 = reader.storage('1d06054cef854eb7735f0245c66f344c77b87f93', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384,), requires_grad=True, is_leaf=True)  # primals_100
    buf100 = reader.storage('b7c13d55a14ede09ab5e4a6bac7acbcc986d72b5', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536, 384), requires_grad=True, is_leaf=True)  # primals_101
    buf101 = reader.storage('903a843caf0aa7fae2303631ee61b87eee6228d3', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1536,), requires_grad=True, is_leaf=True)  # primals_102
    buf102 = reader.storage('26a820ddbbfd20a050d958dcefdcf14ae741709d', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384, 1536), requires_grad=True, is_leaf=True)  # primals_103
    buf103 = reader.storage('95c47509bef8ad4c2b19138f54ac32f7e9703ac3', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf103, (384,), requires_grad=True, is_leaf=True)  # primals_104
    buf104 = reader.storage('0877810c0e8045442d888204071217be025cf541', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_105
    buf105 = reader.storage('b3cf4b44ef65e911373d7e0e708afe5e36c9bfbc', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (384,), requires_grad=True, is_leaf=True)  # primals_106
    buf106 = reader.storage('5b08d485b1c784c677df02184d96f4d64800f85d', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384,), requires_grad=True, is_leaf=True)  # primals_107
    buf107 = reader.storage('3bdb1585bcf0c85168875b4ae669302e8c2d17f6', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384,), requires_grad=True, is_leaf=True)  # primals_108
    buf108 = reader.storage('c39091f05195968dc0101d5e9436308507557cfe', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536, 384), requires_grad=True, is_leaf=True)  # primals_109
    buf109 = reader.storage('828a496194e603aa417670246c1f5b0dff1a5841', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1536,), requires_grad=True, is_leaf=True)  # primals_110
    buf110 = reader.storage('7f9fceb90d94c7d01e582bd1d1233b8d7b9d392e', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384, 1536), requires_grad=True, is_leaf=True)  # primals_111
    buf111 = reader.storage('b5b2cf7e96d3d740d6a141672247bab8524084d8', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), requires_grad=True, is_leaf=True)  # primals_112
    buf112 = reader.storage('ca177057815ea84114e43ebbbb0083fb7c9ce7d4', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_113
    buf113 = reader.storage('080b77524ea5d3e940757ffc4efe8a041ee26e9a', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384,), requires_grad=True, is_leaf=True)  # primals_114
    buf114 = reader.storage('a1718e72d425d0c1fa6c5afaafc856c84b5e08db', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (384,), requires_grad=True, is_leaf=True)  # primals_115
    buf115 = reader.storage('11373639b80788e3272cd8612f28dae09b460a43', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf115, (384,), requires_grad=True, is_leaf=True)  # primals_116
    buf116 = reader.storage('f3fc5e147e6b1548678b9dacd09884c7a7c7752c', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536, 384), requires_grad=True, is_leaf=True)  # primals_117
    buf117 = reader.storage('8ebb643df72395a5aa780300797aa8c1fe9a6e9e', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1536,), requires_grad=True, is_leaf=True)  # primals_118
    buf118 = reader.storage('d6454bfa558f02702040910d2abb900e7d77d801', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf118, (384, 1536), requires_grad=True, is_leaf=True)  # primals_119
    buf119 = reader.storage('1141fc336da337e575a0329dcd58813aec7263fa', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf119, (384,), requires_grad=True, is_leaf=True)  # primals_120
    buf120 = reader.storage('e17b2d981afba5c53e32743d1ecc12a548e2c108', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf120, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_121
    buf121 = reader.storage('8f788bdc576f8930d3abae773b5e2587b317fa84', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384,), requires_grad=True, is_leaf=True)  # primals_122
    buf122 = reader.storage('d72bf4f724a91edf7ac726c1141cb368fadebc70', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384,), requires_grad=True, is_leaf=True)  # primals_123
    buf123 = reader.storage('73546473116d37ad1442cb0c0a80fbc373e3d37d', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), requires_grad=True, is_leaf=True)  # primals_124
    buf124 = reader.storage('2ba2fb6d02d7cce44ba1fef859ac8cfe3ad60ac0', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1536, 384), requires_grad=True, is_leaf=True)  # primals_125
    buf125 = reader.storage('6ed4e424c94c98fad4552e97f98c0381212be06c', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1536,), requires_grad=True, is_leaf=True)  # primals_126
    buf126 = reader.storage('36797da1627f8e29b23917da49ea4ba2b6db0e70', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384, 1536), requires_grad=True, is_leaf=True)  # primals_127
    buf127 = reader.storage('6b222d210cfd83bd37e60e290e12f56c8e98c83a', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), requires_grad=True, is_leaf=True)  # primals_128
    buf128 = reader.storage('a138b744d25ef568ba648bbabdac36f2b4ecd77e', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_129
    buf129 = reader.storage('fe5cb774a2cd2c3727f5517afc5f1ccba7cf5f49', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), requires_grad=True, is_leaf=True)  # primals_130
    buf130 = reader.storage('94b02d4952c81d866825bca017c0d96b0d077e2d', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384,), requires_grad=True, is_leaf=True)  # primals_131
    buf131 = reader.storage('5e9f627851df89873e035ce5f63b2a2ea1777031', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (384,), requires_grad=True, is_leaf=True)  # primals_132
    buf132 = reader.storage('ea5712e2b9ae547615045f4267949f46bf0482bf', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536, 384), requires_grad=True, is_leaf=True)  # primals_133
    buf133 = reader.storage('14c4921c19868487b69a0a8b8ecc653ca383386d', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1536,), requires_grad=True, is_leaf=True)  # primals_134
    buf134 = reader.storage('593c25cb5356277367214769e489b4fb081be01c', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf134, (384, 1536), requires_grad=True, is_leaf=True)  # primals_135
    buf135 = reader.storage('ec02d5b7de299fd92c6dc39fdce9a1dc5a94b829', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf135, (384,), requires_grad=True, is_leaf=True)  # primals_136
    buf136 = reader.storage('f5d55b59c53efe5e89187190fb01a30d82e1ca3a', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf136, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_137
    buf137 = reader.storage('cfe7933f1edb0bf86f209d53c2f89e2a2b642d61', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384,), requires_grad=True, is_leaf=True)  # primals_138
    buf138 = reader.storage('f8f7648e65dba9af3a3622b761cb403217d66c87', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), requires_grad=True, is_leaf=True)  # primals_139
    buf139 = reader.storage('a46a767500bbd391f37765156470bf8f0c7b55e0', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384,), requires_grad=True, is_leaf=True)  # primals_140
    buf140 = reader.storage('adca2a0dc29fa9d95136eb77b5f6391cad58aae9', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1536, 384), requires_grad=True, is_leaf=True)  # primals_141
    buf141 = reader.storage('28f414d7ab568eb99005afbb887992732d88c227', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1536,), requires_grad=True, is_leaf=True)  # primals_142
    buf142 = reader.storage('7a29c6ec183dd9c4045d1c4229ae156939ea9cf5', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf142, (384, 1536), requires_grad=True, is_leaf=True)  # primals_143
    buf143 = reader.storage('369416effb6b1ada2429828fc95d7d2908b7646b', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf143, (384,), requires_grad=True, is_leaf=True)  # primals_144
    buf144 = reader.storage('6e08ec277b7f4c535d602c0c964e392df74cea11', 75264, device=device(type='cuda', index=0))
    reader.tensor(buf144, (384, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_145
    buf145 = reader.storage('eda6ca01b95d310a826efda5fd5eabbbe20acd9c', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384,), requires_grad=True, is_leaf=True)  # primals_146
    buf146 = reader.storage('3370a7fc8672b0c379706b0c41d405e90fb0e06b', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf146, (384,), requires_grad=True, is_leaf=True)  # primals_147
    buf147 = reader.storage('3de5b4b51ebe25a1b911a8247ad2277a049679ea', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf147, (384,), requires_grad=True, is_leaf=True)  # primals_148
    buf148 = reader.storage('9afd595d9d63d800f7b03a84574a32a1b8ee10c8', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf148, (1536, 384), requires_grad=True, is_leaf=True)  # primals_149
    buf149 = reader.storage('cf0d5be87756ad5d932b11422a9839cb8b060569', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (1536,), requires_grad=True, is_leaf=True)  # primals_150
    buf150 = reader.storage('e5d76d934c5169afa75451683f2ef31c1fb7fb67', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf150, (384, 1536), requires_grad=True, is_leaf=True)  # primals_151
    buf151 = reader.storage('77c4bdf14aded06f24b221c5d86128526c839af6', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384,), requires_grad=True, is_leaf=True)  # primals_152
    buf152 = reader.storage('28f0238124091a234b2add82ff94480e27d0ab08', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768, 384, 2, 2), (1536, 1, 768, 384), requires_grad=True, is_leaf=True)  # primals_153
    buf153 = reader.storage('846a836b10b4ee0ea909dbc11a76ba8f62e1e2b5', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768,), requires_grad=True, is_leaf=True)  # primals_154
    buf154 = reader.storage('fec566103510ef2da16ece70ff058dad92d5b837', 150528, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_155
    buf155 = reader.storage('ead0f0796b5436537c48d8f6af6e61f40a702c2e', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), requires_grad=True, is_leaf=True)  # primals_156
    buf156 = reader.storage('68d9217d1c0d7a3264fd127b31d3d7886151d34a', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), requires_grad=True, is_leaf=True)  # primals_157
    buf157 = reader.storage('b136631b64d2815626a91d611488b9d61c87313b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), requires_grad=True, is_leaf=True)  # primals_158
    buf158 = reader.storage('0e404edfbb3238a788191d23a57aa5f10265b2de', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf158, (3072, 768), requires_grad=True, is_leaf=True)  # primals_159
    buf159 = reader.storage('50ddf9ddca8eb91499c616edc326837810e0e210', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf159, (3072,), requires_grad=True, is_leaf=True)  # primals_160
    buf160 = reader.storage('2e9b65d759af6c63287b76a859f59fe5ac2b204e', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768, 3072), requires_grad=True, is_leaf=True)  # primals_161
    buf161 = reader.storage('c22b2a8f124e1853fbe50cb8be8c365590038b5a', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), requires_grad=True, is_leaf=True)  # primals_162
    buf162 = reader.storage('42522f5afb16e2e6283e5d39bfc619a1219ed51b', 150528, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_163
    buf163 = reader.storage('f4ecaaceda58e01efa282e3f8a7a086b9536de76', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), requires_grad=True, is_leaf=True)  # primals_164
    buf164 = reader.storage('03ca0d349b48a02af48b86027c05b81d943f0e45', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), requires_grad=True, is_leaf=True)  # primals_165
    buf165 = reader.storage('adc4868d8aaa42708c42053aebce0f67b903cf65', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768,), requires_grad=True, is_leaf=True)  # primals_166
    buf166 = reader.storage('482ab72d152cd4a73291bd9d21e07171f5c90cac', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf166, (3072, 768), requires_grad=True, is_leaf=True)  # primals_167
    buf167 = reader.storage('3d1bc62ec17832dfaf4c2e4ced818dcbcec4d4eb', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf167, (3072,), requires_grad=True, is_leaf=True)  # primals_168
    buf168 = reader.storage('57e50b0832cefe0fce9bf1f3af9726267971a0ee', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768, 3072), requires_grad=True, is_leaf=True)  # primals_169
    buf169 = reader.storage('ef2e0fbcc34769facc32ad966e49ea7e9305a89e', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768,), requires_grad=True, is_leaf=True)  # primals_170
    buf170 = reader.storage('9e51ef563e23a5e447e3ab090ebe85459ca984fc', 150528, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768, 1, 7, 7), (49, 1, 7, 1), requires_grad=True, is_leaf=True)  # primals_171
    buf171 = reader.storage('ee881fa8c903ce5898420a36755d9c57aee7fa3b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768,), requires_grad=True, is_leaf=True)  # primals_172
    buf172 = reader.storage('c4762b592959f54e2771e887031d82468fcda812', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), requires_grad=True, is_leaf=True)  # primals_173
    buf173 = reader.storage('2d85b7e0d4389f1b044334ee3268d45db3ee508c', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), requires_grad=True, is_leaf=True)  # primals_174
    buf174 = reader.storage('84eb47295b5d794abecd519c131822e222c0130f', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf174, (3072, 768), requires_grad=True, is_leaf=True)  # primals_175
    buf175 = reader.storage('4f55d37d694e9ca2c8b18e8a83621c60377253eb', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf175, (3072,), requires_grad=True, is_leaf=True)  # primals_176
    buf176 = reader.storage('7c73b2e9a086a1822d4d2d697613c8a9fdbf3db8', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768, 3072), requires_grad=True, is_leaf=True)  # primals_177
    buf177 = reader.storage('fbd585d3e9797a35ae9135a1d626c9628965614d', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), requires_grad=True, is_leaf=True)  # primals_178
    buf178 = reader.storage('c4acf8c175bef22d3f2b7d8f628ae153357047cd', 384, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1, 96, 1, 1), (96, 1, 96, 96), requires_grad=True, is_leaf=True)  # primals_179
    buf179 = reader.storage('8cff18dc4878c5f619769227f086c8bf0852d06e', 768, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1, 192, 1, 1), (192, 1, 192, 192), requires_grad=True, is_leaf=True)  # primals_180
    buf180 = reader.storage('5fcb035a1352838d0490bcf78a38d84461715e0c', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1, 384, 1, 1), (384, 1, 384, 384), requires_grad=True, is_leaf=True)  # primals_181
    buf181 = reader.storage('40eb22234dd662e37fbc3202804cce015c835142', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1, 768, 1, 1), (768, 1, 768, 768), requires_grad=True, is_leaf=True)  # primals_182
    buf182 = reader.storage('7bf5cd78d1b189bdcd56e144cf811bf12cee8535', 640, device=device(type='cuda', index=0))
    reader.tensor(buf182, (32, 5, 1, 1), (5, 1, 5, 5), requires_grad=True, is_leaf=True)  # primals_183
    buf183 = reader.storage('7880ea8b4dd378794739003a08fe6d27892018ca', 128, device=device(type='cuda', index=0))
    reader.tensor(buf183, (32,), requires_grad=True, is_leaf=True)  # primals_184
    buf184 = reader.storage('7d665349a222edb3c11a104dff035d267b625cdf', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf184, (32, 32, 1, 1), (32, 1, 32, 32), requires_grad=True, is_leaf=True)  # primals_185
    buf185 = reader.storage('ae47ebc2c012df68fffe844e43c87c12cc07275a', 128, device=device(type='cuda', index=0))
    reader.tensor(buf185, (32,), requires_grad=True, is_leaf=True)  # primals_186
    buf186 = reader.storage('79e5bb9a82586a40dd45f7d7740edfa5d54c46cc', 128, device=device(type='cuda', index=0))
    reader.tensor(buf186, (1, 32, 1, 1), (32, 1, 32, 32), requires_grad=True, is_leaf=True)  # primals_187
    buf187 = reader.storage('a15bb975f5dc190ac8efcad752b6b5c47ad9fea1', 4, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1,), requires_grad=True, is_leaf=True)  # primals_188
    buf188 = reader.storage('6d0ec89046b87ffe93178bcfccc399bd25fd58ee', 12, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1, 3, 1, 1), (3, 1, 3, 3), is_leaf=True)  # primals_189
    buf189 = reader.storage('600ab3a9f5d2fccbbfd84b0ace46e78ee9642433', 12, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1, 3, 1, 1), (3, 1, 3, 3), is_leaf=True)  # primals_190
    buf190 = reader.storage('b8c662976334bc7d8de703148ff3a727e59e33bb', 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf190, (32, 3, 64, 64), (12288, 1, 192, 3), is_leaf=True)  # primals_191
    buf191 = reader.storage('a0727a7f49ca97cb0acb0332dba2df43ef7915f7', 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf191, (32, 3, 64, 64), (12288, 1, 192, 3), is_leaf=True)  # primals_192
    buf192 = reader.storage('74b5a2fdfabc7a51d3f92822e7de87e579c5413f', 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf192, (32, 3, 64, 64), (12288, 1, 192, 3), is_leaf=True)  # primals_193
    buf193 = reader.storage('05fa883c6f27c7ea93c5974565fd0a66f57cf06c', 128, device=device(type='cuda', index=0))
    reader.tensor(buf193, (32, 1, 1, 1), is_leaf=True)  # primals_194
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=True, command='run', save_dir='/home/drhead/PerceptualSimilarity/torch_compile_debug/run_2024_03_07_13_42_30_109863-pid_2401357/minifier/checkpoints', tracing_mode='real', check_str=None)
