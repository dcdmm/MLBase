ܕ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
v
my_dense_30/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namemy_dense_30/w
o
!my_dense_30/w/Read/ReadVariableOpReadVariableOpmy_dense_30/w*
_output_shapes

:
*
dtype0
r
my_dense_30/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namemy_dense_30/b
k
!my_dense_30/b/Read/ReadVariableOpReadVariableOpmy_dense_30/b*
_output_shapes
:
*
dtype0
v
my_dense_31/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namemy_dense_31/w
o
!my_dense_31/w/Read/ReadVariableOpReadVariableOpmy_dense_31/w*
_output_shapes

:

*
dtype0
r
my_dense_31/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namemy_dense_31/b
k
!my_dense_31/b/Read/ReadVariableOpReadVariableOpmy_dense_31/b*
_output_shapes
:
*
dtype0
v
my_dense_32/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namemy_dense_32/w
o
!my_dense_32/w/Read/ReadVariableOpReadVariableOpmy_dense_32/w*
_output_shapes

:
*
dtype0
r
my_dense_32/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemy_dense_32/b
k
!my_dense_32/b/Read/ReadVariableOpReadVariableOpmy_dense_32/b*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/my_dense_30/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/my_dense_30/w/m
}
(Adam/my_dense_30/w/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_30/w/m*
_output_shapes

:
*
dtype0
�
Adam/my_dense_30/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/my_dense_30/b/m
y
(Adam/my_dense_30/b/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_30/b/m*
_output_shapes
:
*
dtype0
�
Adam/my_dense_31/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/my_dense_31/w/m
}
(Adam/my_dense_31/w/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_31/w/m*
_output_shapes

:

*
dtype0
�
Adam/my_dense_31/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/my_dense_31/b/m
y
(Adam/my_dense_31/b/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_31/b/m*
_output_shapes
:
*
dtype0
�
Adam/my_dense_32/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/my_dense_32/w/m
}
(Adam/my_dense_32/w/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_32/w/m*
_output_shapes

:
*
dtype0
�
Adam/my_dense_32/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/my_dense_32/b/m
y
(Adam/my_dense_32/b/m/Read/ReadVariableOpReadVariableOpAdam/my_dense_32/b/m*
_output_shapes
:*
dtype0
�
Adam/my_dense_30/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/my_dense_30/w/v
}
(Adam/my_dense_30/w/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_30/w/v*
_output_shapes

:
*
dtype0
�
Adam/my_dense_30/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/my_dense_30/b/v
y
(Adam/my_dense_30/b/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_30/b/v*
_output_shapes
:
*
dtype0
�
Adam/my_dense_31/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/my_dense_31/w/v
}
(Adam/my_dense_31/w/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_31/w/v*
_output_shapes

:

*
dtype0
�
Adam/my_dense_31/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/my_dense_31/b/v
y
(Adam/my_dense_31/b/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_31/b/v*
_output_shapes
:
*
dtype0
�
Adam/my_dense_32/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/my_dense_32/w/v
}
(Adam/my_dense_32/w/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_32/w/v*
_output_shapes

:
*
dtype0
�
Adam/my_dense_32/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/my_dense_32/b/v
y
(Adam/my_dense_32/b/v/Read/ReadVariableOpReadVariableOpAdam/my_dense_32/b/v*
_output_shapes
:*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�$
value�$B�$ B�$
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
 
`
w
b
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api
`
w
b
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api
`
w
b
	variables
trainable_variables
regularization_losses
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratemEmFmGmHmImJvKvLvMvNvOvP
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
	trainable_variables

regularization_losses
 
TR
VARIABLE_VALUEmy_dense_30/w1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_dense_30/b1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
 
TR
VARIABLE_VALUEmy_dense_31/w1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_dense_31/b1layer_with_weights-1/b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
 
TR
VARIABLE_VALUEmy_dense_32/w1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_dense_32/b1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	<total
	=count
>	variables
?	keras_api
D
	@total
	Acount
B
_fn_kwargs
C	variables
D	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

>	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

C	variables
wu
VARIABLE_VALUEAdam/my_dense_30/w/mMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_30/b/mMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_31/w/mMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_31/b/mMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_32/w/mMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_32/b/mMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_30/w/vMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_30/b/vMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_31/w/vMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_31/b/vMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_32/w/vMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/my_dense_32/b/vMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_dcdmmPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dcdmmmy_dense_30/wmy_dense_30/bmy_dense_31/wmy_dense_31/bmy_dense_32/wmy_dense_32/b*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_33102
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!my_dense_30/w/Read/ReadVariableOp!my_dense_30/b/Read/ReadVariableOp!my_dense_31/w/Read/ReadVariableOp!my_dense_31/b/Read/ReadVariableOp!my_dense_32/w/Read/ReadVariableOp!my_dense_32/b/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/my_dense_30/w/m/Read/ReadVariableOp(Adam/my_dense_30/b/m/Read/ReadVariableOp(Adam/my_dense_31/w/m/Read/ReadVariableOp(Adam/my_dense_31/b/m/Read/ReadVariableOp(Adam/my_dense_32/w/m/Read/ReadVariableOp(Adam/my_dense_32/b/m/Read/ReadVariableOp(Adam/my_dense_30/w/v/Read/ReadVariableOp(Adam/my_dense_30/b/v/Read/ReadVariableOp(Adam/my_dense_31/w/v/Read/ReadVariableOp(Adam/my_dense_31/b/v/Read/ReadVariableOp(Adam/my_dense_32/w/v/Read/ReadVariableOp(Adam/my_dense_32/b/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_33345
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_dense_30/wmy_dense_30/bmy_dense_31/wmy_dense_31/bmy_dense_32/wmy_dense_32/b	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/my_dense_30/w/mAdam/my_dense_30/b/mAdam/my_dense_31/w/mAdam/my_dense_31/b/mAdam/my_dense_32/w/mAdam/my_dense_32/b/mAdam/my_dense_30/w/vAdam/my_dense_30/b/vAdam/my_dense_31/w/vAdam/my_dense_31/b/vAdam/my_dense_32/w/vAdam/my_dense_32/b/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_33436��
�k
�
!__inference__traced_restore_33436
file_prefix0
assignvariableop_my_dense_30_w:
.
 assignvariableop_1_my_dense_30_b:
2
 assignvariableop_2_my_dense_31_w:

.
 assignvariableop_3_my_dense_31_b:
2
 assignvariableop_4_my_dense_32_w:
.
 assignvariableop_5_my_dense_32_b:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: :
(assignvariableop_15_adam_my_dense_30_w_m:
6
(assignvariableop_16_adam_my_dense_30_b_m:
:
(assignvariableop_17_adam_my_dense_31_w_m:

6
(assignvariableop_18_adam_my_dense_31_b_m:
:
(assignvariableop_19_adam_my_dense_32_w_m:
6
(assignvariableop_20_adam_my_dense_32_b_m::
(assignvariableop_21_adam_my_dense_30_w_v:
6
(assignvariableop_22_adam_my_dense_30_b_v:
:
(assignvariableop_23_adam_my_dense_31_w_v:

6
(assignvariableop_24_adam_my_dense_31_b_v:
:
(assignvariableop_25_adam_my_dense_32_w_v:
6
(assignvariableop_26_adam_my_dense_32_b_v:
identity_28��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_my_dense_30_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_my_dense_30_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_my_dense_31_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_my_dense_31_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_my_dense_32_wIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_my_dense_32_bIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_my_dense_30_w_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_my_dense_30_b_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_my_dense_31_w_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_my_dense_31_b_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_my_dense_32_w_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_my_dense_32_b_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_my_dense_30_w_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_my_dense_30_b_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_my_dense_31_w_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_my_dense_31_b_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_my_dense_32_w_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_my_dense_32_b_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911

inputs0
matmul_readvariableop_resource:
)
add_readvariableop_resource:
identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894

inputs0
matmul_readvariableop_resource:

)
add_readvariableop_resource:

identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�:
�

__inference__traced_save_33345
file_prefix,
(savev2_my_dense_30_w_read_readvariableop,
(savev2_my_dense_30_b_read_readvariableop,
(savev2_my_dense_31_w_read_readvariableop,
(savev2_my_dense_31_b_read_readvariableop,
(savev2_my_dense_32_w_read_readvariableop,
(savev2_my_dense_32_b_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_my_dense_30_w_m_read_readvariableop3
/savev2_adam_my_dense_30_b_m_read_readvariableop3
/savev2_adam_my_dense_31_w_m_read_readvariableop3
/savev2_adam_my_dense_31_b_m_read_readvariableop3
/savev2_adam_my_dense_32_w_m_read_readvariableop3
/savev2_adam_my_dense_32_b_m_read_readvariableop3
/savev2_adam_my_dense_30_w_v_read_readvariableop3
/savev2_adam_my_dense_30_b_v_read_readvariableop3
/savev2_adam_my_dense_31_w_v_read_readvariableop3
/savev2_adam_my_dense_31_b_v_read_readvariableop3
/savev2_adam_my_dense_32_w_v_read_readvariableop3
/savev2_adam_my_dense_32_b_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_my_dense_30_w_read_readvariableop(savev2_my_dense_30_b_read_readvariableop(savev2_my_dense_31_w_read_readvariableop(savev2_my_dense_31_b_read_readvariableop(savev2_my_dense_32_w_read_readvariableop(savev2_my_dense_32_b_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_my_dense_30_w_m_read_readvariableop/savev2_adam_my_dense_30_b_m_read_readvariableop/savev2_adam_my_dense_31_w_m_read_readvariableop/savev2_adam_my_dense_31_b_m_read_readvariableop/savev2_adam_my_dense_32_w_m_read_readvariableop/savev2_adam_my_dense_32_b_m_read_readvariableop/savev2_adam_my_dense_30_w_v_read_readvariableop/savev2_adam_my_dense_30_b_v_read_readvariableop/savev2_adam_my_dense_31_w_v_read_readvariableop/savev2_adam_my_dense_31_b_v_read_readvariableop/savev2_adam_my_dense_32_w_v_read_readvariableop/savev2_adam_my_dense_32_b_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
:
:

:
:
:: : : : : : : : : :
:
:

:
:
::
:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: 
�
�
+__inference_my_dense_30_layer_call_fn_33193

inputs
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_32918

inputs#
my_dense_30_32878:

my_dense_30_32880:
#
my_dense_31_32895:


my_dense_31_32897:
#
my_dense_32_32912:

my_dense_32_32914:
identity��#my_dense_30/StatefulPartitionedCall�#my_dense_31/StatefulPartitionedCall�#my_dense_32/StatefulPartitionedCall�
#my_dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsmy_dense_30_32878my_dense_30_32880*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877z
tf.nn.relu_20/ReluRelu,my_dense_30/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_31/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_20/Relu:activations:0my_dense_31_32895my_dense_31_32897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894z
tf.nn.relu_21/ReluRelu,my_dense_31/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_32/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_21/Relu:activations:0my_dense_32_32912my_dense_32_32914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911{
IdentityIdentity,my_dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^my_dense_30/StatefulPartitionedCall$^my_dense_31/StatefulPartitionedCall$^my_dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2J
#my_dense_30/StatefulPartitionedCall#my_dense_30/StatefulPartitionedCall2J
#my_dense_31/StatefulPartitionedCall#my_dense_31/StatefulPartitionedCall2J
#my_dense_32/StatefulPartitionedCall#my_dense_32/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_10_layer_call_fn_33035	
dcdmm
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldcdmmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_33003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_33003

inputs#
my_dense_30_32985:

my_dense_30_32987:
#
my_dense_31_32991:


my_dense_31_32993:
#
my_dense_32_32997:

my_dense_32_32999:
identity��#my_dense_30/StatefulPartitionedCall�#my_dense_31/StatefulPartitionedCall�#my_dense_32/StatefulPartitionedCall�
#my_dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsmy_dense_30_32985my_dense_30_32987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877z
tf.nn.relu_20/ReluRelu,my_dense_30/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_31/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_20/Relu:activations:0my_dense_31_32991my_dense_31_32993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894z
tf.nn.relu_21/ReluRelu,my_dense_31/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_32/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_21/Relu:activations:0my_dense_32_32997my_dense_32_32999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911{
IdentityIdentity,my_dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^my_dense_30/StatefulPartitionedCall$^my_dense_31/StatefulPartitionedCall$^my_dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2J
#my_dense_30/StatefulPartitionedCall#my_dense_30/StatefulPartitionedCall2J
#my_dense_31/StatefulPartitionedCall#my_dense_31/StatefulPartitionedCall2J
#my_dense_32/StatefulPartitionedCall#my_dense_32/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_10_layer_call_fn_33136

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_33003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_33102	
dcdmm
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldcdmmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_32860o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�	
�
F__inference_my_dense_30_layer_call_and_return_conditional_losses_33203

inputs0
matmul_readvariableop_resource:
)
add_readvariableop_resource:

identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_33056	
dcdmm#
my_dense_30_33038:

my_dense_30_33040:
#
my_dense_31_33044:


my_dense_31_33046:
#
my_dense_32_33050:

my_dense_32_33052:
identity��#my_dense_30/StatefulPartitionedCall�#my_dense_31/StatefulPartitionedCall�#my_dense_32/StatefulPartitionedCall�
#my_dense_30/StatefulPartitionedCallStatefulPartitionedCalldcdmmmy_dense_30_33038my_dense_30_33040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877z
tf.nn.relu_20/ReluRelu,my_dense_30/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_31/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_20/Relu:activations:0my_dense_31_33044my_dense_31_33046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894z
tf.nn.relu_21/ReluRelu,my_dense_31/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_32/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_21/Relu:activations:0my_dense_32_33050my_dense_32_33052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911{
IdentityIdentity,my_dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^my_dense_30/StatefulPartitionedCall$^my_dense_31/StatefulPartitionedCall$^my_dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2J
#my_dense_30/StatefulPartitionedCall#my_dense_30/StatefulPartitionedCall2J
#my_dense_31/StatefulPartitionedCall#my_dense_31/StatefulPartitionedCall2J
#my_dense_32/StatefulPartitionedCall#my_dense_32/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_33184

inputs<
*my_dense_30_matmul_readvariableop_resource:
5
'my_dense_30_add_readvariableop_resource:
<
*my_dense_31_matmul_readvariableop_resource:

5
'my_dense_31_add_readvariableop_resource:
<
*my_dense_32_matmul_readvariableop_resource:
5
'my_dense_32_add_readvariableop_resource:
identity��!my_dense_30/MatMul/ReadVariableOp�my_dense_30/add/ReadVariableOp�!my_dense_31/MatMul/ReadVariableOp�my_dense_31/add/ReadVariableOp�!my_dense_32/MatMul/ReadVariableOp�my_dense_32/add/ReadVariableOp�
!my_dense_30/MatMul/ReadVariableOpReadVariableOp*my_dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
my_dense_30/MatMulMatMulinputs)my_dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_30/add/ReadVariableOpReadVariableOp'my_dense_30_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_30/addAddV2my_dense_30/MatMul:product:0&my_dense_30/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
tf.nn.relu_20/ReluRelumy_dense_30/add:z:0*
T0*'
_output_shapes
:���������
�
!my_dense_31/MatMul/ReadVariableOpReadVariableOp*my_dense_31_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
my_dense_31/MatMulMatMul tf.nn.relu_20/Relu:activations:0)my_dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_31/add/ReadVariableOpReadVariableOp'my_dense_31_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_31/addAddV2my_dense_31/MatMul:product:0&my_dense_31/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
tf.nn.relu_21/ReluRelumy_dense_31/add:z:0*
T0*'
_output_shapes
:���������
�
!my_dense_32/MatMul/ReadVariableOpReadVariableOp*my_dense_32_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
my_dense_32/MatMulMatMul tf.nn.relu_21/Relu:activations:0)my_dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
my_dense_32/add/ReadVariableOpReadVariableOp'my_dense_32_add_readvariableop_resource*
_output_shapes
:*
dtype0�
my_dense_32/addAddV2my_dense_32/MatMul:product:0&my_dense_32/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
IdentityIdentitymy_dense_32/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^my_dense_30/MatMul/ReadVariableOp^my_dense_30/add/ReadVariableOp"^my_dense_31/MatMul/ReadVariableOp^my_dense_31/add/ReadVariableOp"^my_dense_32/MatMul/ReadVariableOp^my_dense_32/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!my_dense_30/MatMul/ReadVariableOp!my_dense_30/MatMul/ReadVariableOp2@
my_dense_30/add/ReadVariableOpmy_dense_30/add/ReadVariableOp2F
!my_dense_31/MatMul/ReadVariableOp!my_dense_31/MatMul/ReadVariableOp2@
my_dense_31/add/ReadVariableOpmy_dense_31/add/ReadVariableOp2F
!my_dense_32/MatMul/ReadVariableOp!my_dense_32/MatMul/ReadVariableOp2@
my_dense_32/add/ReadVariableOpmy_dense_32/add/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
 __inference__wrapped_model_32860	
dcdmmE
3model_10_my_dense_30_matmul_readvariableop_resource:
>
0model_10_my_dense_30_add_readvariableop_resource:
E
3model_10_my_dense_31_matmul_readvariableop_resource:

>
0model_10_my_dense_31_add_readvariableop_resource:
E
3model_10_my_dense_32_matmul_readvariableop_resource:
>
0model_10_my_dense_32_add_readvariableop_resource:
identity��*model_10/my_dense_30/MatMul/ReadVariableOp�'model_10/my_dense_30/add/ReadVariableOp�*model_10/my_dense_31/MatMul/ReadVariableOp�'model_10/my_dense_31/add/ReadVariableOp�*model_10/my_dense_32/MatMul/ReadVariableOp�'model_10/my_dense_32/add/ReadVariableOp�
*model_10/my_dense_30/MatMul/ReadVariableOpReadVariableOp3model_10_my_dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
model_10/my_dense_30/MatMulMatMuldcdmm2model_10/my_dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
'model_10/my_dense_30/add/ReadVariableOpReadVariableOp0model_10_my_dense_30_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_10/my_dense_30/addAddV2%model_10/my_dense_30/MatMul:product:0/model_10/my_dense_30/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
s
model_10/tf.nn.relu_20/ReluRelumodel_10/my_dense_30/add:z:0*
T0*'
_output_shapes
:���������
�
*model_10/my_dense_31/MatMul/ReadVariableOpReadVariableOp3model_10_my_dense_31_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
model_10/my_dense_31/MatMulMatMul)model_10/tf.nn.relu_20/Relu:activations:02model_10/my_dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
'model_10/my_dense_31/add/ReadVariableOpReadVariableOp0model_10_my_dense_31_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_10/my_dense_31/addAddV2%model_10/my_dense_31/MatMul:product:0/model_10/my_dense_31/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
s
model_10/tf.nn.relu_21/ReluRelumodel_10/my_dense_31/add:z:0*
T0*'
_output_shapes
:���������
�
*model_10/my_dense_32/MatMul/ReadVariableOpReadVariableOp3model_10_my_dense_32_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
model_10/my_dense_32/MatMulMatMul)model_10/tf.nn.relu_21/Relu:activations:02model_10/my_dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_10/my_dense_32/add/ReadVariableOpReadVariableOp0model_10_my_dense_32_add_readvariableop_resource*
_output_shapes
:*
dtype0�
model_10/my_dense_32/addAddV2%model_10/my_dense_32/MatMul:product:0/model_10/my_dense_32/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentitymodel_10/my_dense_32/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^model_10/my_dense_30/MatMul/ReadVariableOp(^model_10/my_dense_30/add/ReadVariableOp+^model_10/my_dense_31/MatMul/ReadVariableOp(^model_10/my_dense_31/add/ReadVariableOp+^model_10/my_dense_32/MatMul/ReadVariableOp(^model_10/my_dense_32/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2X
*model_10/my_dense_30/MatMul/ReadVariableOp*model_10/my_dense_30/MatMul/ReadVariableOp2R
'model_10/my_dense_30/add/ReadVariableOp'model_10/my_dense_30/add/ReadVariableOp2X
*model_10/my_dense_31/MatMul/ReadVariableOp*model_10/my_dense_31/MatMul/ReadVariableOp2R
'model_10/my_dense_31/add/ReadVariableOp'model_10/my_dense_31/add/ReadVariableOp2X
*model_10/my_dense_32/MatMul/ReadVariableOp*model_10/my_dense_32/MatMul/ReadVariableOp2R
'model_10/my_dense_32/add/ReadVariableOp'model_10/my_dense_32/add/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�
�
(__inference_model_10_layer_call_fn_32933	
dcdmm
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldcdmmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_32918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�	
�
F__inference_my_dense_31_layer_call_and_return_conditional_losses_33222

inputs0
matmul_readvariableop_resource:

)
add_readvariableop_resource:

identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877

inputs0
matmul_readvariableop_resource:
)
add_readvariableop_resource:

identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_33077	
dcdmm#
my_dense_30_33059:

my_dense_30_33061:
#
my_dense_31_33065:


my_dense_31_33067:
#
my_dense_32_33071:

my_dense_32_33073:
identity��#my_dense_30/StatefulPartitionedCall�#my_dense_31/StatefulPartitionedCall�#my_dense_32/StatefulPartitionedCall�
#my_dense_30/StatefulPartitionedCallStatefulPartitionedCalldcdmmmy_dense_30_33059my_dense_30_33061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_30_layer_call_and_return_conditional_losses_32877z
tf.nn.relu_20/ReluRelu,my_dense_30/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_31/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_20/Relu:activations:0my_dense_31_33065my_dense_31_33067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894z
tf.nn.relu_21/ReluRelu,my_dense_31/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������
�
#my_dense_32/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_21/Relu:activations:0my_dense_32_33071my_dense_32_33073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911{
IdentityIdentity,my_dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^my_dense_30/StatefulPartitionedCall$^my_dense_31/StatefulPartitionedCall$^my_dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2J
#my_dense_30/StatefulPartitionedCall#my_dense_30/StatefulPartitionedCall2J
#my_dense_31/StatefulPartitionedCall#my_dense_31/StatefulPartitionedCall2J
#my_dense_32/StatefulPartitionedCall#my_dense_32/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namedcdmm
�
�
(__inference_model_10_layer_call_fn_33119

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_32918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_10_layer_call_and_return_conditional_losses_33160

inputs<
*my_dense_30_matmul_readvariableop_resource:
5
'my_dense_30_add_readvariableop_resource:
<
*my_dense_31_matmul_readvariableop_resource:

5
'my_dense_31_add_readvariableop_resource:
<
*my_dense_32_matmul_readvariableop_resource:
5
'my_dense_32_add_readvariableop_resource:
identity��!my_dense_30/MatMul/ReadVariableOp�my_dense_30/add/ReadVariableOp�!my_dense_31/MatMul/ReadVariableOp�my_dense_31/add/ReadVariableOp�!my_dense_32/MatMul/ReadVariableOp�my_dense_32/add/ReadVariableOp�
!my_dense_30/MatMul/ReadVariableOpReadVariableOp*my_dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
my_dense_30/MatMulMatMulinputs)my_dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_30/add/ReadVariableOpReadVariableOp'my_dense_30_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_30/addAddV2my_dense_30/MatMul:product:0&my_dense_30/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
tf.nn.relu_20/ReluRelumy_dense_30/add:z:0*
T0*'
_output_shapes
:���������
�
!my_dense_31/MatMul/ReadVariableOpReadVariableOp*my_dense_31_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
my_dense_31/MatMulMatMul tf.nn.relu_20/Relu:activations:0)my_dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_31/add/ReadVariableOpReadVariableOp'my_dense_31_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_31/addAddV2my_dense_31/MatMul:product:0&my_dense_31/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
tf.nn.relu_21/ReluRelumy_dense_31/add:z:0*
T0*'
_output_shapes
:���������
�
!my_dense_32/MatMul/ReadVariableOpReadVariableOp*my_dense_32_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
my_dense_32/MatMulMatMul tf.nn.relu_21/Relu:activations:0)my_dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
my_dense_32/add/ReadVariableOpReadVariableOp'my_dense_32_add_readvariableop_resource*
_output_shapes
:*
dtype0�
my_dense_32/addAddV2my_dense_32/MatMul:product:0&my_dense_32/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
IdentityIdentitymy_dense_32/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^my_dense_30/MatMul/ReadVariableOp^my_dense_30/add/ReadVariableOp"^my_dense_31/MatMul/ReadVariableOp^my_dense_31/add/ReadVariableOp"^my_dense_32/MatMul/ReadVariableOp^my_dense_32/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!my_dense_30/MatMul/ReadVariableOp!my_dense_30/MatMul/ReadVariableOp2@
my_dense_30/add/ReadVariableOpmy_dense_30/add/ReadVariableOp2F
!my_dense_31/MatMul/ReadVariableOp!my_dense_31/MatMul/ReadVariableOp2@
my_dense_31/add/ReadVariableOpmy_dense_31/add/ReadVariableOp2F
!my_dense_32/MatMul/ReadVariableOp!my_dense_32/MatMul/ReadVariableOp2@
my_dense_32/add/ReadVariableOpmy_dense_32/add/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_my_dense_31_layer_call_fn_33212

inputs
unknown:


	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_31_layer_call_and_return_conditional_losses_32894o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
F__inference_my_dense_32_layer_call_and_return_conditional_losses_33241

inputs0
matmul_readvariableop_resource:
)
add_readvariableop_resource:
identity��MatMul/ReadVariableOp�add/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������s
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
+__inference_my_dense_32_layer_call_fn_33231

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_my_dense_32_layer_call_and_return_conditional_losses_32911o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
dcdmm.
serving_default_dcdmm:0���������?
my_dense_320
StatefulPartitionedCall:0���������tensorflow/serving/predict:�O
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
Q__call__
*R&call_and_return_all_conditional_losses
S_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
w
b
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
�
w
b
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
�
w
b
	variables
trainable_variables
regularization_losses
 	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratemEmFmGmHmImJvKvLvMvNvOvP"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
	trainable_variables

regularization_losses
Q__call__
S_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
,
Zserving_default"
signature_map
:
2my_dense_30/w
:
2my_dense_30/b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:

2my_dense_31/w
:
2my_dense_31/b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:
2my_dense_32/w
:2my_dense_32/b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	<total
	=count
>	variables
?	keras_api"
_tf_keras_metric
^
	@total
	Acount
B
_fn_kwargs
C	variables
D	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
<0
=1"
trackable_list_wrapper
-
>	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
$:"
2Adam/my_dense_30/w/m
 :
2Adam/my_dense_30/b/m
$:"

2Adam/my_dense_31/w/m
 :
2Adam/my_dense_31/b/m
$:"
2Adam/my_dense_32/w/m
 :2Adam/my_dense_32/b/m
$:"
2Adam/my_dense_30/w/v
 :
2Adam/my_dense_30/b/v
$:"

2Adam/my_dense_31/w/v
 :
2Adam/my_dense_31/b/v
$:"
2Adam/my_dense_32/w/v
 :2Adam/my_dense_32/b/v
�2�
(__inference_model_10_layer_call_fn_32933
(__inference_model_10_layer_call_fn_33119
(__inference_model_10_layer_call_fn_33136
(__inference_model_10_layer_call_fn_33035�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_10_layer_call_and_return_conditional_losses_33160
C__inference_model_10_layer_call_and_return_conditional_losses_33184
C__inference_model_10_layer_call_and_return_conditional_losses_33056
C__inference_model_10_layer_call_and_return_conditional_losses_33077�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_32860dcdmm"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_my_dense_30_layer_call_fn_33193�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_my_dense_30_layer_call_and_return_conditional_losses_33203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_my_dense_31_layer_call_fn_33212�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_my_dense_31_layer_call_and_return_conditional_losses_33222�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_my_dense_32_layer_call_fn_33231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_my_dense_32_layer_call_and_return_conditional_losses_33241�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_33102dcdmm"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_32860s.�+
$�!
�
dcdmm���������
� "9�6
4
my_dense_32%�"
my_dense_32����������
C__inference_model_10_layer_call_and_return_conditional_losses_33056g6�3
,�)
�
dcdmm���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_10_layer_call_and_return_conditional_losses_33077g6�3
,�)
�
dcdmm���������
p

 
� "%�"
�
0���������
� �
C__inference_model_10_layer_call_and_return_conditional_losses_33160h7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_10_layer_call_and_return_conditional_losses_33184h7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
(__inference_model_10_layer_call_fn_32933Z6�3
,�)
�
dcdmm���������
p 

 
� "�����������
(__inference_model_10_layer_call_fn_33035Z6�3
,�)
�
dcdmm���������
p

 
� "�����������
(__inference_model_10_layer_call_fn_33119[7�4
-�*
 �
inputs���������
p 

 
� "�����������
(__inference_model_10_layer_call_fn_33136[7�4
-�*
 �
inputs���������
p

 
� "�����������
F__inference_my_dense_30_layer_call_and_return_conditional_losses_33203\/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� ~
+__inference_my_dense_30_layer_call_fn_33193O/�,
%�"
 �
inputs���������
� "����������
�
F__inference_my_dense_31_layer_call_and_return_conditional_losses_33222\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� ~
+__inference_my_dense_31_layer_call_fn_33212O/�,
%�"
 �
inputs���������

� "����������
�
F__inference_my_dense_32_layer_call_and_return_conditional_losses_33241\/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� ~
+__inference_my_dense_32_layer_call_fn_33231O/�,
%�"
 �
inputs���������

� "�����������
#__inference_signature_wrapper_33102|7�4
� 
-�*
(
dcdmm�
dcdmm���������"9�6
4
my_dense_32%�"
my_dense_32���������