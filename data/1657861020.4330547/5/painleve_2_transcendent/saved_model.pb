��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
9
	IdentityN

input2T
output2T"
T
list(type)(0
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
<
Selu
features"T
activations"T"
Ttype:
2
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
 �"serve*2.8.22v2.8.2-0-g2ea19cbb5758΄
t
first/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namefirst/kernel
m
 first/kernel/Read/ReadVariableOpReadVariableOpfirst/kernel*
_output_shapes

:
*
dtype0
l

first/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
first/bias
e
first/bias/Read/ReadVariableOpReadVariableOp
first/bias*
_output_shapes
:
*
dtype0
v
second/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namesecond/kernel
o
!second/kernel/Read/ReadVariableOpReadVariableOpsecond/kernel*
_output_shapes

:

*
dtype0
n
second/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namesecond/bias
g
second/bias/Read/ReadVariableOpReadVariableOpsecond/bias*
_output_shapes
:
*
dtype0
t
third/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namethird/kernel
m
 third/kernel/Read/ReadVariableOpReadVariableOpthird/kernel*
_output_shapes

:

*
dtype0
l

third/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
third/bias
e
third/bias/Read/ReadVariableOpReadVariableOp
third/bias*
_output_shapes
:
*
dtype0
v
fourth/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namefourth/kernel
o
!fourth/kernel/Read/ReadVariableOpReadVariableOpfourth/kernel*
_output_shapes

:
*
dtype0
n
fourth/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefourth/bias
g
fourth/bias/Read/ReadVariableOpReadVariableOpfourth/bias*
_output_shapes
:*
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
�
Adam/first/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/first/kernel/m
{
'Adam/first/kernel/m/Read/ReadVariableOpReadVariableOpAdam/first/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/first/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/first/bias/m
s
%Adam/first/bias/m/Read/ReadVariableOpReadVariableOpAdam/first/bias/m*
_output_shapes
:
*
dtype0
�
Adam/second/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/second/kernel/m
}
(Adam/second/kernel/m/Read/ReadVariableOpReadVariableOpAdam/second/kernel/m*
_output_shapes

:

*
dtype0
|
Adam/second/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/second/bias/m
u
&Adam/second/bias/m/Read/ReadVariableOpReadVariableOpAdam/second/bias/m*
_output_shapes
:
*
dtype0
�
Adam/third/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/third/kernel/m
{
'Adam/third/kernel/m/Read/ReadVariableOpReadVariableOpAdam/third/kernel/m*
_output_shapes

:

*
dtype0
z
Adam/third/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/third/bias/m
s
%Adam/third/bias/m/Read/ReadVariableOpReadVariableOpAdam/third/bias/m*
_output_shapes
:
*
dtype0
�
Adam/fourth/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/fourth/kernel/m
}
(Adam/fourth/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fourth/kernel/m*
_output_shapes

:
*
dtype0
|
Adam/fourth/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fourth/bias/m
u
&Adam/fourth/bias/m/Read/ReadVariableOpReadVariableOpAdam/fourth/bias/m*
_output_shapes
:*
dtype0
�
Adam/first/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/first/kernel/v
{
'Adam/first/kernel/v/Read/ReadVariableOpReadVariableOpAdam/first/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/first/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/first/bias/v
s
%Adam/first/bias/v/Read/ReadVariableOpReadVariableOpAdam/first/bias/v*
_output_shapes
:
*
dtype0
�
Adam/second/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/second/kernel/v
}
(Adam/second/kernel/v/Read/ReadVariableOpReadVariableOpAdam/second/kernel/v*
_output_shapes

:

*
dtype0
|
Adam/second/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/second/bias/v
u
&Adam/second/bias/v/Read/ReadVariableOpReadVariableOpAdam/second/bias/v*
_output_shapes
:
*
dtype0
�
Adam/third/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/third/kernel/v
{
'Adam/third/kernel/v/Read/ReadVariableOpReadVariableOpAdam/third/kernel/v*
_output_shapes

:

*
dtype0
z
Adam/third/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/third/bias/v
s
%Adam/third/bias/v/Read/ReadVariableOpReadVariableOpAdam/third/bias/v*
_output_shapes
:
*
dtype0
�
Adam/fourth/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/fourth/kernel/v
}
(Adam/fourth/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fourth/kernel/v*
_output_shapes

:
*
dtype0
|
Adam/fourth/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fourth/bias/v
u
&Adam/fourth/bias/v/Read/ReadVariableOpReadVariableOpAdam/fourth/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
loss

signatures*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�
/iter

0beta_1

1beta_2
	2decay
3learning_ratemSmTmUmVmW mX'mY(mZv[v\v]v^v_ v`'va(vb*
<
0
1
2
3
4
 5
'6
(7*
<
0
1
2
3
4
 5
'6
(7*
* 
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 

9serving_default* 
\V
VARIABLE_VALUEfirst/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
first/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEsecond/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEsecond/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEthird/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
third/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEfourth/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEfourth/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

N0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ototal
	Pcount
Q	variables
R	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

Q	variables*
y
VARIABLE_VALUEAdam/first/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/first/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/second/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/second/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/third/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/third/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/fourth/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/fourth/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/first/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/first/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/second/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/second/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/third/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/third/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/fourth/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/fourth/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
serving_default_first_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_first_inputfirst/kernel
first/biassecond/kernelsecond/biasthird/kernel
third/biasfourth/kernelfourth/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_275458295
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename first/kernel/Read/ReadVariableOpfirst/bias/Read/ReadVariableOp!second/kernel/Read/ReadVariableOpsecond/bias/Read/ReadVariableOp third/kernel/Read/ReadVariableOpthird/bias/Read/ReadVariableOp!fourth/kernel/Read/ReadVariableOpfourth/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/first/kernel/m/Read/ReadVariableOp%Adam/first/bias/m/Read/ReadVariableOp(Adam/second/kernel/m/Read/ReadVariableOp&Adam/second/bias/m/Read/ReadVariableOp'Adam/third/kernel/m/Read/ReadVariableOp%Adam/third/bias/m/Read/ReadVariableOp(Adam/fourth/kernel/m/Read/ReadVariableOp&Adam/fourth/bias/m/Read/ReadVariableOp'Adam/first/kernel/v/Read/ReadVariableOp%Adam/first/bias/v/Read/ReadVariableOp(Adam/second/kernel/v/Read/ReadVariableOp&Adam/second/bias/v/Read/ReadVariableOp'Adam/third/kernel/v/Read/ReadVariableOp%Adam/third/bias/v/Read/ReadVariableOp(Adam/fourth/kernel/v/Read/ReadVariableOp&Adam/fourth/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_save_275458588
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefirst/kernel
first/biassecond/kernelsecond/biasthird/kernel
third/biasfourth/kernelfourth/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/first/kernel/mAdam/first/bias/mAdam/second/kernel/mAdam/second/bias/mAdam/third/kernel/mAdam/third/bias/mAdam/fourth/kernel/mAdam/fourth/bias/mAdam/first/kernel/vAdam/first/bias/vAdam/second/kernel/vAdam/second/bias/vAdam/third/kernel/vAdam/third/bias/vAdam/fourth/kernel/vAdam/fourth/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__traced_restore_275458691��
�'
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458233

inputs6
$first_matmul_readvariableop_resource:
3
%first_biasadd_readvariableop_resource:
7
%second_matmul_readvariableop_resource:

4
&second_biasadd_readvariableop_resource:
6
$third_matmul_readvariableop_resource:

3
%third_biasadd_readvariableop_resource:
7
%fourth_matmul_readvariableop_resource:
4
&fourth_biasadd_readvariableop_resource:
identity��first/BiasAdd/ReadVariableOp�first/MatMul/ReadVariableOp�fourth/BiasAdd/ReadVariableOp�fourth/MatMul/ReadVariableOp�second/BiasAdd/ReadVariableOp�second/MatMul/ReadVariableOp�third/BiasAdd/ReadVariableOp�third/MatMul/ReadVariableOp�
first/MatMul/ReadVariableOpReadVariableOp$first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0u
first/MatMulMatMulinputs#first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
first/BiasAdd/ReadVariableOpReadVariableOp%first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
first/BiasAddBiasAddfirst/MatMul:product:0$first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
O

first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	first/mulMulfirst/beta:output:0first/BiasAdd:output:0*
T0*'
_output_shapes
:���������
Y
first/SigmoidSigmoidfirst/mul:z:0*
T0*'
_output_shapes
:���������
o
first/mul_1Mulfirst/BiasAdd:output:0first/Sigmoid:y:0*
T0*'
_output_shapes
:���������
]
first/IdentityIdentityfirst/mul_1:z:0*
T0*'
_output_shapes
:���������
�
first/IdentityN	IdentityNfirst/mul_1:z:0first/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-275458204*:
_output_shapes(
&:���������
:���������
�
second/MatMul/ReadVariableOpReadVariableOp%second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
second/MatMulMatMulfirst/IdentityN:output:0$second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
second/BiasAdd/ReadVariableOpReadVariableOp&second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
second/BiasAddBiasAddsecond/MatMul:product:0%second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
d
second/SigmoidSigmoidsecond/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
third/MatMul/ReadVariableOpReadVariableOp$third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
third/MatMulMatMulsecond/Sigmoid:y:0#third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
third/BiasAdd/ReadVariableOpReadVariableOp%third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
third/BiasAddBiasAddthird/MatMul:product:0$third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
third/SigmoidSigmoidthird/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
fourth/MatMul/ReadVariableOpReadVariableOp%fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
fourth/MatMulMatMulthird/Sigmoid:y:0$fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
fourth/BiasAdd/ReadVariableOpReadVariableOp&fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
fourth/BiasAddBiasAddfourth/MatMul:product:0%fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������^
fourth/SeluSelufourth/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentityfourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/BiasAdd/ReadVariableOp^first/MatMul/ReadVariableOp^fourth/BiasAdd/ReadVariableOp^fourth/MatMul/ReadVariableOp^second/BiasAdd/ReadVariableOp^second/MatMul/ReadVariableOp^third/BiasAdd/ReadVariableOp^third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2<
first/BiasAdd/ReadVariableOpfirst/BiasAdd/ReadVariableOp2:
first/MatMul/ReadVariableOpfirst/MatMul/ReadVariableOp2>
fourth/BiasAdd/ReadVariableOpfourth/BiasAdd/ReadVariableOp2<
fourth/MatMul/ReadVariableOpfourth/MatMul/ReadVariableOp2>
second/BiasAdd/ReadVariableOpsecond/BiasAdd/ReadVariableOp2<
second/MatMul/ReadVariableOpsecond/MatMul/ReadVariableOp2<
third/BiasAdd/ReadVariableOpthird/BiasAdd/ReadVariableOp2:
third/MatMul/ReadVariableOpthird/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�'
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458272

inputs6
$first_matmul_readvariableop_resource:
3
%first_biasadd_readvariableop_resource:
7
%second_matmul_readvariableop_resource:

4
&second_biasadd_readvariableop_resource:
6
$third_matmul_readvariableop_resource:

3
%third_biasadd_readvariableop_resource:
7
%fourth_matmul_readvariableop_resource:
4
&fourth_biasadd_readvariableop_resource:
identity��first/BiasAdd/ReadVariableOp�first/MatMul/ReadVariableOp�fourth/BiasAdd/ReadVariableOp�fourth/MatMul/ReadVariableOp�second/BiasAdd/ReadVariableOp�second/MatMul/ReadVariableOp�third/BiasAdd/ReadVariableOp�third/MatMul/ReadVariableOp�
first/MatMul/ReadVariableOpReadVariableOp$first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0u
first/MatMulMatMulinputs#first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
first/BiasAdd/ReadVariableOpReadVariableOp%first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
first/BiasAddBiasAddfirst/MatMul:product:0$first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
O

first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	first/mulMulfirst/beta:output:0first/BiasAdd:output:0*
T0*'
_output_shapes
:���������
Y
first/SigmoidSigmoidfirst/mul:z:0*
T0*'
_output_shapes
:���������
o
first/mul_1Mulfirst/BiasAdd:output:0first/Sigmoid:y:0*
T0*'
_output_shapes
:���������
]
first/IdentityIdentityfirst/mul_1:z:0*
T0*'
_output_shapes
:���������
�
first/IdentityN	IdentityNfirst/mul_1:z:0first/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-275458243*:
_output_shapes(
&:���������
:���������
�
second/MatMul/ReadVariableOpReadVariableOp%second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
second/MatMulMatMulfirst/IdentityN:output:0$second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
second/BiasAdd/ReadVariableOpReadVariableOp&second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
second/BiasAddBiasAddsecond/MatMul:product:0%second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
d
second/SigmoidSigmoidsecond/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
third/MatMul/ReadVariableOpReadVariableOp$third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
third/MatMulMatMulsecond/Sigmoid:y:0#third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
third/BiasAdd/ReadVariableOpReadVariableOp%third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
third/BiasAddBiasAddthird/MatMul:product:0$third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
third/SigmoidSigmoidthird/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
fourth/MatMul/ReadVariableOpReadVariableOp%fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
fourth/MatMulMatMulthird/Sigmoid:y:0$fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
fourth/BiasAdd/ReadVariableOpReadVariableOp&fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
fourth/BiasAddBiasAddfourth/MatMul:product:0%fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������^
fourth/SeluSelufourth/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentityfourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/BiasAdd/ReadVariableOp^first/MatMul/ReadVariableOp^fourth/BiasAdd/ReadVariableOp^fourth/MatMul/ReadVariableOp^second/BiasAdd/ReadVariableOp^second/MatMul/ReadVariableOp^third/BiasAdd/ReadVariableOp^third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2<
first/BiasAdd/ReadVariableOpfirst/BiasAdd/ReadVariableOp2:
first/MatMul/ReadVariableOpfirst/MatMul/ReadVariableOp2>
fourth/BiasAdd/ReadVariableOpfourth/BiasAdd/ReadVariableOp2<
fourth/MatMul/ReadVariableOpfourth/MatMul/ReadVariableOp2>
second/BiasAdd/ReadVariableOpsecond/BiasAdd/ReadVariableOp2<
second/MatMul/ReadVariableOpsecond/MatMul/ReadVariableOp2<
third/BiasAdd/ReadVariableOpthird/BiasAdd/ReadVariableOp2:
third/MatMul/ReadVariableOpthird/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�B
�
"__inference__traced_save_275458588
file_prefix+
'savev2_first_kernel_read_readvariableop)
%savev2_first_bias_read_readvariableop,
(savev2_second_kernel_read_readvariableop*
&savev2_second_bias_read_readvariableop+
'savev2_third_kernel_read_readvariableop)
%savev2_third_bias_read_readvariableop,
(savev2_fourth_kernel_read_readvariableop*
&savev2_fourth_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_first_kernel_m_read_readvariableop0
,savev2_adam_first_bias_m_read_readvariableop3
/savev2_adam_second_kernel_m_read_readvariableop1
-savev2_adam_second_bias_m_read_readvariableop2
.savev2_adam_third_kernel_m_read_readvariableop0
,savev2_adam_third_bias_m_read_readvariableop3
/savev2_adam_fourth_kernel_m_read_readvariableop1
-savev2_adam_fourth_bias_m_read_readvariableop2
.savev2_adam_first_kernel_v_read_readvariableop0
,savev2_adam_first_bias_v_read_readvariableop3
/savev2_adam_second_kernel_v_read_readvariableop1
-savev2_adam_second_bias_v_read_readvariableop2
.savev2_adam_third_kernel_v_read_readvariableop0
,savev2_adam_third_bias_v_read_readvariableop3
/savev2_adam_fourth_kernel_v_read_readvariableop1
-savev2_adam_fourth_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_first_kernel_read_readvariableop%savev2_first_bias_read_readvariableop(savev2_second_kernel_read_readvariableop&savev2_second_bias_read_readvariableop'savev2_third_kernel_read_readvariableop%savev2_third_bias_read_readvariableop(savev2_fourth_kernel_read_readvariableop&savev2_fourth_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_first_kernel_m_read_readvariableop,savev2_adam_first_bias_m_read_readvariableop/savev2_adam_second_kernel_m_read_readvariableop-savev2_adam_second_bias_m_read_readvariableop.savev2_adam_third_kernel_m_read_readvariableop,savev2_adam_third_bias_m_read_readvariableop/savev2_adam_fourth_kernel_m_read_readvariableop-savev2_adam_fourth_bias_m_read_readvariableop.savev2_adam_first_kernel_v_read_readvariableop,savev2_adam_first_bias_v_read_readvariableop/savev2_adam_second_kernel_v_read_readvariableop-savev2_adam_second_bias_v_read_readvariableop.savev2_adam_third_kernel_v_read_readvariableop,savev2_adam_third_bias_v_read_readvariableop/savev2_adam_fourth_kernel_v_read_readvariableop-savev2_adam_fourth_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	�
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
�: :
:
:

:
:

:
:
:: : : : : : : :
:
:

:
:

:
:
::
:
:

:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
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

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	
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

:
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

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
:: 

_output_shapes
: 
�
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458146
first_input!
first_275458125:

first_275458127:
"
second_275458130:


second_275458132:
!
third_275458135:


third_275458137:
"
fourth_275458140:

fourth_275458142:
identity��first/StatefulPartitionedCall�fourth/StatefulPartitionedCall�second/StatefulPartitionedCall�third/StatefulPartitionedCall�
first/StatefulPartitionedCallStatefulPartitionedCallfirst_inputfirst_275458125first_275458127*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_first_layer_call_and_return_conditional_losses_275457894�
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_275458130second_275458132*
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
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_second_layer_call_and_return_conditional_losses_275457911�
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_275458135third_275458137*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_third_layer_call_and_return_conditional_losses_275457928�
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_275458140fourth_275458142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_fourth_layer_call_and_return_conditional_losses_275457945v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_input
�
�
)__inference_first_layer_call_fn_275458304

inputs
unknown:
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_first_layer_call_and_return_conditional_losses_275457894o
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�
$__inference__wrapped_model_275457869
first_inputE
3sequential_131_first_matmul_readvariableop_resource:
B
4sequential_131_first_biasadd_readvariableop_resource:
F
4sequential_131_second_matmul_readvariableop_resource:

C
5sequential_131_second_biasadd_readvariableop_resource:
E
3sequential_131_third_matmul_readvariableop_resource:

B
4sequential_131_third_biasadd_readvariableop_resource:
F
4sequential_131_fourth_matmul_readvariableop_resource:
C
5sequential_131_fourth_biasadd_readvariableop_resource:
identity��+sequential_131/first/BiasAdd/ReadVariableOp�*sequential_131/first/MatMul/ReadVariableOp�,sequential_131/fourth/BiasAdd/ReadVariableOp�+sequential_131/fourth/MatMul/ReadVariableOp�,sequential_131/second/BiasAdd/ReadVariableOp�+sequential_131/second/MatMul/ReadVariableOp�+sequential_131/third/BiasAdd/ReadVariableOp�*sequential_131/third/MatMul/ReadVariableOp�
*sequential_131/first/MatMul/ReadVariableOpReadVariableOp3sequential_131_first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
sequential_131/first/MatMulMatMulfirst_input2sequential_131/first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
+sequential_131/first/BiasAdd/ReadVariableOpReadVariableOp4sequential_131_first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_131/first/BiasAddBiasAdd%sequential_131/first/MatMul:product:03sequential_131/first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
^
sequential_131/first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential_131/first/mulMul"sequential_131/first/beta:output:0%sequential_131/first/BiasAdd:output:0*
T0*'
_output_shapes
:���������
w
sequential_131/first/SigmoidSigmoidsequential_131/first/mul:z:0*
T0*'
_output_shapes
:���������
�
sequential_131/first/mul_1Mul%sequential_131/first/BiasAdd:output:0 sequential_131/first/Sigmoid:y:0*
T0*'
_output_shapes
:���������
{
sequential_131/first/IdentityIdentitysequential_131/first/mul_1:z:0*
T0*'
_output_shapes
:���������
�
sequential_131/first/IdentityN	IdentityNsequential_131/first/mul_1:z:0%sequential_131/first/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-275457840*:
_output_shapes(
&:���������
:���������
�
+sequential_131/second/MatMul/ReadVariableOpReadVariableOp4sequential_131_second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
sequential_131/second/MatMulMatMul'sequential_131/first/IdentityN:output:03sequential_131/second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
,sequential_131/second/BiasAdd/ReadVariableOpReadVariableOp5sequential_131_second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_131/second/BiasAddBiasAdd&sequential_131/second/MatMul:product:04sequential_131/second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
sequential_131/second/SigmoidSigmoid&sequential_131/second/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
*sequential_131/third/MatMul/ReadVariableOpReadVariableOp3sequential_131_third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
sequential_131/third/MatMulMatMul!sequential_131/second/Sigmoid:y:02sequential_131/third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
+sequential_131/third/BiasAdd/ReadVariableOpReadVariableOp4sequential_131_third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_131/third/BiasAddBiasAdd%sequential_131/third/MatMul:product:03sequential_131/third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
sequential_131/third/SigmoidSigmoid%sequential_131/third/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
+sequential_131/fourth/MatMul/ReadVariableOpReadVariableOp4sequential_131_fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
sequential_131/fourth/MatMulMatMul sequential_131/third/Sigmoid:y:03sequential_131/fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_131/fourth/BiasAdd/ReadVariableOpReadVariableOp5sequential_131_fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_131/fourth/BiasAddBiasAdd&sequential_131/fourth/MatMul:product:04sequential_131/fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_131/fourth/SeluSelu&sequential_131/fourth/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_131/fourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_131/first/BiasAdd/ReadVariableOp+^sequential_131/first/MatMul/ReadVariableOp-^sequential_131/fourth/BiasAdd/ReadVariableOp,^sequential_131/fourth/MatMul/ReadVariableOp-^sequential_131/second/BiasAdd/ReadVariableOp,^sequential_131/second/MatMul/ReadVariableOp,^sequential_131/third/BiasAdd/ReadVariableOp+^sequential_131/third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2Z
+sequential_131/first/BiasAdd/ReadVariableOp+sequential_131/first/BiasAdd/ReadVariableOp2X
*sequential_131/first/MatMul/ReadVariableOp*sequential_131/first/MatMul/ReadVariableOp2\
,sequential_131/fourth/BiasAdd/ReadVariableOp,sequential_131/fourth/BiasAdd/ReadVariableOp2Z
+sequential_131/fourth/MatMul/ReadVariableOp+sequential_131/fourth/MatMul/ReadVariableOp2\
,sequential_131/second/BiasAdd/ReadVariableOp,sequential_131/second/BiasAdd/ReadVariableOp2Z
+sequential_131/second/MatMul/ReadVariableOp+sequential_131/second/MatMul/ReadVariableOp2Z
+sequential_131/third/BiasAdd/ReadVariableOp+sequential_131/third/BiasAdd/ReadVariableOp2X
*sequential_131/third/MatMul/ReadVariableOp*sequential_131/third/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_input
�
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458058

inputs!
first_275458037:

first_275458039:
"
second_275458042:


second_275458044:
!
third_275458047:


third_275458049:
"
fourth_275458052:

fourth_275458054:
identity��first/StatefulPartitionedCall�fourth/StatefulPartitionedCall�second/StatefulPartitionedCall�third/StatefulPartitionedCall�
first/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_275458037first_275458039*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_first_layer_call_and_return_conditional_losses_275457894�
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_275458042second_275458044*
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
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_second_layer_call_and_return_conditional_losses_275457911�
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_275458047third_275458049*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_third_layer_call_and_return_conditional_losses_275457928�
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_275458052fourth_275458054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_fourth_layer_call_and_return_conditional_losses_275457945v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275457952

inputs!
first_275457895:

first_275457897:
"
second_275457912:


second_275457914:
!
third_275457929:


third_275457931:
"
fourth_275457946:

fourth_275457948:
identity��first/StatefulPartitionedCall�fourth/StatefulPartitionedCall�second/StatefulPartitionedCall�third/StatefulPartitionedCall�
first/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_275457895first_275457897*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_first_layer_call_and_return_conditional_losses_275457894�
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_275457912second_275457914*
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
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_second_layer_call_and_return_conditional_losses_275457911�
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_275457929third_275457931*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_third_layer_call_and_return_conditional_losses_275457928�
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_275457946fourth_275457948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_fourth_layer_call_and_return_conditional_losses_275457945v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
'__inference_signature_wrapper_275458295
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_275457869o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_input
�
�
D__inference_first_layer_call_and_return_conditional_losses_275458322

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:


identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������
�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-275458314*:
_output_shapes(
&:���������
:���������
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_275458502
result_grads_0
result_grads_1
mul_first_beta
mul_first_biasadd
identityp
mulMulmul_first_betamul_first_biasadd^result_grads_0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
a
mul_1Mulmul_first_betamul_first_biasadd*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*N
_input_shapes=
;:���������
:���������
: :���������
:W S
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������

�
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458122
first_input!
first_275458101:

first_275458103:
"
second_275458106:


second_275458108:
!
third_275458111:


third_275458113:
"
fourth_275458116:

fourth_275458118:
identity��first/StatefulPartitionedCall�fourth/StatefulPartitionedCall�second/StatefulPartitionedCall�third/StatefulPartitionedCall�
first/StatefulPartitionedCallStatefulPartitionedCallfirst_inputfirst_275458101first_275458103*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_first_layer_call_and_return_conditional_losses_275457894�
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_275458106second_275458108*
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
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_second_layer_call_and_return_conditional_losses_275457911�
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_275458111third_275458113*
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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_third_layer_call_and_return_conditional_losses_275457928�
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_275458116fourth_275458118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_fourth_layer_call_and_return_conditional_losses_275457945v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_input
�
}
&__inference_internal_grad_fn_275458538
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*N
_input_shapes=
;:���������
:���������
: :���������
:W S
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������

�	
�
2__inference_sequential_131_layer_call_fn_275458098
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_input
�

�
D__inference_third_layer_call_and_return_conditional_losses_275458362

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
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
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
D__inference_first_layer_call_and_return_conditional_losses_275457894

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:


identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������
�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-275457886*:
_output_shapes(
&:���������
:���������
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_fourth_layer_call_and_return_conditional_losses_275457945

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_275458520
result_grads_0
result_grads_1
mul_first_beta
mul_first_biasadd
identityp
mulMulmul_first_betamul_first_biasadd^result_grads_0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
a
mul_1Mulmul_first_betamul_first_biasadd*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*N
_input_shapes=
;:���������
:���������
: :���������
:W S
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������

�	
�
2__inference_sequential_131_layer_call_fn_275458194

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_second_layer_call_fn_275458331

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
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_second_layer_call_and_return_conditional_losses_275457911o
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
D__inference_third_layer_call_and_return_conditional_losses_275457928

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
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
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
)__inference_third_layer_call_fn_275458351

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
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_third_layer_call_and_return_conditional_losses_275457928o
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
�
2__inference_sequential_131_layer_call_fn_275458173

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_131_layer_call_and_return_conditional_losses_275457952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_275458484
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*N
_input_shapes=
;:���������
:���������
: :���������
:W S
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������

�

�
E__inference_fourth_layer_call_and_return_conditional_losses_275458382

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_275458466
result_grads_0
result_grads_1!
mul_sequential_131_first_beta$
 mul_sequential_131_first_biasadd
identity�
mulMulmul_sequential_131_first_beta mul_sequential_131_first_biasadd^result_grads_0*
T0*'
_output_shapes
:���������
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������

mul_1Mulmul_sequential_131_first_beta mul_sequential_131_first_biasadd*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*N
_input_shapes=
;:���������
:���������
: :���������
:W S
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������

�

�
E__inference_second_layer_call_and_return_conditional_losses_275457911

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
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
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
E__inference_second_layer_call_and_return_conditional_losses_275458342

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
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
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�|
�
%__inference__traced_restore_275458691
file_prefix/
assignvariableop_first_kernel:
+
assignvariableop_1_first_bias:
2
 assignvariableop_2_second_kernel:

,
assignvariableop_3_second_bias:
1
assignvariableop_4_third_kernel:

+
assignvariableop_5_third_bias:
2
 assignvariableop_6_fourth_kernel:
,
assignvariableop_7_fourth_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: 9
'assignvariableop_15_adam_first_kernel_m:
3
%assignvariableop_16_adam_first_bias_m:
:
(assignvariableop_17_adam_second_kernel_m:

4
&assignvariableop_18_adam_second_bias_m:
9
'assignvariableop_19_adam_third_kernel_m:

3
%assignvariableop_20_adam_third_bias_m:
:
(assignvariableop_21_adam_fourth_kernel_m:
4
&assignvariableop_22_adam_fourth_bias_m:9
'assignvariableop_23_adam_first_kernel_v:
3
%assignvariableop_24_adam_first_bias_v:
:
(assignvariableop_25_adam_second_kernel_v:

4
&assignvariableop_26_adam_second_bias_v:
9
'assignvariableop_27_adam_third_kernel_v:

3
%assignvariableop_28_adam_third_bias_v:
:
(assignvariableop_29_adam_fourth_kernel_v:
4
&assignvariableop_30_adam_fourth_bias_v:
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_first_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_first_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_second_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_second_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_third_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_third_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_fourth_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_fourth_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_first_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_first_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_second_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_second_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_third_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_third_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_fourth_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_fourth_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_first_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_first_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_second_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_second_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_third_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_third_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_fourth_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_fourth_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
�
�
*__inference_fourth_layer_call_fn_275458371

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_fourth_layer_call_and_return_conditional_losses_275457945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
�
2__inference_sequential_131_layer_call_fn_275457971
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_131_layer_call_and_return_conditional_losses_275457952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namefirst_inputB
&__inference_internal_grad_fn_275458466CustomGradient-275457840B
&__inference_internal_grad_fn_275458484CustomGradient-275457886B
&__inference_internal_grad_fn_275458502CustomGradient-275458204B
&__inference_internal_grad_fn_275458520CustomGradient-275458243B
&__inference_internal_grad_fn_275458538CustomGradient-275458314"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
first_input4
serving_default_first_input:0���������:
fourth0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�k
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
loss

signatures"
_tf_keras_sequential
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/iter

0beta_1

1beta_2
	2decay
3learning_ratemSmTmUmVmW mX'mY(mZv[v\v]v^v_ v`'va(vb"
	optimizer
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_sequential_131_layer_call_fn_275457971
2__inference_sequential_131_layer_call_fn_275458173
2__inference_sequential_131_layer_call_fn_275458194
2__inference_sequential_131_layer_call_fn_275458098�
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
�2�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458233
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458272
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458122
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458146�
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
$__inference__wrapped_model_275457869first_input"�
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
�2��
���
FullArgSpec
args�
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
9serving_default"
signature_map
:
2first/kernel
:
2
first/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_first_layer_call_fn_275458304�
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
D__inference_first_layer_call_and_return_conditional_losses_275458322�
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
:

2second/kernel
:
2second/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_second_layer_call_fn_275458331�
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
E__inference_second_layer_call_and_return_conditional_losses_275458342�
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
:

2third/kernel
:
2
third/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_third_layer_call_fn_275458351�
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
D__inference_third_layer_call_and_return_conditional_losses_275458362�
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
:
2fourth/kernel
:2fourth/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_fourth_layer_call_fn_275458371�
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
E__inference_fourth_layer_call_and_return_conditional_losses_275458382�
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_signature_wrapper_275458295first_input"�
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
 
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
	Ototal
	Pcount
Q	variables
R	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
#:!
2Adam/first/kernel/m
:
2Adam/first/bias/m
$:"

2Adam/second/kernel/m
:
2Adam/second/bias/m
#:!

2Adam/third/kernel/m
:
2Adam/third/bias/m
$:"
2Adam/fourth/kernel/m
:2Adam/fourth/bias/m
#:!
2Adam/first/kernel/v
:
2Adam/first/bias/v
$:"

2Adam/second/kernel/v
:
2Adam/second/bias/v
#:!

2Adam/third/kernel/v
:
2Adam/third/bias/v
$:"
2Adam/fourth/kernel/v
:2Adam/fourth/bias/v
EbC
sequential_131/first/beta:0$__inference__wrapped_model_275457869
HbF
sequential_131/first/BiasAdd:0$__inference__wrapped_model_275457869
PbN
beta:0D__inference_first_layer_call_and_return_conditional_losses_275457894
SbQ
	BiasAdd:0D__inference_first_layer_call_and_return_conditional_losses_275457894
_b]
first/beta:0M__inference_sequential_131_layer_call_and_return_conditional_losses_275458233
bb`
first/BiasAdd:0M__inference_sequential_131_layer_call_and_return_conditional_losses_275458233
_b]
first/beta:0M__inference_sequential_131_layer_call_and_return_conditional_losses_275458272
bb`
first/BiasAdd:0M__inference_sequential_131_layer_call_and_return_conditional_losses_275458272
PbN
beta:0D__inference_first_layer_call_and_return_conditional_losses_275458322
SbQ
	BiasAdd:0D__inference_first_layer_call_and_return_conditional_losses_275458322�
$__inference__wrapped_model_275457869q '(4�1
*�'
%�"
first_input���������
� "/�,
*
fourth �
fourth����������
D__inference_first_layer_call_and_return_conditional_losses_275458322\/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� |
)__inference_first_layer_call_fn_275458304O/�,
%�"
 �
inputs���������
� "����������
�
E__inference_fourth_layer_call_and_return_conditional_losses_275458382\'(/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� }
*__inference_fourth_layer_call_fn_275458371O'(/�,
%�"
 �
inputs���������

� "�����������
&__inference_internal_grad_fn_275458466�cde�b
[�X

 
(�%
result_grads_0���������

(�%
result_grads_1���������

� "$�!

 
�
1���������
�
&__inference_internal_grad_fn_275458484�efe�b
[�X

 
(�%
result_grads_0���������

(�%
result_grads_1���������

� "$�!

 
�
1���������
�
&__inference_internal_grad_fn_275458502�ghe�b
[�X

 
(�%
result_grads_0���������

(�%
result_grads_1���������

� "$�!

 
�
1���������
�
&__inference_internal_grad_fn_275458520�ije�b
[�X

 
(�%
result_grads_0���������

(�%
result_grads_1���������

� "$�!

 
�
1���������
�
&__inference_internal_grad_fn_275458538�kle�b
[�X

 
(�%
result_grads_0���������

(�%
result_grads_1���������

� "$�!

 
�
1���������
�
E__inference_second_layer_call_and_return_conditional_losses_275458342\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� }
*__inference_second_layer_call_fn_275458331O/�,
%�"
 �
inputs���������

� "����������
�
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458122o '(<�9
2�/
%�"
first_input���������
p 

 
� "%�"
�
0���������
� �
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458146o '(<�9
2�/
%�"
first_input���������
p

 
� "%�"
�
0���������
� �
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458233j '(7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
M__inference_sequential_131_layer_call_and_return_conditional_losses_275458272j '(7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
2__inference_sequential_131_layer_call_fn_275457971b '(<�9
2�/
%�"
first_input���������
p 

 
� "�����������
2__inference_sequential_131_layer_call_fn_275458098b '(<�9
2�/
%�"
first_input���������
p

 
� "�����������
2__inference_sequential_131_layer_call_fn_275458173] '(7�4
-�*
 �
inputs���������
p 

 
� "�����������
2__inference_sequential_131_layer_call_fn_275458194] '(7�4
-�*
 �
inputs���������
p

 
� "�����������
'__inference_signature_wrapper_275458295� '(C�@
� 
9�6
4
first_input%�"
first_input���������"/�,
*
fourth �
fourth����������
D__inference_third_layer_call_and_return_conditional_losses_275458362\ /�,
%�"
 �
inputs���������

� "%�"
�
0���������

� |
)__inference_third_layer_call_fn_275458351O /�,
%�"
 �
inputs���������

� "����������
