??!
?-?-
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
<
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ݝ 
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_45588
?
StatefulPartitionedCall_1StatefulPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_45593
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
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?6
Const_4Const*
_output_shapes	
:?*
dtype0*?6
value?6B?6?BatoBatBnoBdoorsB311BtoBtheBlateBforBminutesBbpdBaBholdBreleasedBonBtrainBofBs202BandBnowBinBdelayB5BisBtrainsBoutBwillBtrackBdueBdoorBm201BissuedBplatformBbackBt103BminuteBfromBs201BallBwasB1BwithBt441Bm901Bl30BcommuteBclearBcarBt449Bt229BopenBmanualBm161B6Bt377BholdingBdelayedBt451Bt363BpsfBperBm202B	departingBcheckBworkBt373Bt119Bt113Bk30BwelfareBtechBt371Bt367BreportsBordersBinoutBdelaysBcycleBcarsBboipBboardBadvisedBt385BsecondBscramB	reportingBmadeBm902Bm90BholeBhasBbeBauxBarrivingBareaBa50B8B10BturnBtpaBthatBtangoBt235Bt223Bt107BspeedBrrBpatronBodyBocyBnotBmedicalBm301B
inspectionBasBareB7B4BupBt519Bt383Bt381Bt369Bt365Bt329B
proceduresBpersonBitBbyBbutBanB9B3B13Bt915Bt517Bt505Bt503Bt379Bt327Bt231Bt225Bt109B
supervisorBstationB	shortagesBsfBserviceBsaBrunningBroadBreportBr302BoperatorBnewBm162BlineBintoBgoB	followingBdisturbanceBbeenBapproachingB2ByardBtowerBthroughBtbtBt513Bt511Bt455Bt445Bt361Bt337Bt241Bt239Bt221Bt101BsuspectBsubjectBstoppedBstaffedBspareBshortageBremovedBpatronsB
passengersBoryBoosBohyBmovingBm802Bm701Bm70Bm60Bm16Bm102BheBfwBearlierB
congestionB	compliantBcentralBbBwereBwaysideBwayBwatchBw40BtracksBtheyBt515Bt443Bt331Bt323Bt227BstaffingBsmokingBshortBsetBsecondsBrunBrollingBresidualBreportedBreplacementBr502BofficersBoffboardBmpBm601Bm402Bm302Bm101Bk302BgetBfotfBfightBextendedBestablishedB	emergencyBdwellsBdispatchingBc502BblanketBbalanceBagentB311sB15By10Bw101BverifiesB	unstaffedBunableBtryingBtroubleshootingBtrackwayBtotalBtimeBthenBtango1Bt901Bt509Bt461Bt459Bt453Bt375Bt325Bt237Bt233Bt117Bt115Bt105BsomeoneB	secondaryBs20BrushBruleBrouteBrestrictionBresetB	procedureBpossibleBonceBoffBnearBmultipleBmedic16Bm702Bm602Bm401Bm2Bm10Bk304Bk10B
instructedB
individualBholesBheavyBgoodBdoB
continuingBcontinueBconsistB
compoundedBclearedBcheckedB	charlie53BcatBc802Bc80Bc402Bc401Bc101BbypassBbrakeBalignedBagainBafterBaerialBa201Ba101BzeroBy103By05BwosBwomanBwhoBwhenBw403Bw15xlBventilationBuorBunitBtryBtripBtriedBtrackingBthruBthreeBthisBthereBtbt10Bt771Bt507Bt501Bt457Bt4539Bt447Bt4439Bt4Bt333Bt301Bt2335BswitchBsweepingBsuspectsBsorsBsmokersBsmellBsingleBsfiaBsentB
schedulingBsaysBs1s2BroutedBrichmondBretrieveB
respondingB	requestedBrepairsBrefusedBratedBranBr601Br50Br402Br401Br2Br101BquickBpossiblyBpmBplannerBphoneB
performingB	passengerBpassedBoverBoutofserviceBotherB	operatorsBopenedBnumbersBnotesBmlineB
mitigationBminimalBmeetBmdrxBmakingBm903BlodgerBleavingBleadBklineB
keyimposedBk202Bk20Bk103Bk102BjustBintoxicatedBinterferenceBinsteadBinspectBifBholdsBherBheadingBhaveBhalfBhadBgunBgroundsBfireBfdBfareBevaderB	equipmentBendBemployeeBdwellBdrunkBdroppingB
dispatchesB
dispatchedBdepartedBcyclesBcutoutBcrewBcranksBcrankedBcoverBconfigurationBconeBcodesBclampsBclampedBcellBcategoryBcanBcabBc88Bc35Bc302BbetweenBbehindBbecauseBbacktrackersB	availableBaskedBapologyBanotherBannouncementsBamBagainstBafBactivateBaboutBa90Ba602Ba502Ba202Ba2Ba102Ba10B599B499B399B397B331B329B2carsB20B16B15sB11B	yesterdayBy102By101By05fdBxBwouldBworkedBwontBwoBwlineblanketBwithoutBwhyBwhileBwentBweeklyBweedBweBw30Bw10BvpiBveryBverifiedBusingBusedBupdatedBunawareBturnsBturningBtroubleshootsBtrippedBtransferB	trains581B	trains568BtrailingBtowersBtosBtmsBtmBthreateningBthemBtheirB	terminateBtc5BtardyBtakenBtakeBt950Bt917Bt905Bt767Bt519t919Bt5195Bt501mBt5018Bt44335Bt3675Bt3657Bt36110Bt3379Bt3339Bt331931Bt329959Bt3299Bt32135B	t2379t329Bt2255Bt2235Bt11310Bt111BswitchedBsweepsBsweepB
suspiciousBsureBsuperintendentBsummaryB
structuresBstillBstayedBstatusBstationsBstartingBstartedBstandBstBsparesBsouthBsoonBsomeBsolidBsoBsmokeBsmallBslumpedBslotBsinceBsheBsfoboundBseriesBseparateBselfBseizureBsearchB
scramspareBschexnayderBsafeBs55eotBrunsBroutesBromeo12BrmBrlineBrideBreverseBreturnBresumeBresponseB	respondedBrepositioningBrepositionedB
reportedlyBremainBreliefBreleaseBreconfiguringBreasonBreadbackBratingBrainBradioB
r65xl27mphBr65xlBr602Br60Br501Br4Br301Br202Br201BqolBputBpullBpucBprotectionsBprotectBproperBpropBprogressBproblemBpriorityBpriorBpresentBpracticeBpostBpositionBportableBpoorB	personnelB	performedBpdbcoBpanelBownBoutsideBoutoundBoutboundBoneBoliverBofftrackB
offboardedBoccBobservedBnoteB	normalizeBnightBneedingBneededBmxcxBmwBmovedBmoreB
monitoringBmodifiedBmlBmitigationpassengerBmissedB	misroutedBmileBmidBmedicsBmedic10BmayB	marijuanaBmanBmaleBmakeBmaintainBm97Bm93xlBm87xlBm85xlBm801Bm80Bm501Bm50B
m45johnsonBm45Bm40B
m3matthewsBm30Bm20Bm17Bm15Bm1BmBluckBlookingBlongBlogBlockedB	locationsBlocateBloadsBlittleBlikeBlegacyBlastBl25l13Bl102Bl1BkeyingBkeepingBk301Bk25cdBk201Bk101BitsBissuestaffingBinvestigatingBinverterBintooutB	installedB	inspectedBinboundBifoBhourBhomelessBhomeBhighBheardBhazardBhavingB	handcuffsBgrierBgraveBgrateBgraffitiBgotBgoingBgivenBfurtherBfullBfroBformalBfoBfirstBfipBfineBfillBfileBfenceBfemaleBfellBfeelsBfailedBe’sB	extremelyBextendBexitB	excellentB
eventuallyBetaBerrorBerraticBeorBendsBendedB
encampmentBenB
electricalBeachBd’sBdumpedBdrugBdraggingBdowntownBdownBdoingBdoesBdispatchBdidntBdidBdeniedB	deliveredBdayBdailyBc’sBcyclingBcto1BcrowdsB
coverboardBcoupleBcountBcouldB
controllerB	consumingBconsoleB
complianceB
completingB	completedBcommunicationsBcolvinBclosestBclosedBcloseBclineBclearsB	clearanceBckBcircuitBchuehBcheckingBcharlie3BchargingBchargeBchangingBchangeBcdBcausingBcameBcalledBcallBcabledBc89Bc85Bc801Bc75Bc67Bc602Bc601Bc501Bc4014Bc3grierBc377Bc35abBc301Bc30Bc3Bc2m1Bc252Bc2503Bc21scottlylesBc201Bc1851Bc1633Bc1607Bc1595Bc1519Bc1270Bc102Bc10Bc1BcBb’sBbreakerBboothBboltsBboltedBboardupBboardingBboardedBblockingBblockedBbeingBbeforeBbarriersBbalancedBbadBa’sB
authorizedB
assistanceBassistBassaultBarrivignBarrivedBaroundBarmedBapproachBalsoBalsBalpha99Balpha68BallowBalarmBalBairportBaheadB
aggressiveBactivtyB	activatedBacknowledgedBableBabBa901Ba801Ba78Ba75xlBa60Ba501Ba44Ba302Ba301Ba2dangerfieldBa20a04Ba10abBa1B93B90B79B76B65B64B63B610B5bs1cB55B51B48B45B442B441B3bB367B35B332B323B321B2cB2901B287B27mphB275B25B24B220B2056B201B1stB1carB172B	16minutesB15765683B156154B154156B14B12carB	11minutesB10upB10carB108B105B101B100s
?<
Const_5Const*
_output_shapes	
:?*
dtype0	*?<
value?<B?<	?"?<                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
StatefulPartitionedCall_2StatefulPartitionedCallStatefulPartitionedCallConst_4Const_5*
Tin
2	*
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_45563
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_45569
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_2
?
3None_lookup_table_export_values/LookupTableExportV2LookupTableExportV2StatefulPartitionedCall_1*
Tkeys0*
Tvalues0	*,
_class"
 loc:@StatefulPartitionedCall_1*
_output_shapes

::
?
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer

signatures
#	_self_saveable_object_factories

regularization_losses
trainable_variables
	variables
	keras_api
M
_index_lookup_layer
#_self_saveable_object_factories
	keras_api
?

embeddings
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
 trainable_variables
!	variables
"	keras_api
?

#kernel
$bias
#%_self_saveable_object_factories
&regularization_losses
'trainable_variables
(	variables
)	keras_api
?

*kernel
+bias
#,_self_saveable_object_factories
-regularization_losses
.trainable_variables
/	variables
0	keras_api
 
 
 
 
1
0
1
2
#3
$4
*5
+6
1
1
2
3
#4
$5
*6
+7
?

1layers

regularization_losses
trainable_variables
2layer_metrics
3non_trainable_variables
	variables
4layer_regularization_losses
5metrics
X
6lookup_table
7token_counts
#8_self_saveable_object_factories
9	keras_api
 
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
?

:layers
regularization_losses
trainable_variables
;layer_metrics
<non_trainable_variables
	variables
=layer_regularization_losses
>metrics
 
 
 
 
?

?layers
regularization_losses
trainable_variables
@layer_metrics
Anon_trainable_variables
	variables
Blayer_regularization_losses
Cmetrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?

Dlayers
regularization_losses
 trainable_variables
Elayer_metrics
Fnon_trainable_variables
!	variables
Glayer_regularization_losses
Hmetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

#0
$1

#0
$1
?

Ilayers
&regularization_losses
'trainable_variables
Jlayer_metrics
Knon_trainable_variables
(	variables
Llayer_regularization_losses
Mmetrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

*0
+1

*0
+1
?

Nlayers
-regularization_losses
.trainable_variables
Olayer_metrics
Pnon_trainable_variables
/	variables
Qlayer_regularization_losses
Rmetrics
*
0
1
2
3
4
5
 
 
 

S0
T1

U_initializer
RP
tableGlayer_with_weights-0/_index_lookup_layer/token_counts/.ATTRIBUTES/table
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
	Vtotal
	Wcount
X	variables
Y	keras_api
D
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

X	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

]	variables
?
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_3StatefulPartitionedCall(serving_default_text_vectorization_inputStatefulPartitionedCallConstConst_1Const_2embedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_44653
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp3None_lookup_table_export_values/LookupTableExportV25None_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst_6*
Tin
2	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_45647
?
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasStatefulPartitionedCall_1totalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_45693??
?
?
%__inference_dense_layer_call_fn_45399

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_442262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_41932
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?&
?
__inference__traced_save_45647
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop>
:savev2_none_lookup_table_export_values_lookuptableexportv2@
<savev2_none_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-0/_index_lookup_layer/token_counts/.ATTRIBUTES/table-keysBNlayer_with_weights-0/_index_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop:savev2_none_lookup_table_export_values_lookuptableexportv2<savev2_none_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*b
_input_shapesQ
O: :	?N::::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
::	

_output_shapes
::
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
: 
?
,
__inference__destroyer_45493
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_454892
PartitionedCallP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
٤
}
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45390

inputs
inputs_1	
identity?
&RaggedReduceMean/RaggedReduceSum/ShapeShapeinputs_1*
T0	*
_output_shapes
:2(
&RaggedReduceMean/RaggedReduceSum/Shape?
4RaggedReduceMean/RaggedReduceSum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4RaggedReduceMean/RaggedReduceSum/strided_slice/stack?
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1?
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2?
.RaggedReduceMean/RaggedReduceSum/strided_sliceStridedSlice/RaggedReduceMean/RaggedReduceSum/Shape:output:0=RaggedReduceMean/RaggedReduceSum/strided_slice/stack:output:0?RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1:output:0?RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.RaggedReduceMean/RaggedReduceSum/strided_slice?
&RaggedReduceMean/RaggedReduceSum/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&RaggedReduceMean/RaggedReduceSum/sub/y?
$RaggedReduceMean/RaggedReduceSum/subSub7RaggedReduceMean/RaggedReduceSum/strided_slice:output:0/RaggedReduceMean/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2&
$RaggedReduceMean/RaggedReduceSum/sub?
MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
GRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceinputs_1VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2I
GRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceinputs_1XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2K
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
=RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubPRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2?
=RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeinputs_1*
T0	*
_output_shapes
:*
out_type0	2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
ARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2C
ARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubRRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2F
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0CRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCastARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
NRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2P
NRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackWRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToIRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2N
LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxURaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2F
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
HRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumSRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2J
HRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2^
\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsURaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2Z
XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastaRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
KRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2M
KRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileTRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
aRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2c
aRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0jRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2]
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2f
dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProddRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0mRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Pack[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
YRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2[
YRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0bRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2V
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
WRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeVRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2Y
WRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere`RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueeze[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
3RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSuminputs_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0(RaggedReduceMean/RaggedReduceSum/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????25
3RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumz
 RaggedReduceMean/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2"
 RaggedReduceMean/ones_like/Shape?
 RaggedReduceMean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 RaggedReduceMean/ones_like/Const?
RaggedReduceMean/ones_likeFill)RaggedReduceMean/ones_like/Shape:output:0)RaggedReduceMean/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
RaggedReduceMean/ones_like?
(RaggedReduceMean/RaggedReduceSum_1/ShapeShapeinputs_1*
T0	*
_output_shapes
:2*
(RaggedReduceMean/RaggedReduceSum_1/Shape?
6RaggedReduceMean/RaggedReduceSum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack?
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1?
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2?
0RaggedReduceMean/RaggedReduceSum_1/strided_sliceStridedSlice1RaggedReduceMean/RaggedReduceSum_1/Shape:output:0?RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack:output:0ARaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1:output:0ARaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0RaggedReduceMean/RaggedReduceSum_1/strided_slice?
(RaggedReduceMean/RaggedReduceSum_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(RaggedReduceMean/RaggedReduceSum_1/sub/y?
&RaggedReduceMean/RaggedReduceSum_1/subSub9RaggedReduceMean/RaggedReduceSum_1/strided_slice:output:01RaggedReduceMean/RaggedReduceSum_1/sub/y:output:0*
T0*
_output_shapes
: 2(
&RaggedReduceMean/RaggedReduceSum_1/sub?
ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
IRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceinputs_1XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2K
IRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceinputs_1ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2M
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1?
?RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/subSubRRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice:output:0TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2A
?RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/ShapeShapeinputs_1*
T0	*
_output_shapes
:*
out_type0	2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2M
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2?
CRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
CRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1SubTRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta?
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/CastCastPRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1CastPRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/rangeRangeJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast:y:0ERaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1:z:0LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/CastCastCRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ShapeShapeJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape?
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
PRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceQRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2R
PRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackYRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToKRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2P
NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const?
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxWRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max?
LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
JRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumURaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2L
JRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2`
^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsWRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2\
ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastcRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
MRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2O
MRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPack\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/TileTileVRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
cRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2e
cRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0lRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2_
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2h
fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdfRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0oRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Pack]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
[RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0dRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
YRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeXRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2[
YRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWherebRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueeze]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
5RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSumUnsortedSegmentSum#RaggedReduceMean/ones_like:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0*RaggedReduceMean/RaggedReduceSum_1/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????27
5RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum?
RaggedReduceMean/truedivRealDiv<RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum:output:0>RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:?????????2
RaggedReduceMean/truedivp
IdentityIdentityRaggedReduceMean/truediv:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
U
(__inference_restored_function_body_45588
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference__creator_421602
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
'__inference_dense_2_layer_call_fn_45439

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_442602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_44243

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_44556
text_vectorization_inputQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	"
embedding_44535:	?N
dense_44540:
dense_44542:
dense_1_44545:
dense_1_44547:
dense_2_44550:
dense_2_44552:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall4text_vectorization/string_lookup_3/Identity:output:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0embedding_44535*
Tin
2		*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":?????????:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_440402#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_442132*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_44540dense_44542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_442262
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44545dense_1_44547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_442432!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_44550dense_2_44552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_442602!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_global_average_pooling1d_layer_call_fn_45209

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_439642
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
8
(__inference_restored_function_body_45509
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__initializer_426902
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
*__inference_sequential_layer_call_fn_44707

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_444362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_45450

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__initializer_42690
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_418157
3key_value_init5760_lookuptableimportv2_table_handle/
+key_value_init5760_lookuptableimportv2_keys1
-key_value_init5760_lookuptableimportv2_values	
identity??&key_value_init5760/LookupTableImportV2?
&key_value_init5760/LookupTableImportV2LookupTableImportV23key_value_init5760_lookuptableimportv2_table_handle+key_value_init5760_lookuptableimportv2_keys-key_value_init5760_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init5760/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constw
NoOpNoOp'^key_value_init5760/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOpX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init5760/LookupTableImportV2&key_value_init5760/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_44267

inputsQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	"
embedding_44041:	?N
dense_44227:
dense_44229:
dense_1_44244:
dense_1_44246:
dense_2_44261:
dense_2_44263:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2|
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall4text_vectorization/string_lookup_3/Identity:output:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0embedding_44041*
Tin
2		*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":?????????:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_440402#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_442132*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_44227dense_44229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_442262
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44244dense_1_44246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_442432!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_44261dense_2_44263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_442602!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_45204

inputs	
inputs_1	A
.embedding_lookup_ragged_embedding_lookup_45197:	?N
identity

identity_1	??(embedding_lookup_ragged/embedding_lookup?
(embedding_lookup_ragged/embedding_lookupResourceGather.embedding_lookup_ragged_embedding_lookup_45197inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@embedding_lookup_ragged/embedding_lookup/45197*'
_output_shapes
:?????????*
dtype02*
(embedding_lookup_ragged/embedding_lookup?
1embedding_lookup_ragged/embedding_lookup/IdentityIdentity1embedding_lookup_ragged/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding_lookup_ragged/embedding_lookup/45197*'
_output_shapes
:?????????23
1embedding_lookup_ragged/embedding_lookup/Identity?
3embedding_lookup_ragged/embedding_lookup/Identity_1Identity:embedding_lookup_ragged/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????25
3embedding_lookup_ragged/embedding_lookup/Identity_1?
IdentityIdentity<embedding_lookup_ragged/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityc

Identity_1Identityinputs_1^NoOp*
T0	*#
_output_shapes
:?????????2

Identity_1y
NoOpNoOp)^embedding_lookup_ragged/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????:?????????: 2T
(embedding_lookup_ragged/embedding_lookup(embedding_lookup_ragged/embedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_44292
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_442672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
[
(__inference_restored_function_body_45593
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference__creator_422232
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_45430

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45221

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
r
__inference_<lambda>_45563
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_454722
StatefulPartitionedCallS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?22
StatefulPartitionedCallStatefulPartitionedCall:!

_output_shapes	
:?:!

_output_shapes	
:?
?	
?
)__inference_embedding_layer_call_fn_45193

inputs	
inputs_1	
unknown:	?N
identity

identity_1	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown*
Tin
2		*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":?????????:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_440402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity{

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*#
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
*
__inference_<lambda>_45569
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_455092
PartitionedCallS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
*__inference_sequential_layer_call_fn_44488
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_444362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_44945

inputsQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	K
8embedding_embedding_lookup_ragged_embedding_lookup_44754:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?2embedding/embedding_lookup_ragged/embedding_lookup?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2|
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
2embedding/embedding_lookup_ragged/embedding_lookupResourceGather8embedding_embedding_lookup_ragged_embedding_lookup_447544text_vectorization/string_lookup_3/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*K
_classA
?=loc:@embedding/embedding_lookup_ragged/embedding_lookup/44754*'
_output_shapes
:?????????*
dtype024
2embedding/embedding_lookup_ragged/embedding_lookup?
;embedding/embedding_lookup_ragged/embedding_lookup/IdentityIdentity;embedding/embedding_lookup_ragged/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@embedding/embedding_lookup_ragged/embedding_lookup/44754*'
_output_shapes
:?????????2=
;embedding/embedding_lookup_ragged/embedding_lookup/Identity?
=embedding/embedding_lookup_ragged/embedding_lookup/Identity_1IdentityDembedding/embedding_lookup_ragged/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2?
=embedding/embedding_lookup_ragged/embedding_lookup/Identity_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape?
Mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2?
Gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_sliceStridedSliceHglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape:output:0Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y?
=global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/subSubPglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice:output:0Hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2?
=global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub?
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2h
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2X
Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1Subkglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastgglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2_
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1Castgglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCastZglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlicehglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2i
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackpglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTobglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2g
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2_
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2e
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumlglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2w
uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastzglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPacksglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTilemglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2|
zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Packtglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2t
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0{global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshapeoglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezetglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumFembedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2N
Lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum?
9global_average_pooling1d/RaggedReduceMean/ones_like/ShapeShapeFembedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2;
9global_average_pooling1d/RaggedReduceMean/ones_like/Shape?
9global_average_pooling1d/RaggedReduceMean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2;
9global_average_pooling1d/RaggedReduceMean/ones_like/Const?
3global_average_pooling1d/RaggedReduceMean/ones_likeFillBglobal_average_pooling1d/RaggedReduceMean/ones_like/Shape:output:0Bglobal_average_pooling1d/RaggedReduceMean/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????25
3global_average_pooling1d/RaggedReduceMean/ones_like?
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2C
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack?
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1?
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2?
Iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_sliceStridedSliceJglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack:output:0Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1:output:0Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice?
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2C
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/subSubRglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice:output:0Jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y:output:0*
T0*
_output_shapes
: 2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_sliceStridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/subSubkglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice:output:0mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlicecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2?
\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2^
\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1Submglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/CastCastiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1Castiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/rangeRangecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast:y:0^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1:z:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/CastCast\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ShapeShapecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlicejglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2k
iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackrglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTodglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2i
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxpglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max?
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2g
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2e
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2y
wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimspglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2u
sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLesswglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimscglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2h
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/TileTileoglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Packvglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshapeqglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2t
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere{global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezevglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSumUnsortedSegmentSum<global_average_pooling1d/RaggedReduceMean/ones_like:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2P
Nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum?
1global_average_pooling1d/RaggedReduceMean/truedivRealDivUglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum:output:0Wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:?????????23
1global_average_pooling1d/RaggedReduceMean/truediv?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul5global_average_pooling1d/RaggedReduceMean/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmaxt
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp3^embedding/embedding_lookup_ragged/embedding_lookupA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2h
2embedding/embedding_lookup_ragged/embedding_lookup2embedding/embedding_lookup_ragged/embedding_lookup2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
U
(__inference_restored_function_body_45457
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference__creator_421602
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
.
__inference__initializer_45513
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_455092
PartitionedCallP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_43964

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
٤
}
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_44213

inputs
inputs_1	
identity?
&RaggedReduceMean/RaggedReduceSum/ShapeShapeinputs_1*
T0	*
_output_shapes
:2(
&RaggedReduceMean/RaggedReduceSum/Shape?
4RaggedReduceMean/RaggedReduceSum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4RaggedReduceMean/RaggedReduceSum/strided_slice/stack?
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1?
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2?
.RaggedReduceMean/RaggedReduceSum/strided_sliceStridedSlice/RaggedReduceMean/RaggedReduceSum/Shape:output:0=RaggedReduceMean/RaggedReduceSum/strided_slice/stack:output:0?RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1:output:0?RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.RaggedReduceMean/RaggedReduceSum/strided_slice?
&RaggedReduceMean/RaggedReduceSum/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&RaggedReduceMean/RaggedReduceSum/sub/y?
$RaggedReduceMean/RaggedReduceSum/subSub7RaggedReduceMean/RaggedReduceSum/strided_slice:output:0/RaggedReduceMean/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2&
$RaggedReduceMean/RaggedReduceSum/sub?
MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
GRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceinputs_1VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2I
GRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceinputs_1XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2K
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
=RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubPRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2?
=RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeinputs_1*
T0	*
_output_shapes
:*
out_type0	2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2K
IRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
ARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2C
ARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubRRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2F
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0CRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2A
?RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCastARaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
NRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2P
NRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackWRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToIRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2N
LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2H
FRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxURaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2F
DRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
HRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumSRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0MRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2J
HRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2^
\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsURaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2Z
XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastaRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
KRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsHRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2M
KRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2S
QRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackZRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0LRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2Q
ORaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileTRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0XRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2G
ERaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
aRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2c
aRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice\RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0jRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2]
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2f
dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProddRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0mRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2T
RRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2e
cRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0lRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0nRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Pack[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
YRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2[
YRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0bRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2V
TRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeNRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2_
]RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
WRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeVRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0fRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2Y
WRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere`RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2U
SRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueeze[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2W
URaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0^RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0dRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2X
VRaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
3RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSuminputs_RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0(RaggedReduceMean/RaggedReduceSum/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????25
3RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumz
 RaggedReduceMean/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2"
 RaggedReduceMean/ones_like/Shape?
 RaggedReduceMean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 RaggedReduceMean/ones_like/Const?
RaggedReduceMean/ones_likeFill)RaggedReduceMean/ones_like/Shape:output:0)RaggedReduceMean/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
RaggedReduceMean/ones_like?
(RaggedReduceMean/RaggedReduceSum_1/ShapeShapeinputs_1*
T0	*
_output_shapes
:2*
(RaggedReduceMean/RaggedReduceSum_1/Shape?
6RaggedReduceMean/RaggedReduceSum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack?
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1?
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2?
0RaggedReduceMean/RaggedReduceSum_1/strided_sliceStridedSlice1RaggedReduceMean/RaggedReduceSum_1/Shape:output:0?RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack:output:0ARaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1:output:0ARaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0RaggedReduceMean/RaggedReduceSum_1/strided_slice?
(RaggedReduceMean/RaggedReduceSum_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(RaggedReduceMean/RaggedReduceSum_1/sub/y?
&RaggedReduceMean/RaggedReduceSum_1/subSub9RaggedReduceMean/RaggedReduceSum_1/strided_slice:output:01RaggedReduceMean/RaggedReduceSum_1/sub/y:output:0*
T0*
_output_shapes
: 2(
&RaggedReduceMean/RaggedReduceSum_1/sub?
ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
IRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceinputs_1XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2K
IRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceinputs_1ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2M
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1?
?RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/subSubRRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice:output:0TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2A
?RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/ShapeShapeinputs_1*
T0	*
_output_shapes
:*
out_type0	2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2M
KRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2?
CRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
CRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1SubTRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta?
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/CastCastPRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1CastPRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1?
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/rangeRangeJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast:y:0ERaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1:z:0LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2C
ARaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/CastCastCRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ShapeShapeJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape?
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
PRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceQRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2R
PRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackYRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToKRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2P
NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
HRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const?
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxWRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2H
FRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max?
LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
JRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumURaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0ORaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2L
JRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2`
^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsWRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2\
ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastcRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
MRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsJRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2O
MRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2U
SRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPack\RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0NRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2S
QRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/TileTileVRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0ZRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2I
GRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
cRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2e
cRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice^RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0lRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2_
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2h
fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdfRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0oRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2V
TRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
eRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0nRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0pRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Pack]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
[RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0dRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapePRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2a
_RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
YRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeXRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0hRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2[
YRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWherebRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2W
URaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueeze]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2Y
WRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0`RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0fRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2Z
XRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
5RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSumUnsortedSegmentSum#RaggedReduceMean/ones_like:output:0aRaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0*RaggedReduceMean/RaggedReduceSum_1/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????27
5RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum?
RaggedReduceMean/truedivRealDiv<RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum:output:0>RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:?????????2
RaggedReduceMean/truedivp
IdentityIdentityRaggedReduceMean/truediv:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_44040

inputs	
inputs_1	A
.embedding_lookup_ragged_embedding_lookup_44033:	?N
identity

identity_1	??(embedding_lookup_ragged/embedding_lookup?
(embedding_lookup_ragged/embedding_lookupResourceGather.embedding_lookup_ragged_embedding_lookup_44033inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@embedding_lookup_ragged/embedding_lookup/44033*'
_output_shapes
:?????????*
dtype02*
(embedding_lookup_ragged/embedding_lookup?
1embedding_lookup_ragged/embedding_lookup/IdentityIdentity1embedding_lookup_ragged/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding_lookup_ragged/embedding_lookup/44033*'
_output_shapes
:?????????23
1embedding_lookup_ragged/embedding_lookup/Identity?
3embedding_lookup_ragged/embedding_lookup/Identity_1Identity:embedding_lookup_ragged/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????25
3embedding_lookup_ragged/embedding_lookup/Identity_1?
IdentityIdentity<embedding_lookup_ragged/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityc

Identity_1Identityinputs_1^NoOp*
T0	*#
_output_shapes
:?????????2

Identity_1y
NoOpNoOp)^embedding_lookup_ragged/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????:?????????: 2T
(embedding_lookup_ragged/embedding_lookup(embedding_lookup_ragged/embedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
 __inference__wrapped_model_43954
text_vectorization_input\
Xsequential_text_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle]
Ysequential_text_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	9
5sequential_text_vectorization_string_lookup_3_equal_y<
8sequential_text_vectorization_string_lookup_3_selectv2_t	V
Csequential_embedding_embedding_lookup_ragged_embedding_lookup_43763:	?NA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?=sequential/embedding/embedding_lookup_ragged/embedding_lookup?Ksequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
)sequential/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:?????????2+
)sequential/text_vectorization/StringLower?
0sequential/text_vectorization/StaticRegexReplaceStaticRegexReplace2sequential/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 22
0sequential/text_vectorization/StaticRegexReplace?
/sequential/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 21
/sequential/text_vectorization/StringSplit/Const?
7sequential/text_vectorization/StringSplit/StringSplitV2StringSplitV29sequential/text_vectorization/StaticRegexReplace:output:08sequential/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:29
7sequential/text_vectorization/StringSplit/StringSplitV2?
=sequential/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential/text_vectorization/StringSplit/strided_slice/stack?
?sequential/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2A
?sequential/text_vectorization/StringSplit/strided_slice/stack_1?
?sequential/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential/text_vectorization/StringSplit/strided_slice/stack_2?
7sequential/text_vectorization/StringSplit/strided_sliceStridedSliceAsequential/text_vectorization/StringSplit/StringSplitV2:indices:0Fsequential/text_vectorization/StringSplit/strided_slice/stack:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_1:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask29
7sequential/text_vectorization/StringSplit/strided_slice?
?sequential/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/text_vectorization/StringSplit/strided_slice_1/stack?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_1?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_2?
9sequential/text_vectorization/StringSplit/strided_slice_1StridedSlice?sequential/text_vectorization/StringSplit/StringSplitV2:shape:0Hsequential/text_vectorization/StringSplit/strided_slice_1/stack:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2;
9sequential/text_vectorization/StringSplit/strided_slice_1?
`sequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast@sequential/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2b
`sequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastBsequential/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2d
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapedsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2l
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2l
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2k
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
nsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2p
nsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterrsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0wsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2n
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastpsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2k
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2n
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2j
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2l
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2qsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2j
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulmsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2j
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2n
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2n
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2n
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
msequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2o
msequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2i
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumtsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2d
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2m
ksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2i
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2tsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2d
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Ksequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_text_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle@sequential/text_vectorization/StringSplit/StringSplitV2:values:0Ysequential_text_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2M
Ksequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
3sequential/text_vectorization/string_lookup_3/EqualEqual@sequential/text_vectorization/StringSplit/StringSplitV2:values:05sequential_text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????25
3sequential/text_vectorization/string_lookup_3/Equal?
6sequential/text_vectorization/string_lookup_3/SelectV2SelectV27sequential/text_vectorization/string_lookup_3/Equal:z:08sequential_text_vectorization_string_lookup_3_selectv2_tTsequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????28
6sequential/text_vectorization/string_lookup_3/SelectV2?
6sequential/text_vectorization/string_lookup_3/IdentityIdentity?sequential/text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????28
6sequential/text_vectorization/string_lookup_3/Identity?
=sequential/embedding/embedding_lookup_ragged/embedding_lookupResourceGatherCsequential_embedding_embedding_lookup_ragged_embedding_lookup_43763?sequential/text_vectorization/string_lookup_3/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*V
_classL
JHloc:@sequential/embedding/embedding_lookup_ragged/embedding_lookup/43763*'
_output_shapes
:?????????*
dtype02?
=sequential/embedding/embedding_lookup_ragged/embedding_lookup?
Fsequential/embedding/embedding_lookup_ragged/embedding_lookup/IdentityIdentityFsequential/embedding/embedding_lookup_ragged/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@sequential/embedding/embedding_lookup_ragged/embedding_lookup/43763*'
_output_shapes
:?????????2H
Fsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity?
Hsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity_1IdentityOsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2J
Hsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity_1?
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/ShapeShapeksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2L
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape?
Xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Z
Xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack?
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1?
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2?
Rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_sliceStridedSliceSsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape:output:0asequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack:output:0csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1:output:0csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2T
Rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice?
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y?
Hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/subSub[sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice:output:0Ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2J
Hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub?
qsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2s
qsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2m
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2o
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
asequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubtsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2c
asequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2e
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSlicelsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2o
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2g
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1Subvsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2e
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2k
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2k
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastrsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2j
hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1Castrsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2l
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangelsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0gsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2e
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCastesequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2k
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapelsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2l
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2|
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlicessequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2t
rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePack{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2x
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTomsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2r
psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2l
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2j
hsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2p
nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumwsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0qsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2n
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2~
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2x
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2x
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimslsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2q
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPack~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTilexsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2k
isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapersequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2x
vsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapersequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapersequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Packsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
}sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
}sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapersequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshapezsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezesequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2|
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumQsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2Y
Wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum?
Dsequential/global_average_pooling1d/RaggedReduceMean/ones_like/ShapeShapeQsequential/embedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2F
Dsequential/global_average_pooling1d/RaggedReduceMean/ones_like/Shape?
Dsequential/global_average_pooling1d/RaggedReduceMean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2F
Dsequential/global_average_pooling1d/RaggedReduceMean/ones_like/Const?
>sequential/global_average_pooling1d/RaggedReduceMean/ones_likeFillMsequential/global_average_pooling1d/RaggedReduceMean/ones_like/Shape:output:0Msequential/global_average_pooling1d/RaggedReduceMean/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2@
>sequential/global_average_pooling1d/RaggedReduceMean/ones_like?
Lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/ShapeShapeksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2N
Lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape?
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2\
Zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack?
\sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1?
\sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2?
Tsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_sliceStridedSliceUsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape:output:0csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack:output:0esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1:output:0esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
Tsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice?
Lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y?
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/subSub]sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice:output:0Usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y:output:0*
T0*
_output_shapes
: 2L
Jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub?
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2u
ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2o
msequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2q
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1?
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/subSubvsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice:output:0xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2e
csequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub?
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/ShapeShapeksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2g
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlicensequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2q
osequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2?
gsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2i
gsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y?
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1Subxsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2g
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1?
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2m
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start?
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2m
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta?
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/CastCasttsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2l
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast?
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1Casttsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2n
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1?
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/rangeRangensequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast:y:0isequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1:z:0psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2g
esequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range?
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/CastCastgsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2m
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast?
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ShapeShapensequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2n
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape?
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2|
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2~
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2~
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
tsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceusequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2v
tsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePack}sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToosequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2t
rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2n
lsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const?
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaxMax{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2l
jsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max?
psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2r
psequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0ssequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2p
nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRange?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDims{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2?
~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLess?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
qsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsnsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2s
qsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2y
wsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPack?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0rsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2w
usequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/TileTilezsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0~sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2m
ksequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapetsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2z
xsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapetsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapetsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlice?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Pack?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2|
zsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapetsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
}sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshape|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2
}sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2{
ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueeze?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2}
{sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2~
|sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSumUnsortedSegmentSumGsequential/global_average_pooling1d/RaggedReduceMean/ones_like:output:0?sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Nsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2[
Ysequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum?
<sequential/global_average_pooling1d/RaggedReduceMean/truedivRealDiv`sequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum:output:0bsequential/global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:?????????2>
<sequential/global_average_pooling1d/RaggedReduceMean/truediv?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul@sequential/global_average_pooling1d/RaggedReduceMean/truediv:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Relu?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Relu?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Softmax
IdentityIdentity$sequential/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp>^sequential/embedding/embedding_lookup_ragged/embedding_lookupL^sequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2~
=sequential/embedding/embedding_lookup_ragged/embedding_lookup=sequential/embedding/embedding_lookup_ragged/embedding_lookup2?
Ksequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2Ksequential/text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_45524
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_455202
PartitionedCallP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
G
__inference__creator_45460
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_454572
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
*__inference_sequential_layer_call_fn_44680

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_442672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_45183

inputsQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	K
8embedding_embedding_lookup_ragged_embedding_lookup_44992:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?2embedding/embedding_lookup_ragged/embedding_lookup?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2|
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
2embedding/embedding_lookup_ragged/embedding_lookupResourceGather8embedding_embedding_lookup_ragged_embedding_lookup_449924text_vectorization/string_lookup_3/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*K
_classA
?=loc:@embedding/embedding_lookup_ragged/embedding_lookup/44992*'
_output_shapes
:?????????*
dtype024
2embedding/embedding_lookup_ragged/embedding_lookup?
;embedding/embedding_lookup_ragged/embedding_lookup/IdentityIdentity;embedding/embedding_lookup_ragged/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@embedding/embedding_lookup_ragged/embedding_lookup/44992*'
_output_shapes
:?????????2=
;embedding/embedding_lookup_ragged/embedding_lookup/Identity?
=embedding/embedding_lookup_ragged/embedding_lookup/Identity_1IdentityDembedding/embedding_lookup_ragged/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2?
=embedding/embedding_lookup_ragged/embedding_lookup/Identity_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape?
Mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2?
Gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_sliceStridedSliceHglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/Shape:output:0Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_1:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y?
=global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/subSubPglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/strided_slice:output:0Hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2?
=global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub?
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2h
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2X
Vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1Subkglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastgglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2_
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1Castgglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCastZglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlicehglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2i
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackpglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTobglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2g
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2_
]global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2e
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumlglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2w
uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastzglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsaglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPacksglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTilemglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2`
^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2|
zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2m
kglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Packtglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2t
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0{global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapegglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshapeoglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezetglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumFembedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2N
Lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum?
9global_average_pooling1d/RaggedReduceMean/ones_like/ShapeShapeFembedding/embedding_lookup_ragged/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2;
9global_average_pooling1d/RaggedReduceMean/ones_like/Shape?
9global_average_pooling1d/RaggedReduceMean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2;
9global_average_pooling1d/RaggedReduceMean/ones_like/Const?
3global_average_pooling1d/RaggedReduceMean/ones_likeFillBglobal_average_pooling1d/RaggedReduceMean/ones_like/Shape:output:0Bglobal_average_pooling1d/RaggedReduceMean/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????25
3global_average_pooling1d/RaggedReduceMean/ones_like?
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:2C
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape?
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack?
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1?
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2?
Iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_sliceStridedSliceJglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/Shape:output:0Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack:output:0Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_1:output:0Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice?
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2C
Aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/subSubRglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/strided_slice:output:0Jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub/y:output:0*
T0*
_output_shapes
: 2A
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub?
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2j
hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_sliceStridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2d
bglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSlice`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1?
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/subSubkglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice:output:0mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????2Z
Xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/ShapeShape`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*
_output_shapes
:*
out_type0	2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlicecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Shape:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0uglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2f
dglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2?
\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2^
\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1Submglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/CastCastiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1Castiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1?
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/rangeRangecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast:y:0^global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub_1:z:0eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2\
Zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/CastCast\global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ShapeShapecglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlicejglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2k
iglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackrglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTodglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2i
gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
aglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const?
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxpglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2a
_global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max?
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2g
eglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumnglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0hglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2e
cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2y
wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimspglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2u
sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLesswglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimscglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/range:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2h
fglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2n
lglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackuglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0gglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2l
jglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/TileTileoglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0sglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2b
`global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2~
|global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSlicewglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2o
mglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
~global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceyglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1Packvglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2v
tglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0}global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2q
oglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeiglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2z
xglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1Reshapeqglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0?global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2t
rglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhere{global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2p
nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezevglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2r
pglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2x
vglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0yglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0global_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2s
qglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
Nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSumUnsortedSegmentSum<global_average_pooling1d/RaggedReduceMean/ones_like:output:0zglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0Cglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/sub:z:0*
T0*
Tindices0	*'
_output_shapes
:?????????2P
Nglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum?
1global_average_pooling1d/RaggedReduceMean/truedivRealDivUglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum/UnsortedSegmentSum:output:0Wglobal_average_pooling1d/RaggedReduceMean/RaggedReduceSum_1/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:?????????23
1global_average_pooling1d/RaggedReduceMean/truediv?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul5global_average_pooling1d/RaggedReduceMean/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmaxt
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp3^embedding/embedding_lookup_ragged/embedding_lookupA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2h
2embedding/embedding_lookup_ragged/embedding_lookup2embedding/embedding_lookup_ragged/embedding_lookup2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_44624
text_vectorization_inputQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	"
embedding_44603:	?N
dense_44608:
dense_44610:
dense_1_44613:
dense_1_44615:
dense_2_44618:
dense_2_44620:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall4text_vectorization/string_lookup_3/Identity:output:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0embedding_44603*
Tin
2		*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":?????????:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_440402#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_442132*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_44608dense_44610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_442262
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44613dense_1_44615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_442432!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_44618dense_2_44620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_442602!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
8__inference_global_average_pooling1d_layer_call_fn_45215

inputs
inputs_1	
identity?
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_442132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_restored_function_body_45472
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__initializer_418152
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?22
StatefulPartitionedCallStatefulPartitionedCall:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
'__inference_dense_1_layer_call_fn_45419

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_442432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_44260

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
8
(__inference_restored_function_body_45489
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__destroyer_422192
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_44436

inputsQ
Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_3_equal_y1
-text_vectorization_string_lookup_3_selectv2_t	"
embedding_44415:	?N
dense_44420:
dense_44422:
dense_1_44425:
dense_1_44427:
dense_2_44430:
dense_2_44432:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2|
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:?????????2 
text_vectorization/StringLower?
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%text_vectorization/StaticRegexReplace?
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$text_vectorization/StringSplit/Const?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2.
,text_vectorization/StringSplit/StringSplitV2?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2text_vectorization/StringSplit/strided_slice/stack?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4text_vectorization/StringSplit/strided_slice/stack_1?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4text_vectorization/StringSplit/strided_slice/stack_2?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,text_vectorization/StringSplit/strided_slice?
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4text_vectorization/StringSplit/strided_slice_1/stack?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_1?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6text_vectorization/StringSplit/strided_slice_1/stack_2?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.text_vectorization/StringSplit/strided_slice_1?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2W
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2d
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2Y
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2?
(text_vectorization/string_lookup_3/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_3_equal_y*
T0*#
_output_shapes
:?????????2*
(text_vectorization/string_lookup_3/Equal?
+text_vectorization/string_lookup_3/SelectV2SelectV2,text_vectorization/string_lookup_3/Equal:z:0-text_vectorization_string_lookup_3_selectv2_tItext_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/SelectV2?
+text_vectorization/string_lookup_3/IdentityIdentity4text_vectorization/string_lookup_3/SelectV2:output:0*
T0	*#
_output_shapes
:?????????2-
+text_vectorization/string_lookup_3/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall4text_vectorization/string_lookup_3/Identity:output:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0embedding_44415*
Tin
2		*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":?????????:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_440402#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_442132*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_44420dense_44422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_442262
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44425dense_1_44427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_442432!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_44430dense_2_44432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_442602!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_3/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_42219
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
__inference_restore_fn_45551
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
__inference_save_fn_45543
checkpoint_keyD
@none_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??3None_lookup_table_export_values/LookupTableExportV2?
3None_lookup_table_export_values/LookupTableExportV2LookupTableExportV2@none_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::25
3None_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity:None_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity<None_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp4^None_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2j
3None_lookup_table_export_values/LookupTableExportV23None_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
:
__inference__creator_42160
identity??
hash_table?

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*&
shared_name5761_load_41804_42156*
use_node_name_sharing(*
value_dtype0	2

hash_table[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOpc
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
#__inference_signature_wrapper_44653
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_439542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
M
__inference__creator_45503
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_455002
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?9
?
!__inference__traced_restore_45693
file_prefix8
%assignvariableop_embedding_embeddings:	?N1
assignvariableop_1_dense_kernel:+
assignvariableop_2_dense_bias:3
!assignvariableop_3_dense_1_kernel:-
assignvariableop_4_dense_1_bias:3
!assignvariableop_5_dense_2_kernel:-
assignvariableop_6_dense_2_bias:V
Lmutablehashtable_table_restore_lookuptableimportv2_statefulpartitionedcall_1: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: 
identity_12??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-0/_index_lookup_layer/token_counts/.ATTRIBUTES/table-keysBNlayer_with_weights-0/_index_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Lmutablehashtable_table_restore_lookuptableimportv2_statefulpartitionedcall_1RestoreV2:tensors:7RestoreV2:tensors:8*	
Tin0*

Tout0	*,
_class"
 loc:@StatefulPartitionedCall_1*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2k

Identity_7IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11f
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_12?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_12Identity_12:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:2.
,
_class"
 loc:@StatefulPartitionedCall_1
?
F
__inference__creator_42223
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*&
shared_nametable_5667_load_41804*
value_dtype0	2
MutableHashTablea
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOpi
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?T
?
__inference_adapt_step_42208
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
22
IteratorGetNextl
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:?????????2
StringLower?
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2
StaticRegexReplaceg
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
StringSplit/Const?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2
StringSplit/StringSplitV2?
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
StringSplit/strided_slice/stack?
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!StringSplit/strided_slice/stack_1?
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!StringSplit/strided_slice/stack_2?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
StringSplit/strided_slice?
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!StringSplit/strided_slice_1/stack?
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_1?
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_2?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
StringSplit/strided_slice_1?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2D
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2R
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2Q
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2O
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
8
(__inference_restored_function_body_45520
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__destroyer_419322
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
v
__inference__initializer_45482
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
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
GPU 2J 8? *1
f,R*
(__inference_restored_function_body_454722
StatefulPartitionedCallP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?22
StatefulPartitionedCallStatefulPartitionedCall:!

_output_shapes	
:?:!

_output_shapes	
:?
?
[
(__inference_restored_function_body_45500
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference__creator_422232
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?

?
@__inference_dense_layer_call_and_return_conditional_losses_44226

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_45410

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0?????????=
dense_22
StatefulPartitionedCall_3:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer

signatures
#	_self_saveable_object_factories

regularization_losses
trainable_variables
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature"
_tf_keras_sequential
{
_index_lookup_layer
#_self_saveable_object_factories
	keras_api
b_adapt_function"
_tf_keras_layer
?

embeddings
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
 trainable_variables
!	variables
"	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
#%_self_saveable_object_factories
&regularization_losses
'trainable_variables
(	variables
)	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
#,_self_saveable_object_factories
-regularization_losses
.trainable_variables
/	variables
0	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
,
mserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
#3
$4
*5
+6"
trackable_list_wrapper
Q
1
2
3
#4
$5
*6
+7"
trackable_list_wrapper
?

1layers

regularization_losses
trainable_variables
2layer_metrics
3non_trainable_variables
	variables
4layer_regularization_losses
5metrics
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
q
6lookup_table
7token_counts
#8_self_saveable_object_factories
9	keras_api"
_tf_keras_layer
 "
trackable_dict_wrapper
"
_generic_user_object
':%	?N2embedding/embeddings
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

:layers
regularization_losses
trainable_variables
;layer_metrics
<non_trainable_variables
	variables
=layer_regularization_losses
>metrics
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?layers
regularization_losses
trainable_variables
@layer_metrics
Anon_trainable_variables
	variables
Blayer_regularization_losses
Cmetrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Dlayers
regularization_losses
 trainable_variables
Elayer_metrics
Fnon_trainable_variables
!	variables
Glayer_regularization_losses
Hmetrics
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?

Ilayers
&regularization_losses
'trainable_variables
Jlayer_metrics
Knon_trainable_variables
(	variables
Llayer_regularization_losses
Mmetrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?

Nlayers
-regularization_losses
.trainable_variables
Olayer_metrics
Pnon_trainable_variables
/	variables
Qlayer_regularization_losses
Rmetrics
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
R
U_initializer
n_create_resource
o_initialize
p_destroy_resourceR 
O
q_create_resource
r_initialize
s_destroy_resourceR Z
tabletu
 "
trackable_dict_wrapper
"
_generic_user_object
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
N
	Vtotal
	Wcount
X	variables
Y	keras_api"
_tf_keras_metric
^
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api"
_tf_keras_metric
"
_generic_user_object
:  (2total
:  (2count
.
V0
W1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
?2?
*__inference_sequential_layer_call_fn_44292
*__inference_sequential_layer_call_fn_44680
*__inference_sequential_layer_call_fn_44707
*__inference_sequential_layer_call_fn_44488?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_44945
E__inference_sequential_layer_call_and_return_conditional_losses_45183
E__inference_sequential_layer_call_and_return_conditional_losses_44556
E__inference_sequential_layer_call_and_return_conditional_losses_44624?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_43954text_vectorization_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_42208?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_embedding_layer_call_fn_45193?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_45204?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_global_average_pooling1d_layer_call_fn_45209
8__inference_global_average_pooling1d_layer_call_fn_45215?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45221
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45390?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_45399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_45410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_45419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_45430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_45439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_45450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_44653text_vectorization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_45460?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_45482?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_45493?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_45503?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_45513?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_45524?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_45543checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_45551restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_56
__inference__creator_45460?

? 
? "? 6
__inference__creator_45503?

? 
? "? 8
__inference__destroyer_45493?

? 
? "? 8
__inference__destroyer_45524?

? 
? "? ?
__inference__initializer_454826z{?

? 
? "? :
__inference__initializer_45513?

? 
? "? ?
 __inference__wrapped_model_439546vwx#$*+=?:
3?0
.?+
text_vectorization_input?????????
? "1?.
,
dense_2!?
dense_2?????????g
__inference_adapt_step_42208G7y=?:
3?0
.?+?
??????????IteratorSpec
? "
 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_45430\#$/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_45419O#$/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_45450\*+/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_45439O*+/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_45410\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_45399O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_embedding_layer_call_and_return_conditional_losses_45204?X?U
N?K
I?F0?-
???????????????????
?	
`
?	RaggedTensorSpec
? "W?T
M?J4?1
!???????????????????
?
`
?	RaggedTensorSpec
? ?
)__inference_embedding_layer_call_fn_45193?X?U
N?K
I?F0?-
???????????????????
?	
`
?	RaggedTensorSpec
? "M?J4?1
!???????????????????
?
`
?	RaggedTensorSpec?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45221{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45390?`?]
V?S
M?J4?1
!???????????????????
?
`
?	RaggedTensorSpec

 
? "%?"
?
0?????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_45209nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
8__inference_global_average_pooling1d_layer_call_fn_45215|`?]
V?S
M?J4?1
!???????????????????
?
`
?	RaggedTensorSpec

 
? "??????????y
__inference_restore_fn_45551Y7K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_45543?7&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
E__inference_sequential_layer_call_and_return_conditional_losses_44556{6vwx#$*+E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_44624{6vwx#$*+E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_44945i6vwx#$*+3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45183i6vwx#$*+3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_44292n6vwx#$*+E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_44488n6vwx#$*+E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_44680\6vwx#$*+3?0
)?&
?
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_44707\6vwx#$*+3?0
)?&
?
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_44653?6vwx#$*+Y?V
? 
O?L
J
text_vectorization_input.?+
text_vectorization_input?????????"1?.
,
dense_2!?
dense_2?????????