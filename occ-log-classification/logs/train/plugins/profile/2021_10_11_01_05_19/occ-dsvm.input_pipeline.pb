  *	?"??~>w@2Q
Iterator::Root::FiniteTake?7?a?A??!9??p.zI@)??+?F<??1???MF@:Preprocessing2?
hIterator::Root::FiniteTake::Prefetch::BatchV2::Shuffle::LegacyParallelInterleaveV2[0]::IgnoreErrors::CSVu?BY???!????3@)u?BY???1????3@:Preprocessing2?
RIterator::Root::FiniteTake::Prefetch::BatchV2::Shuffle::LegacyParallelInterleaveV2?d:tzޱ?!??EI??2@)?d:tzޱ?1??EI??2@:Preprocessing2[
$Iterator::Root::FiniteTake::PrefetchŐ?L?*??!`?]?Bb@)Ő?L?*??1`?]?Bb@:Preprocessing2?
cIterator::Root::FiniteTake::Prefetch::BatchV2::Shuffle::LegacyParallelInterleaveV2[0]::IgnoreErrorsi??r????!?/˺?7@)???QI??1?????@:Preprocessing2d
-Iterator::Root::FiniteTake::Prefetch::BatchV2K?*nܶ?!$q??8@)A??ǘ???1?0&??@:Preprocessing2m
6Iterator::Root::FiniteTake::Prefetch::BatchV2::Shuffle?????!??4?5@)?J?4??1??+[?@:Preprocessing2E
Iterator::Root?}?֤???!0??U?J@)U???N@s?1 ???\8??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.