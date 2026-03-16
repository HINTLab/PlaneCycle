# `seg_model.py`

`FeatureDecoder` is a wrapper combining with a backbone and a head.

`build_segmentation_decoder`is a function which will return a `FeatureDecoder` type model.

# `seg_head.py`

`ProgressiveUpHead` is a Progressively Upsampling 3D Decoder.

`UpBlock3D` is the smallest upsampling block in`ProgressiveUpHead`.



# `backbone_utils.py`

`_get_backbone_out_indices` can select to concatenate the number of layers of outputs from a backbone，selecting from "last", "four_last", "four_even_intervals".

`ModelWithIntermediateLayers` pack up the backbone into a new class，which allows to straightly use `get_intermediate_layers` to extract the outputs from the backbone.