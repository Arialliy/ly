# Correspondence diagnostic illustrations

Selection was fixed before reading individual outcomes: seed 46 chooses three rows without replacement from sample_id order. Then first competition proxy (delta 0.05) with KLT failure / success is chosen globally and for each dev clip. Duplicates retain every reason. There are at most nine unique figures, below the cap of 12.

Each figure contains untouched native grayscale display [0,255] above a copy with marks, for all eight frames. Crop coordinates are fixed at current anchor +/-32 px. Heatmaps show rich17 normalized correlation at index 3, with a fixed [-1,1] scale.

Yellow: original, undilated GT box. Cyan circle: best path. Green plus: valid KLT; green x: invalid KLT. Magenta diamond: the original alternative when distinct. When the original alternative exactly equals best, display the first different already-generated candidate with 7/7 GT hits at +1 px and gap/frame <=0.05, or omit magenta if none exists. This choice is post hoc using GT for display, not model selection and not a GT-constrained oracle. All original and display paths/reasons are retained in the per-figure JSON. This visual set is not a prevalence estimate or proof of intrinsic ambiguity.

|Sample|Sequence|Selection reason|Figure|
|---|---|---|---|
|78|DJI_0030_9|seed46_uniform_sample_without_replacement|[0078_DJI_0030_9_f079.png](0078_DJI_0030_9_f079.png)|
|409|DJI_0115_2|seed46_uniform_sample_without_replacement|[0409_DJI_0115_2_f111.png](0409_DJI_0115_2_f111.png)|
|40|DJI_0030_9|seed46_uniform_sample_without_replacement|[0040_DJI_0030_9_f047.png](0040_DJI_0030_9_f047.png)|
|340|DJI_0057_1|all_first_competition_klt_failure|[0340_DJI_0057_1_f087.png](0340_DJI_0057_1_f087.png)|
|386|DJI_0115_2|all_first_competition_klt_success|[0386_DJI_0115_2_f023.png](0386_DJI_0115_2_f023.png)|
|635|DJI_0067_2|DJI_0067_2_first_competition_klt_failure|[0635_DJI_0067_2_f023.png](0635_DJI_0067_2_f023.png)|
|680|DJI_0067_2|DJI_0067_2_first_competition_klt_success|[0680_DJI_0067_2_f071.png](0680_DJI_0067_2_f071.png)|
