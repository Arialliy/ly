# Table 2 and manual columns

Numeric table uses all frozen objects. Peak inside/outside columns describe coordinates only; an unannotated location is not a proven background object. Neighborhood max is not the actual component output peak. No foreground is added and detection remains at 0.5.

Visual labels apply only to the 72 presampled unique target frames (144 arrangement rows); all other rows stay unreviewed. Labels: suspected_target / background_or_other / uncertain. Numerators and frame/track/clip denominators are separate. Grouping uses each frame's true frozen flags, not the sample anchor group. No population rate is inferred from this finite visual set.

Distances are to the original half-open box; center is not a true target center. The logit and probability argmax difference is a numerical tie/saturation diagnostic; it does not by itself explain a detection failure.
