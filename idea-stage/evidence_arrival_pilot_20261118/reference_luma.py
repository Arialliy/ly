"""Post hoc late-reveal control: classical reconstruction without the carrier as input."""
import json
import sys

import numpy as np

import witness

sys.path.insert(0, str(witness.V1.parent / "vsr_sampling_x2_20261118"))
from registration import estimate_shifts
from solver import reconstruct


def main():
    manifest = json.loads((witness.V1 / "data" / "MANIFEST.json").read_text())
    records = sorted((r for r in manifest["files"] if r["split"] == "train" and r["frame"] == 32),
                     key=lambda r: r["sequence"])[:8]
    rows = []
    for index, record in enumerate(records):
        background, carrier = witness.scene(record, 0.37 + index * 0.51)
        for polarity in (-1, 1):
            for fact in (-1, 1):
                reference, revealing = witness.make_world(background, carrier, polarity, fact)
                observations = np.stack([reference]*4 + [revealing]) @ witness.RGB_WEIGHTS
                known = np.array([[0., 0.]]*4 + [[0.5, 0.]])
                estimated = estimate_shifts(observations, reference=2)
                row = dict(sequence=record["sequence"], polarity=polarity, fact=fact,
                           estimated_last=estimated[-1].tolist())
                for label, offsets in (("known", known), ("estimated", estimated)):
                    restored = reconstruct(observations, offsets, reference=2)["classic"]
                    # The carrier only decodes the fact for evaluation after reconstruction.
                    odd = np.repeat(restored[::2, 1::2, None], 3, axis=2)
                    score = witness.coeff(odd, carrier) * witness.coeff(reference, carrier)
                    row[label] = witness.class_score(score, fact)
                rows.append(row)
    assert len(rows) == 32
    results = dict(n=len(rows), correct={k: sum(r[k] for r in rows) for k in ("known", "estimated")},
                   cases=rows)
    (witness.HERE / "REFERENCE_LUMA.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results["correct"])


if __name__ == "__main__":
    main()
