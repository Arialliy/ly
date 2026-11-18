"""Native S4 score maps from the frozen detector entry point and S0 image-only loader."""
from dataclasses import replace

import torch


@torch.inference_mode()
def predict_full(model, clip, *, query_chunk=1024, velocity_chunk=81):
    """Use larger independent query/velocity batches for evaluation; restore training settings."""
    aggregator = model.video_aggregator
    chunked = hasattr(aggregator, "query_chunk") and hasattr(aggregator, "velocity_chunk")
    if chunked:
        if any(type(size) is not int or size <= 0 for size in (query_chunk, velocity_chunk)):
            raise ValueError("Evaluation chunk sizes must be positive integers")
        old_chunks = aggregator.query_chunk, aggregator.velocity_chunk
        aggregator.query_chunk, aggregator.velocity_chunk = query_chunk, velocity_chunk
    old_training = model.training
    try:
        model.eval()
        prediction = model(clip)
        if prediction.output_frame_id != clip.output_frame_id or prediction.logits.dtype != torch.float32:
            raise ValueError("Prediction ID or FP32 score domain changed")
        return prediction
    finally:
        model.train(old_training)
        if chunked:
            aggregator.query_chunk, aggregator.velocity_chunk = old_chunks


def stream_predictions(model, dataset, center_ids, device, *, query_chunk=1024, velocity_chunk=81):
    """Yield (source/frame ID, native float32 logits[H,W]) in the explicit requested order."""
    lookup = {f"{source}/{number:05d}": i for i, (source, number) in enumerate(dataset.centers)}
    ids = list(center_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty list of unique evaluation centers")
    missing = [key for key in ids if key not in lookup]
    if missing:
        raise ValueError(f"Centers outside this dataset split: {missing}")
    for sample_id in ids:
        clip = dataset.inference_item(lookup[sample_id])
        if clip.output_frame_id != (sample_id,) or not bool(clip.pixel_valid.all()):
            raise ValueError(f"Expected a complete native inference item at {sample_id}")
        clip = replace(clip, frames=clip.frames.to(device), pixel_valid=clip.pixel_valid.to(device),
                       time_offsets=clip.time_offsets.to(device))
        pred = predict_full(model, clip, query_chunk=query_chunk, velocity_chunk=velocity_chunk)
        if pred.logits.shape != (1, 1, *clip.scene_meta[0].original_hw):
            raise ValueError(f"Incorrect native score-map shape at {sample_id}")
        scores = pred.logits[0, 0].cpu()
        if not torch.isfinite(scores).all():
            raise ValueError(f"Nonfinite native score at {sample_id}")
        yield sample_id, scores.numpy()
