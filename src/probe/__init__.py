"""Linear and V-probes for analysing the pre-trained iGSM GPT2-RoPE model.

* :mod:`src.probe.extract` -- Stage A: run the *frozen* model, cache hidden states + labels.
* :mod:`src.probe.probe`   -- Stage B: train a tiny ``nn.Linear`` on the cached tensors.
* :mod:`src.probe.vprobe`  -- V-probe: query-conditioned probing through the frozen model.
* :mod:`src.probe.labels`  -- adapter over iGSM's ``Problem.lora_label`` (ground-truth labels).
"""
