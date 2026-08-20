# Attribution implementation notes

## DeepJIT and SimCom/Com

Both implementations use the same hierarchical CNN geometry. The code tensor is
`[batch, code_line, code_length]`. `convs_code_line` produces one fixed-width
representation per serialized code-change row; `convs_code_file` then convolves
over those row representations.

Grad-CAM hooks every `convs_code_file` branch. For kernel size `K`, one CAM
position covers input rows `[position, position + K)`. Scores are projected back
through these exact receptive fields and averaged over coverage and kernel
branches. Generic interpolation is intentionally not used.

The target is the class logit (positive-class logit by default), not the
thresholded prediction or post-sigmoid probability. Hooks are installed for one
forward/backward pass and always removed in a `finally` block. The legacy model
forward still returns only probability unless structured attribution output is
explicitly requested; no parameters or checkpoint keys changed.

Rows are the newline-separated units consumed by the historical preprocessing
pipeline. Without a provenance sidecar they remain
`serialized_code_change_rows`, because the dataset alone does not prove
file/hunk/source-line provenance. With an exactly matching sidecar, token-stage
Grad-CAM scores are aggregated onto canonical changed-line IDs.

SimCom attribution splits a patch into contiguous chunks of 10 rows by default,
never exceeding the checkpoint's `code_line` dimension. The unchanged commit
message is repeated for every chunk. Local row/token tensor positions are
translated back to global patch positions before line aggregation, so trailing
rows are observed instead of truncated. Each chunk keeps its own prediction and
ranking; the commit export retains the top-ranked source line from every chunk.
DeepJIT retains its existing single-input truncation behavior.

DeepJIT/JITFine merge input uses semantic `<ADD>` and `<REMOVE>` regions. The
historical SimCom patch serializer reverses those marker regions. Provenance
therefore stores separate `merge` and `patch` markers in addition to the real
Git `change_type`.

For SimCom, Grad-CAM runs only through Com. For a full SimCom checkpoint, each
chunk prediction is the mean of the unchanged Sim probability and that chunk's
Com probability. The commit-level summary is explicitly labelled as the maximum
chunk probability; it is not presented as the old unchunked prediction. Neither
model's message branch, and neither SimCom's Sim component, receives a row
attribution.
