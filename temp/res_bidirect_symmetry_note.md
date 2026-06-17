# Note: why `res_bidirect` changes when the pair order is swapped

Protein-protein interaction is an undirected biological relation, so ideally the model should satisfy:

```text
score(p1, p2) == score(p2, p1)
```

The current `interaction_res_bidirect` model does not enforce that property. The observed difference comes from order-sensitive parts of `dscript/models/interaction_res_bidirect.py`:

1. Separate sequence-side modules are learned for side 0 and side 1:
   - `sa0` and `sa1` are different attention modules.
   - `ln0_*` and `ln1_*` are different layer norms.
   - `ff0` and `ff1` are different feed-forward modules.
   - `pool0` and `pool1` are different pooling heads.

   These are defined around lines 392-405 and used around lines 491-509. If `p1` and `p2` are swapped, each protein passes through a different set of learned parameters, so the pooled vectors can change.

2. The global pair feature includes an antisymmetric term:

   ```python
   int_sub = p0 - p1
   ```

   This is at line 514. When the pair is swapped, this term becomes `p1 - p0`, which is the negative of the original term. It is then concatenated into the classifier input at line 597:

   ```python
   g = torch.cat([g_add, g_mul, int_abs, int_sub], dim=1)
   ```

   Because the classifier sees `int_sub`, it can learn different outputs for `(p1, p2)` and `(p2, p1)`.

3. The contact-map branch also is not explicitly symmetrized before classification. The model builds `C` from `(contact_e0, contact_e1)` at line 482 and sends `yhat` into `self.clf(yhat, g)` at line 598. Unless the contact-map construction and classifier are designed to be transpose-invariant, swapping the protein order can change this branch too.

## Should we modify the model?

If the task is standard PPI prediction, yes, I would modify the model or inference so that the final prediction is symmetric. The cleanest options are:

1. **Inference-time symmetry, easiest and compatible with existing checkpoints**

   Predict both directions and average:

   ```text
   score_sym(p1, p2) = 0.5 * (score(p1, p2) + score(p2, p1))
   ```

   This does not require retraining and is the safest immediate fix for reporting/evaluation.

2. **Model-level symmetry, better for future training**

   Remove or replace order-sensitive features. For example:

   - Share the side-specific modules, or use one common attention/pooling stack for both proteins.
   - Remove `int_sub`, or replace it with only symmetric features such as `p0 + p1`, `p0 * p1`, and `abs(p0 - p1)`.
   - Symmetrize the contact-map path, for example by combining predictions from `(p1, p2)` and `(p2, p1)` inside `forward`, or by making the classifier operate on transpose-invariant summaries.

3. **Training-time consistency regularization**

   Add a loss term that penalizes disagreement between both directions:

   ```text
   loss = BCE(score(p1, p2), y) + BCE(score(p2, p1), y)
          + lambda * abs(score(p1, p2) - score(p2, p1))
   ```

   This helps, but by itself it does not guarantee exact equality unless the architecture or inference also enforces symmetry.

Recommendation: for the current trained checkpoint, use inference-time averaging. For the next model version, make the architecture symmetric and then retrain, because the current checkpoint was trained with order-sensitive parameters and features.
