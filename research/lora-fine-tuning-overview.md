# LoRA Fine-Tuning Overview

Low-Rank Adaptation (LoRA) adds small trainable matrices to a frozen LLM, greatly reducing fine-tuning cost. LoRA-specific hyperparameters – primarily the adapter rank (r) and scaling factor (α) – critically affect performance. Research shows that different tasks and models often need different settings. In practice for text-generation and memory-augmented tasks, one should start with standard defaults and then adjust. For example, many tutorials for ~3B models use r=8 with α=8 (so α/r=1). Others use larger ranks; one study found optimal ranks 16–64 on 3B models for NLP tasks. Importantly, LoRA adapters should be enabled on all transformer layers (attention Q/K/V/O and MLP) to maximize learning

## Key LoRA Hyperparameters

- Rank (r) – Controls adapter size. A higher rank gives more capacity but uses more memory and risk of overfitting. Common practice for small models (3–4B) is to try r=4, 8, 16; large models may use 64–256. For example, Meta’s LoRA examples used 8 or 16, while other work suggests 32+ may help (though performance often saturates above a certain r). As a rule of thumb, start with a modest rank (e.g. 8 or 16) and increase if the model underfits the adaptation task.
- Scaling (α) – Scales the LoRA update before adding to model weights. A common heuristic is α ≈ r or α = 2·r. (In the official LoRA code, α=2×r is used by default) Setting α=r makes the update magnitude equal to the learned weights (1× scaling), while α=2r doubles it. Some experiments found α=2r works well, but smaller α (even 0.5×r) has given good results too. In practice, try α=r and α=2r as starting points and adjust if fine-tuning is too weak or strong.
- Dropout – A small dropout on the LoRA weights helps prevent overfitting. Typical values are around 0.05–0.1. For mid-size models (7B–13B), QLoRA used 0.1. For very large models (30B+), lower dropout (≈0.05) was found useful. For 3–4B models, setting lora_dropout=0.1 is a reasonable default.
- Learning Rate – Although not LoRA-specific, the learning rate (LR) is crucial. Fine-tuning with LoRA often uses a relatively high LR. A typical range is 2e-4 down to 5e-6. As a starting point, many use ~2e-4 for instruction fine-tuning. Smaller rates (e.g. 1e-5) may be better for very noisy or long training. The AMD tutorial for Llama-3.2B used LR=4e-5. Adjust LR upward if training is too slow (underfitting) and downward if loss blows up or overfits quickly.
- Epochs – Number of passes over the data. Few epochs (1–3) are usually sufficient. LoRA tends to converge quickly on instruction data. More epochs risk memorization. A good default is 2 epochs, checking validation metrics to avoid overtraining.
- Batch Size – LoRA fine-tuning often works best with small batches. In practice, batches of 1–4 (possibly with gradient accumulation) are common. A recent LoRA study found that smaller batches consistently gave better accuracy. For example, optimal settings often used batch size 1 or 2. This is partly because updating only a small subset of parameters (the LoRA weights) can benefit from noisier gradient estimates.
- Weight Decay – A small weight decay (e.g. 0.001–0.01) can regularize LoRA. If overfitting is observed, increasing weight decay to ~0.01–0.1 is recommended. Default values like weight_decay=0.001 are fine to start.

## Default Recommendations (3–4B Models)

Based on research and practice, a reasonable starting configuration for ~3–4B LLMs is:

- Rank (r): 8 or 16 (increase to 32 if underfitting). Example: Llama-3.2B tutorial used r=8.
- Alpha (α): set equal to r or 2×r. For r=8 try α=8 or α=16.
- Dropout: 0.1.
- Learning Rate: ~5e-5 to 2e-4. For example, 4e-5 was used for 3B in one guide; 2e-4 is a common starting point. Adjust ±3× to test sensitivity.
- Epochs: 1–2.
- Batch Size: 1–4 (possibly with gradient accumulation to simulate larger effective batch). Always use small batches.
- Target Modules: Apply LoRA to all layers’ attention and MLP weights.

These defaults should be adjusted by observing training. If training loss drops very low (<0.2) and accuracy stagnates, the model may be overfitting; try lowering LR or reducing r and adding regularization (increase dropout or weight decay). If loss remains high (underfitting), consider increasing r or LR and/or training more epochs.

## Tuning Strategies

Finding optimal LoRA hyperparameters generally requires empirical search. Useful strategies include:
- Iterative search: Vary one hyperparameter at a time (e.g. try a few values of r while fixing others) to gauge its effect. LoRA studies suggest doing a grid over (r ∈ {8,16,32,…}, α/r ∈ {1,2}, LR ∈ [5e-5,2e-4], batch size ∈ {1,2,4}) can find good combos.
- Small batch, high noise: Since LoRA uses few trainable params, stick to small batches. This improves generalization as noted in recent LoRA work.
- Monitor for over/underfitting: Plot training/validation loss. If train loss falls very low but validation stalls, reduce capacity: lower r or LR, or add dropout/weight-decay. If loss stays high, increase capacity: raise r or LR, or train more epochs.
- Alpha vs. LR tradeoff: Lower α (relative to r) increases update magnitude, similar to raising LR. If fine-tuning seems too weak, one can increase LR or decrease α.
- Layerwise adjustments: In most settings, using the same r for all layers works well. More advanced users might allow larger r in higher layers, but this is task-dependent and adds complexity.
- Hyperparameter tools: If resources allow, use random search or Bayesian optimization over a small grid. Note that LoRA tuning is relatively cheap (only adapter weights), but hardware can still be underutilized, so focus on coarse search and pruning.

No single recipe fits all tasks. As the PLoRA study observes, optimal LoRA configs differ by task and model size. For example, one study found tasks like classification vs. reasoning favored different (r,LR,batch). In practice, start from defaults above and then fine-tune parameters based on target performance.

## Memory Module Considerations

When LoRA is used to implement memory-inspired modules (like HEI-NW, SGC-RSS, SMPD), the nature of the task can guide hyperparameter choice:
- Episodic Memory (HEI-NW): This module must rapidly memorize one-shot events. You may choose a moderate LoRA rank (e.g. 16–32) and a learning rate that allows quick adaptation to high-salience inputs. A higher α (e.g. =2*r) could emphasize the new memory updates. However, because these updates are sparse and gated, overfitting is less likely – still monitor to prevent specialization on rare events.
- Schema/Relational Memory (SGC-RSS): Here the model integrates structured facts. Updates should be more gradual. A smaller rank or LR might be used so that the LoRA adapter gently adjusts weights for repeated patterns, avoiding too-rapid shifts. Dropout could be kept moderate (0.1) to prevent the adapter from memorizing noise in episodic input before consolidation.
- Spatial Map and Policy (SMPD): Learning repeated trajectories may need a balance. LoRA here could focus on encoding navigation skills, so mid-range rank (e.g. 16–32) is sensible. Since successful paths are replayed multiple times, larger r or longer training could consolidate these patterns. Regularization (dropout ~0.1) helps ensure generalized route encoding rather than spurious details.

For all these modules, the same tuning principles apply. Begin with the defaults above, then adjust based on the specific behavior of the memory system. For example, if the episodic adapter fails to recall a one-shot event, try increasing its rank or LR; if the schema adapter overfits to particular episodes, lower its LR or raise dropout. Because these memory tasks often involve rare but important updates, it can help to train with a few extra epochs on memorization tasks and then prune with smaller LR or weight averaging to consolidate knowledge, as suggested by neuroscientific inspiration (e.g., sleep replay).

## Conclusion

Selecting LoRA hyperparameters is a critical step in fine-tuning LLMs, especially when adding episodic, relational, or spatial memory modules. In summary:
- Rank (r): Start small (8–16 for 3–4B models) and increase as needed.
- Alpha (α): Set ~1–2×r.
- Dropout: ≈0.1 as a default.
- Learning Rate: ~2e-4 to 5e-5 range, tuned per task.
- Batch Size: Keep very small (1–4).
- Epochs: 1–3, with early stopping if loss plateaus.

Always watch for overfitting (reduce capacity or increase regularization) and underfitting (increase capacity or training time). Empirical tuning – possibly using a small grid search or sequential experiments – is needed to adapt to a given memory-task workload. By following these guidelines and citing best practices from recent studies, an LLM can be guided to choose effective LoRA settings for advanced memory-enhanced architectures.-
