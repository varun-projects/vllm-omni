# Debugging Case Studies

Six real bugs from this repository, worked through end to end: the reported
symptom, the diagnosis, the actual fix, and the invariant the fix restored.

Every case study here is a **closed issue with a merged fix**. The commit
hashes are on `main` and the diffs are reproducible with `git show`.

## Why read these

Design documents tell you how a system is supposed to work. Bugs tell you where
it is actually hard. Each of these was chosen because it teaches something that
is not obvious from reading the code in a healthy state:

- that a correct-looking batch can make a request's output depend on its
  neighbours,
- that `encode(decode(x)) != x`, and what that does to a cache,
- that "the client is active" and "the model is active" are different facts,
- that a `daemon=True` child process is not enough to prevent an orphan,
- that broadening a safety check can break a valid configuration,
- and that an omitted parameter is not the same as a parameter set to its
  placeholder.

## Index

| # | Case study | Difficulty | Subsystem | Issue → Fix |
| --- | --- | --- | --- | --- |
| 01 | [Mixed-length batches corrupt Qwen-Image-2.1 output](01-qwen-image-21-mixed-batch-rope.md) | Difficult | Diffusion batching, RoPE | [#10374](https://github.com/vllm-project/vllm-omni/issues/10374) → [#10376](https://github.com/vllm-project/vllm-omni/pull/10376) |
| 02 | [AR-to-DiT KV reuse breaks on token re-encoding](02-hunyuanimage3-ar-token-identity.md) | Difficult | Cross-stage KV reuse, tokenization | [#9214](https://github.com/vllm-project/vllm-omni/issues/9214) → [#9323](https://github.com/vllm-project/vllm-omni/pull/9323) |
| 03 | [Duplex sessions expire while the user is still talking](03-duplex-idle-lease.md) | Difficult | Duplex session lifetime | [#10579](https://github.com/vllm-project/vllm-omni/issues/10579) → [#10617](https://github.com/vllm-project/vllm-omni/pull/10617) |
| 04 | [Diffusion workers survive a serve-root SIGKILL](04-diffusion-worker-orphans.md) | Difficult | Process lifecycle, startup | [#10398](https://github.com/vllm-project/vllm-omni/issues/10398) → [#10440](https://github.com/vllm-project/vllm-omni/pull/10440) |
| 05 | [A broadened guard blocks a valid speculative config](05-cosyvoice3-dspark-capture-contract.md) | Medium | MRv2 output contracts | [#10425](https://github.com/vllm-project/vllm-omni/issues/10425) → [#10485](https://github.com/vllm-project/vllm-omni/pull/10485) |
| 06 | [An omitted guidance_scale silently disables CFG](06-sdxl-guidance-scale-default.md) | Medium | Sampling-parameter defaults | [#10370](https://github.com/vllm-project/vllm-omni/issues/10370) → [#10462](https://github.com/vllm-project/vllm-omni/pull/10462) |

## Suggested reading order

Difficulty here means *diagnostic* difficulty, not diff size.

If you are new to the codebase, read **06** first. It is the smallest complete
story — one wrong line, a clear invariant — and it establishes the shape the
others follow. Then **05**, which is about a contract between a model and a
runner. Then the four difficult ones in any order; **01** and **02** are about
numerical and cache correctness, **03** and **04** about lifetime and
concurrency.

## The common shape

Reading them together, a pattern emerges. Five of the six are cases where
something **implicit** was relied upon:

| Case | The implicit assumption | What broke it |
| --- | --- | --- |
| 01 | "Batch padding does not affect per-request math" | Position embeddings derived from the padded layout |
| 02 | "The same text means the same tokens" | Tokenization is not invertible |
| 03 | "An append means the model ran" | Appends can be buffered without a model append |
| 04 | "`daemon=True` plus PDEATHSIG prevents orphans" | A race window before PDEATHSIG is armed |
| 06 | "An unset field is a field set to its default" | The placeholder default was a meaningful value |

Case 05 is the inverse and just as instructive: a **fix** made a safety check
stricter than the invariant it was protecting, and broke a legitimate
configuration.

The practical lesson for contributors: when you add a check, state precisely
which invariant it protects, and make the check match that invariant exactly —
neither broader nor narrower.

## How these were reconstructed

Each case study was built from the GitHub issue, the merged pull request, and
`git show <commit>` on the fix. To follow along:

```bash
# Case 01
git show e13bd170e801a678294d9fc248c007ca22d194c4
# Case 02
git show 6039984e397d3b4b28afd4538fa0207594e3ecba
# Case 03
git show 7a827054152275de4fa43653f45155138945bbd8
# Case 04
git show bc7a0f487937cc16f2bc5dcfc3a48713cd3dd51e
# Case 05
git show 8164ed75bb6fbe2a51728ebb7470169fe5e8e579
# Case 06
git show 7fa9bbf24892d55e87a453408ecbbfd7a9a949d9
```
