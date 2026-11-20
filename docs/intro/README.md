# vLLM-Omni Code Introduction

A guided, code-level introduction to vLLM-Omni for new contributors.

## What this folder is

The [design documents](../design/index.md) in this repository explain *what*
each feature does and *how it is configured*. This folder answers a different
question: **where does the code live, and what actually happens when a request
flows through it?**

Every document here is anchored to concrete files, classes, and functions in
`vllm_omni/`, with Mermaid flowcharts that mirror the real control flow rather
than an idealized picture of it.

## Reading order

Read these three in order; they build on each other.

| # | Document | What you get |
| --- | --- | --- |
| 1 | [Project Overview](01-project-overview.md) | What problem vLLM-Omni solves, the vocabulary, and a map of the repository |
| 2 | [Architecture in Detail](02-architecture.md) | The layer model, config resolution, process topology, and who owns what |
| 3 | [Request Lifecycle](03-request-lifecycle.md) | One request traced end to end through the actual call stack |

## Feature deep dives

One document per major subsystem. Each is self-contained; read the three
documents above first, then jump to whichever subsystem you are touching.

| Document | Subsystem | Primary source |
| --- | --- | --- |
| [Stage Runtime and Orchestrator](features/01-stage-runtime-and-orchestrator.md) | Control plane: placement, lifecycle, cross-stage routing | `vllm_omni/engine/` |
| [AR Stage Scheduling](features/02-ar-stage-scheduling.md) | Autoregressive stages built on vLLM's scheduler | `vllm_omni/core/sched/`, `vllm_omni/worker_v2/` |
| [Diffusion Stage Execution](features/03-diffusion-stage.md) | Denoising runtime, request/step batching, executors | `vllm_omni/diffusion/` |
| [OmniConnector Transport](features/04-omni-connector-transport.md) | Moving tensors and KV between stages | `vllm_omni/distributed/omni_connectors/` |
| [Full-Duplex Sessions](features/05-full-duplex-sessions.md) | Realtime streaming audio in and out, session leases | `vllm_omni/engine/duplex/` |
| [Prefix Caching and KV Reuse](features/06-prefix-caching-and-kv.md) | Reusing computation across and within stages | `vllm_omni/core/prefix_cache/` |
| [Distributed Parallelism](features/07-distributed-parallelism.md) | TP, SP/USP, CFG, EP, HSDP, VAE patch parallel | `vllm_omni/distributed/`, `vllm_omni/config/composable_parallel/` |
| [Entrypoints and APIs](features/08-entrypoints-and-apis.md) | Offline Python, OpenAI-compatible HTTP, duplex WebSocket | `vllm_omni/entrypoints/` |

## Debugging case studies

The [case-studies](case-studies/README.md) folder works through real bugs that
were reported, diagnosed, and fixed in this repository. Each one is a complete
story: the symptom, the diagnosis, the actual fix diff, a flowchart of the
broken and repaired control flow, and the invariant the fix restored.

These are the fastest way to build intuition about where this system is
genuinely hard. Start with the
[index](case-studies/README.md), which sorts them by difficulty and by the
subsystem they teach.

## Conventions used in these documents

- File references are repository-relative, e.g. `vllm_omni/engine/orchestrator.py:1369`.
- Flowcharts use Mermaid. GitHub and MkDocs both render them inline.
- Where a claim depends on a specific revision, the revision is named. This
  codebase moves quickly; verify a surprising claim against the code before
  relying on it.
