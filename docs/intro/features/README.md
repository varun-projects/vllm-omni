# Feature Deep Dives

One document per major subsystem of vLLM-Omni. Read
[Project Overview](../01-project-overview.md),
[Architecture in Detail](../02-architecture.md), and
[Request Lifecycle](../03-request-lifecycle.md) first — these assume that
vocabulary.

| # | Document | Read this when you are… |
| --- | --- | --- |
| 01 | [Stage Runtime and Orchestrator](01-stage-runtime-and-orchestrator.md) | Changing stage placement, replica lifecycle, startup, or cross-stage routing |
| 02 | [AR Stage Scheduling](02-ar-stage-scheduling.md) | Touching the AR scheduler, Model Runner V2, or CUDA graph contracts |
| 03 | [Diffusion Stage Execution](03-diffusion-stage.md) | Working on denoising, diffusion batching, or a DiT pipeline |
| 04 | [OmniConnector Transport](04-omni-connector-transport.md) | Moving payloads or KV between stages, or adding a connector |
| 05 | [Full-Duplex Sessions](05-full-duplex-sessions.md) | Working on realtime audio, turn detection, or session lifetime |
| 06 | [Prefix Caching and KV Reuse](06-prefix-caching-and-kv.md) | Working on caching, reuse correctness, or AR→DiT KV hand-off |
| 07 | [Distributed Parallelism](07-distributed-parallelism.md) | Combining TP/SP/CFG/EP/HSDP or debugging a parallel topology |
| 08 | [Entrypoints and APIs](08-entrypoints-and-apis.md) | Adding an endpoint or changing request/response shapes |

Each document states its primary source directory at the top and links to the
relevant design document where one exists. The design documents are
authoritative for configuration and compatibility; these are oriented toward
reading and changing the code.
