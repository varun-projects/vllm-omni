# Architecture in Detail

This document covers the layers of vLLM-Omni, the configuration pipeline that
turns a model name into running processes, and the ownership boundaries that
the codebase enforces. [Request Lifecycle](03-request-lifecycle.md) then traces
a single request through what is built here.

## The four planes

It helps to think of vLLM-Omni as four planes with deliberately narrow
interfaces between them. Most review disagreements in this repository are
really arguments about which plane a piece of logic belongs to.

```mermaid
flowchart TB
    subgraph api["Public surface"]
        direction LR
        offline["Omni / AsyncOmni<br/>offline Python"]
        http["OpenAI-compatible server<br/>vllm serve --omni"]
        ws["DuplexOmni<br/>realtime WebSocket"]
    end

    subgraph control["Control plane"]
        direction LR
        engine["AsyncOmniEngine<br/>composition root"]
        orch["Orchestrator<br/>request state + routing"]
        runtime["StageRuntime<br/>placement + lifecycle"]
    end

    subgraph exec["Execution plane"]
        direction LR
        arstage["AR / generation stage<br/>vLLM engine core + MRv2"]
        diffstage["Diffusion stage<br/>diffusion engine + worker"]
    end

    subgraph data["Data plane"]
        direction LR
        conn["OmniConnector<br/>payload + KV transport"]
        out["MultimodalPayload<br/>OmniRequestOutput"]
    end

    offline --> engine
    http --> engine
    ws --> engine
    engine --> orch
    engine --> runtime
    orch -->|submit| arstage
    orch -->|submit| diffstage
    runtime -.->|"launches, owns"| arstage
    runtime -.->|"launches, owns"| diffstage
    arstage -->|payload| conn
    diffstage -->|payload| conn
    conn -->|payload| arstage
    conn -->|payload| diffstage
    arstage --> out
    diffstage --> out
    out --> orch
```

The rules that keep this honest:

1. **The control plane never touches tensors.** The Orchestrator routes request
   *identities* and small metadata. Large intermediates travel through the data
   plane, and the Orchestrator only holds handles to them.
2. **The data plane never makes routing decisions.** A connector moves bytes
   from A to B because it was told to. Choosing B is the Orchestrator's job.
   This is stated explicitly in the key-components table of
   [the design overview](../design/architecture_overview.md).
3. **The execution plane does not know the topology.** A stage knows its own
   configuration and its input and output contracts. It does not know how many
   stages exist or where they run.
4. **Entrypoints do not own engine state.** They translate protocols. Every
   entrypoint funnels into the same `AsyncOmniEngine`.

## Configuration resolution

This is the part of the system that most often surprises newcomers, because a
single `vllm serve <model> --omni` expands into a validated, multi-stage,
transport-safe configuration before a single weight is loaded.

```mermaid
flowchart TB
    subgraph L1["Layer 1 · Authoring inputs"]
        reg["Pipeline registry<br/>vllm_omni/config/pipeline_registry.py<br/>(developer-authored PipelineConfig)"]
        dep["deploy/&lt;model&gt;.yaml<br/>DeployConfig<br/>(the file users edit)"]
        cli["CLI / Python overrides<br/>cli_overrides, stage_overrides"]
    end

    subgraph L2["Layer 2 · Resolution boundary"]
        resolve["resolve_omni_config()<br/>vllm_omni/config/resolver.py:215"]
        factory["StageConfigFactory.create_from_model()<br/>config_factory.py:130"]
        vomni["VllmOmniConfig.from_pipeline_config()<br/>omni_config.py:2569"]
    end

    subgraph L3["Layer 3 · Startup hand-off"]
        res["OmniConfigResolution<br/>resolver.py:27<br/>pipeline_config + typed stage_configs"]
    end

    subgraph L4["Layer 4 · Launch planning"]
        sr["StageRuntime<br/>engine/stage_runtime.py:167"]
        plan["LogicalStageInitPlan<br/>→ ReplicaInitPlan<br/>(runtime-private)"]
    end

    subgraph L5["Layer 5 · Engine materialization"]
        vcfg["VllmConfig<br/>(in the AR stage process)"]
        ocfg["OmniDiffusionConfig<br/>(in the diffusion stage process)"]
    end

    reg --> resolve
    dep --> resolve
    cli --> resolve
    resolve --> factory
    resolve --> vomni
    factory --> res
    vomni --> res
    res --> sr --> plan
    plan --> vcfg
    plan --> ocfg
```

### Why these layers exist

`resolve_omni_config()` is the **only** production resolution entrypoint. Its
signature (`vllm_omni/config/resolver.py:215`) shows exactly what is allowed to
influence a resolved configuration:

```python
def resolve_omni_config(
    model: str,
    *,
    trust_remote_code: bool | None,
    deploy_config_path: str | None,
    cli_overrides: Mapping[str, Any] | None,
    stage_overrides: Mapping[str, Mapping[str, Any]] | None,
    strategy_config_path: str | None,
) -> OmniConfigResolution:
```

Two paths come out of it:

- **Registered pipeline.** `StageConfigFactory.create_from_model()` finds a
  `PipelineConfig` in the registry by `model_type`, by HF `architectures`
  matching `hf_architectures`, or — for sibling generations that ship the same
  architecture string — by evaluating `hf_config_predicate`. MiniCPM-o 4.5 and
  2.6 both declare `architectures=["MiniCPMO"]`, so the 4.5 pipeline
  disambiguates on `hf_config.version`.
- **Generic diffusion.** A model with no registered pipeline but a recognizable
  `model_index.json` falls through to
  `StageConfigFactory.create_typed_default_diffusion()`, which synthesizes a
  single-stage diffusion topology.

The result is an `OmniConfigResolution` (`resolver.py:27`), a frozen dataclass
carrying the effective `PipelineConfig` *and* the typed `stage_configs`. Both
views must describe the same topology, including stages the factory injected.
This object is the sole startup hand-off: `AsyncOmniEngine` and the headless
startup path both consume it and nothing else.

### Ownership rules

These are enforced by convention and reviewed strictly:

| Object | Owns | Must not |
| --- | --- | --- |
| `PipelineConfig` | Stage topology, execution types, model capabilities, stage relationships | Be user-editable; encode placement |
| `DeployConfig` | Placement, replicas, devices, connectors, deploy-time defaults | Redefine the model graph |
| `OmniConfigResolution` | The single startup hand-off | Be bypassed by a second resolution path |
| `StageRuntime` | Launch planning, replica lifecycle | Leak `ReplicaInitPlan` as user configuration |
| `VllmConfig` / `OmniDiffusionConfig` | Process-local engine materialization | Be constructed before placement is known |

`DeployConfig` (`vllm_omni/config/stage_config.py:918`) draws one more internal
line that is easy to trip over: its top-level fields are **pipeline-wide** and
apply uniformly to every stage, while anything that legitimately varies per
stage lives in the `StageDeployConfig` entries under `stages:`.

```python
class DeployConfig:
    """Loaded from deploy/<model>.yaml — the only config file users edit.

    Top-level fields (``trust_remote_code``, ``distributed_executor_backend``,
    ``dtype``, ``quantization``, ``enable_prefix_caching``,
    ``enable_chunked_prefill``, ``data_parallel_size``,
    ``pipeline_parallel_size``) are pipeline-wide: they apply uniformly to
    every stage. Fields that legitimately vary per stage live in the
    individual ``StageDeployConfig`` entries under ``stages:``.
    """
```

The legacy `stage_args` YAML path has been removed. If you find documentation
or a blog post referencing it, that content predates the typed config path.

## Process topology

A deployment is a tree of processes. Which processes exist depends entirely on
`DeployConfig`, not on the model.

```mermaid
flowchart TB
    root["serve root / API server process<br/>uvicorn + AsyncOmniEngine + Orchestrator"]

    subgraph stage0["Stage 0 replicas"]
        s0r0["EngineCore proc<br/>stage_engine_core_proc.py"]
        s0r1["EngineCore proc<br/>(replica 1)"]
    end

    subgraph stage1["Stage 1 replicas"]
        s1r0["EngineCore proc"]
    end

    subgraph stage2["Stage 2 (diffusion)"]
        dproc["stage_diffusion_proc.py"]
        dw0["DiffusionWorker-0<br/>GPU 0"]
        dw1["DiffusionWorker-1<br/>GPU 1"]
    end

    root -->|ZMQ| s0r0
    root -->|ZMQ| s0r1
    root -->|ZMQ| s1r0
    root -->|ZMQ| dproc
    dproc -->|"spawn, daemon=True"| dw0
    dproc -->|"spawn, daemon=True"| dw1

    s0r0 -.->|"payload via connector"| s1r0
    s1r0 -.->|"payload via connector"| dproc
```

Key implementation facts:

- Stage engine processes are launched by `StageRuntime.launch_stage_engines()`
  (`vllm_omni/engine/stage_runtime.py:359`) and spoken to over ZMQ through
  `StageClient` / `StageEngineCoreClient`.
- A `StagePool` (`vllm_omni/engine/stage_pool.py:196`) holds the clients for
  one stage's replicas and owns replica selection, request-to-replica binding,
  and liveness. The Orchestrator asks a pool to `pick()` a replica; it never
  addresses a replica directly.
- Diffusion stages add a second level: the stage process owns a
  `MultiprocDiffusionExecutor` which spawns one `DiffusionWorker` per GPU rank.
  Those workers are `daemon=True` *and* arm a Linux parent-death signal — the
  subtleties of getting that right are
  [case study 04](case-studies/04-diffusion-worker-orphans.md).
- Stages can also run **co-located**: a diffusion stage reachable in-process
  uses `InlineStageDiffusionClient` (`vllm_omni/diffusion/inline_stage_diffusion_client.py`)
  instead of a process boundary. The Orchestrator's code path is identical.

## The execution plane, by stage type

### AR and generation stages

These reuse vLLM's engine core, worker, and model runner, with omni-specific
subclasses layered on:

```mermaid
flowchart TB
    client["StageEngineCoreClient"] -->|"EngineCoreRequest"| core["EngineCore (vLLM)"]
    core --> sched["OmniARScheduler / OmniGenerationScheduler<br/>vllm_omni/core/sched/"]
    sched -->|"OmniSchedulerOutput"| worker["GPU worker<br/>vllm_omni/worker/gpu_ar_worker.py"]
    worker --> runner["OmniGPUModelRunner (MRv2)<br/>vllm_omni/worker_v2/omni_model_runner.py"]
    runner --> model["Model implementation<br/>vllm_omni/model_executor/models/..."]
    model --> runner
    runner -->|"hidden states, sampled tokens,<br/>multimodal outputs"| worker
    worker -->|"OmniEngineCoreOutput"| core
    core -->|"ZMQ"| client
```

The omni schedulers subclass vLLM's scheduler through `OmniSchedulerMixin`
(`vllm_omni/core/sched/omni_scheduler_mixin.py:198`). They add what the extra
modalities need: multimodal prefix ranges, chunk receive handles, prewarm
hints, and cross-stage admission. `OmniSchedulerOutput`
(`vllm_omni/core/sched/output.py:134`) extends vLLM's `SchedulerOutput` with
those fields rather than passing them out of band.

### Diffusion stages

Diffusion does not use a vLLM scheduler. It has its own:

```mermaid
flowchart TB
    dclient["StageDiffusionClient"] -->|"OmniDiffusionRequest"| dengine["DiffusionEngine<br/>diffusion/diffusion_engine.py"]
    dengine --> dsched["RequestScheduler or StepScheduler<br/>diffusion/sched/"]
    dsched -->|"DiffusionSchedulerOutput"| dexec["DiffusionExecutor<br/>uniproc / multiproc / ray"]
    dexec --> dworker["DiffusionWorker<br/>diffusion/worker/diffusion_worker.py"]
    dworker --> dmr["DiffusionModelRunner"]
    dmr --> pipe["Pipeline<br/>diffusion/models/&lt;model&gt;/"]
```

The two schedulers implement different batching philosophies, both deriving
from `BaseScheduler` (`vllm_omni/diffusion/sched/base_scheduler.py:74`):

- **`RequestScheduler`** admits a batch of requests and runs all their
  denoising steps together. Requires compatible shapes and sampling params.
- **`StepScheduler`** advances requests step by step, so requests can join and
  leave the running batch between steps — closer in spirit to continuous
  batching.

Compatibility is decided by a *batch sampling params key*
(`RequestBatchSamplingParamsKey` / `StepBatchSamplingParamsKey` in
`diffusion/sched/interface.py`). Two requests batch together only if their keys
match. This is why a request's output can be affected by what else is in its
batch when a pipeline's batching contract is subtly wrong — the subject of
[case study 01](case-studies/01-qwen-image-21-mixed-batch-rope.md).

## Data plane: payloads and outputs

Two abstractions carry results:

- **`MultimodalPayload`** separates tensor content from metadata so the control
  plane can route a payload by its metadata without materializing or
  deserializing the tensors.
- **`OmniRequestOutput`** is the common output envelope for both pipeline and
  diffusion results, so entrypoints have one shape to render.

Transport between stages goes through an `OmniConnector`
(`vllm_omni/distributed/omni_connectors/connectors/base.py:12` defines
`OmniConnectorBase`). The available implementations are:

| Connector | Scope |
| --- | --- |
| `shm_connector.py` | Shared memory, single node |
| `mooncake_store_connector.py`, `mooncake_transfer_engine_connector.py` | Multi-node Mooncake |
| `nixl_connector.py` | NIXL |
| `mori_transfer_engine_connector.py` | Mori |
| `yuanrong_connector.py`, `yuanrong_transfer_engine_connector.py` | Yuanrong |

KV-cache transfer (as opposed to payload transfer) is coordinated by
`OmniKVTransferManager` (`omni_connectors/kv_transfer_manager.py:352`). See
[OmniConnector Transport](features/04-omni-connector-transport.md).

## What to read next

- [Request Lifecycle](03-request-lifecycle.md) — the same architecture, traced
  dynamically through one request.
- [Stage Runtime and Orchestrator](features/01-stage-runtime-and-orchestrator.md)
  — the control plane in depth.
