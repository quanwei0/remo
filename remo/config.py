from dataclasses import dataclass


@dataclass
class RemoConfig:
    """Knobs of Algorithms 1 (ReMo) and 2 (AdaReMo).

    mode            "remo": retry on every incorrect verdict, store every admitted lesson (Alg. 1)
                    "adaremo": critic-governed refine/store decisions + saturation freeze (Alg. 2)
    K               round budget (max attempts per task)
    redundant_mode  what AdaReMo does with a lesson the critic judges already covered:
                    "reinforce" (default): helpful+1 on the cited entry; "gate": discard; "off": store anyway
    freeze_w / freeze_rho / probe_p
                    saturation freeze: over the last freeze_w admitted episodes, if fewer than
                    freeze_rho*freeze_w touched memory (stored or reinforced) consolidation is frozen;
                    every probe_p tasks one probe episode may unfreeze it
    inject_cap_chars
                    cap on the memory text prepended to the solver (None = whole playbook)
    use_memory      False disables the inter-task memory entirely; with K=1 this is plain ReAct,
                    with K>1 refinement only. mode="remo", use_memory=True, K=1 is memory only.
    """
    mode: str = "adaremo"
    K: int = 3
    redundant_mode: str = "reinforce"
    freeze_w: int = 20
    freeze_rho: float = 0.1
    probe_p: int = 20
    inject_cap_chars: int | None = None
    use_memory: bool = True            # False: no memory read or written (ReAct when K=1, refinement-only when K>1)

    def __post_init__(self):
        assert self.mode in ("remo", "adaremo"), self.mode
        assert self.redundant_mode in ("reinforce", "gate", "off"), self.redundant_mode
        assert self.K >= 1

    @property
    def adaptive(self) -> bool:
        return self.mode == "adaremo"
