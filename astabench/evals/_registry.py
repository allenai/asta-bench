from .arxivdigestables import (
    arxivdigestables,
    arxivdigestables_test,
    arxivdigestables_validation,
    arxivdigestables_micro,
)
from .code_diagnostics import (
    code_diagnostics,
    code_diagnostics_dev,
    code_diagnostics_small_dev,
    code_diagnostics_test,
)
from .demo import arithmetic_demo, arithmetic_with_tools, code_demo
from .discoverybench import (
    discoverybench,
    discoverybench_test,
    discoverybench_validation,
    discoverybench_micro,
)
from .e2e_discovery import (
    e2e_discovery_hard_test,
    e2e_discovery_hard_validation,
    e2e_discovery_test,
    e2e_discovery_validation,
    e2e_discovery_micro,
)
from .inspect_eval_wrappers import (
    core_bench_test,
    core_bench_validation,
    ds1000,
    ds1000_test,
    ds1000_validation,
    core_bench_micro,
    ds1000_micro,
)
from .labbench import litqa2, litqa2_inspect, litqa2_test, litqa2_validation, litqa2_micro
from .paper_finder import (
    paper_finder_litqa2_test,
    paper_finder_litqa2_validation,
    paper_finder_test,
    paper_finder_validation,
    paper_finder_micro,
    paper_finder_litqa2_micro,
)
from .sqa import sqa, sqa_dev, sqa_test, sqa_micro
from .super import super, super_test, super_validation, super_micro

__all__ = [
    "arithmetic_demo",
    "arithmetic_with_tools",
    "code_diagnostics",
    "code_diagnostics_dev",
    "code_diagnostics_test",
    "code_diagnostics_small_dev",
    "sqa",
    "sqa_dev",
    "sqa_test",
    "super",
    "super_validation",
    "super_test",
    "litqa2",
    "litqa2_inspect",
    "litqa2_test",
    "litqa2_validation",
    "paper_finder_test",
    "paper_finder_litqa2_test",
    "paper_finder_validation",
    "paper_finder_litqa2_validation",
    "code_demo",
    "core_bench_test",
    "core_bench_validation",
    "ds1000",
    "ds1000_test",
    "ds1000_validation",
    "discoverybench",
    "discoverybench_test",
    "discoverybench_validation",
    "arxivdigestables",
    "arxivdigestables_test",
    "arxivdigestables_validation",
    "e2e_discovery_test",
    "e2e_discovery_validation",
    "e2e_discovery_hard_test",
    "e2e_discovery_hard_validation",
    # Micro variants
    "arxivdigestables_micro",
    "sqa_micro",
    "litqa2_micro",
    "paper_finder_micro",
    "paper_finder_litqa2_micro",
    "discoverybench_micro",
    "core_bench_micro",
    "ds1000_micro",
    "e2e_discovery_micro",
    "super_micro",
]
