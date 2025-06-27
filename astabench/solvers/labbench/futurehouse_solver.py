from inspect_ai.solver import solver, Solver, chain
from astabench.solvers.futurehouse import futurehouse_solver
from astabench.solvers.labbench.multi_choice_solver import multi_choice_solver

@solver
def litqa_solver(
    model: str, agent: str = "CROW"
) -> Solver:
    chainlist = [
        futurehouse_solver(agent, max_wait_time=60*15, polling_interval=10),
        multi_choice_solver(model, mcq_string=False),
    ]
    return chain(chainlist)
