from pathlib import Path
import aiohttp
import requests
from inspect_ai.solver import Solver, solver, TaskState, Generate
import json
import os
import logging

logger = logging.getLogger(__name__)
YOUCOM_API_KEY = os.environ["YOUCOM_API_KEY"]
S2_API_KEY = os.environ["S2_API_KEY"]


def remove_suffix_after_dash(title: str) -> str:
    return title.split(" - ")[0]


async def query_youcom_search(
    question: str,
    api_key: str,
):
    url = "https://api.ydc-index.io/search"
    query = {"query": question}
    headers = {"X-API-Key": api_key}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=query) as response:
            response.raise_for_status()
            res = await response.json()
    search_results = res.get("hits", [])
    return search_results


@solver
def youcom_solver() -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.metadata.get("raw_query"):
            raise ValueError(
                "State must contain 'raw_query' in metadata to use youcom solver."
            )

        question = state.metadata["raw_query"]
        sample_id = state.sample_id
        docs = []

        try:
            search_results = await query_youcom_search(
                question,
                YOUCOM_API_KEY,
            )

            # log youcom results
            Path("youcom_search_results").mkdir(exist_ok=True)
            output_path = Path(f"youcom_search_results/youcom_{sample_id}.json")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(search_results, f, indent=2)

            titles = [remove_suffix_after_dash(res["title"]) for res in search_results if "title" in res and res["title"]]

            # call s2 api to get corpus ids from titles
            for title in titles:
                try:
                    r = requests.get(
                        'https://api.semanticscholar.org/graph/v1/paper/search/match',
                        params={"query": title, "fields": 'title,corpusId'},
                        headers={"X-API-Key": S2_API_KEY},
                        timeout=60,
                    )

                    if r.status_code == 200:
                        res = r.json()
                        docs.append(
                            {
                                "paper_id": str(res["data"][0]["corpusId"]),
                                "markdown_evidence": "",
                            }
                        )
                    else:
                        logging.error(f"Title match not found {r.status_code}: {r.text}")
                except Exception as err:
                    logging.exception(f"Request failed for title: {title}: {err}")

        except Exception as e:
            logger.exception(f"Error: {e}")
            return state

        realres = {"output": {"query_id": state.sample_id, "results": docs}}

        state.output.completion = json.dumps(realres)

        return state

    return solve
