import asyncio
from pathlib import Path
import aiohttp
import httpx
import requests
from inspect_ai.solver import Solver, solver, TaskState, Generate
import json
import os
import logging

logger = logging.getLogger(__name__)
YOUCOM_SEARCH_API_KEY = os.environ["YOUCOM_SEARCH_API_KEY"]
S2_API_KEY = os.environ["S2_API_KEY"]


def clean_title(title: str) -> str:
    return title.split(" - ")[0].replace("...", "")


def get_pmc_id(url: str) -> str | None:
    if "PMC" in url:
        return "PMCID:" + url.rstrip("/").split("/")[-1].replace("PMC", "")
    return None


async def fetch_s2_data(client, title, pmc_id):
    headers = {"X-API-Key": S2_API_KEY}
    match_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    pmc_url = f"https://api.semanticscholar.org/graph/v1/paper/{pmc_id}" if pmc_id else None

    match_task = client.get(
        match_url,
        params={"query": title, "fields": "title,corpusId"},
        headers=headers,
        timeout=60,
    )

    pmc_task = None
    if pmc_id:
        pmc_task = client.get(
            pmc_url,
            params={"fields": "corpusId"},
            headers=headers,
            timeout=60,
        )

    responses = await asyncio.gather(
        match_task,
        pmc_task if pmc_task else asyncio.sleep(0),  # dummy awaitable if no pmc_id
        return_exceptions=True,
    )
    return responses


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
                YOUCOM_SEARCH_API_KEY,
            )

            # log youcom results
            Path("youcom_search_results").mkdir(exist_ok=True)
            output_path = Path(f"youcom_search_results/youcom_{sample_id}.json")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(search_results, f, indent=2)

            # call s2 api to get corpus ids from titles
            async with httpx.AsyncClient() as client:
                tasks = []
                for sr in search_results:
                    title = clean_title(sr["title"])
                    pmc_id = get_pmc_id(sr["url"])
                    tasks.append(fetch_s2_data(client, title, pmc_id))

                results = await asyncio.gather(*tasks)

                for i, (match_res, pmc_res) in enumerate(results):
                    title = clean_title(search_results[i]["title"])

                    try:
                        if isinstance(match_res, httpx.Response) and match_res.status_code == 200:
                            res_json = match_res.json()
                            docs.append({
                                "paper_id": str(res_json["data"][0]["corpusId"]),
                                "markdown_evidence": "",
                            })
                        elif isinstance(pmc_res, httpx.Response) and pmc_res.status_code == 200:
                            res_json = pmc_res.json()
                            docs.append({
                                "paper_id": str(res_json["corpusId"]),
                                "markdown_evidence": "",
                            })
                        else:
                            logging.error(f"Failed to find match for: {title}, response: {match_res} (for pmc doc: {pmc_res})")
                    except Exception as err:
                        logging.exception(f"Request failed for title: {title}: {err}")

        except Exception as e:
            logger.exception(f"Error: {e}")
            return state

        logging.warning(f"Found {len(docs)} titles out of {len(search_results)} titles returned by You.com")

        realres = {"output": {"query_id": state.sample_id, "results": docs}}

        state.output.completion = json.dumps(realres)

        return state

    return solve
