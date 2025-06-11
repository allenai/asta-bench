import sys
import json
from astabench.evals.arxivdigestables.solvers.asta_table_agent import generate_table


def main():
    # Expect the question as the first command-line argument.
    if len(sys.argv) < 3:
        sys.exit("Usage: python table_subprocess.py <INPUT PROMPT> <CORPUS ID LIST>")
    input_prompt = sys.argv[1]
    corpus_ids = sys.argv[2].split(",")
    response = generate_table(input_prompt=input_prompt, corpus_ids=corpus_ids)
    print("<START>" + json.dumps(response.model_dump(mode="json")))


if __name__ == "__main__":
    main()
