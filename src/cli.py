import argparse
import sys


def cmd_ingest(args):
    raise NotImplementedError("ingestion pipeline not yet implemented")


def cmd_query(args):
    raise NotImplementedError("query pipeline not yet implemented")


def cmd_logs(args):
    raise NotImplementedError("log review not yet implemented")


def main():
    parser = argparse.ArgumentParser(prog="sommelier")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index Apache Pinot docs into Qdrant")
    ingest.add_argument("--docs-path", required=True, help="Path to apache/pinot/website/docs")
    ingest.add_argument("--version", default="latest", help="Pinot version tag to store on chunks")
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Run a one-shot query through the full pipeline")
    query.add_argument("question", help="Question to ask Sommelier")
    query.add_argument("--version", default="latest", help="Pinot version filter")
    query.set_defaults(func=cmd_query)

    logs = sub.add_parser("logs", help="Review query traces")
    logs.add_argument("--review", action="store_true", help="Interactive review with golden set promotion")
    logs.set_defaults(func=cmd_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
