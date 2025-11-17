# sdk/python/example_run.py
"""
Example usage of the ERCP Python SDK.
Run the reference server first:
    uvicorn server.ercp_server:app --reload --port 8080
Then run this script.
"""

from ercp_client import ERCPClient
import time

def main():
    client = ERCPClient(base_url="http://localhost:8080", api_key=None)

    problem = "Why does water boil at different temperatures at different altitudes?"
    print("Sending ERCP run request...")
    try:
        resp = client.run(problem_description=problem)
        trace_id = resp.get("trace_id")
        print("Trace ID:", trace_id)
        print("Status:", resp.get("status"))
        print("Final reasoning (short):", resp.get("final_reasoning", {}).get("reasoning_text"))
    except Exception as e:
        print("Error calling ERCP server:", e)

if __name__ == "__main__":
    main()
