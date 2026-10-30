import argparse
import asyncio
import csv
import sys
from pathlib import Path
from time import time

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from model import get_model

MCP_SERVER_CONFIG = {
    "my_mcp_server": {
        "url": "http://localhost:8000/mcp",
        "transport": "http"
    }
}

REPOS = [
    "apache_kafka",
    "apache_tomcat",
    "hibernate_hibernate-orm",
    "spring-projects_spring-boot",
    "jenkinsci_github-plugin"
]

QUESTIONS = [
    "What is the name and description of the repository?",
    "When was the repository created, and when was it last updated?",
    "How many contributors have contributed to this repository?",
    "Which contributor has made the highest number of commits?",
    "How many total commits are recorded in the repository?",
    "What is the most recent commit message and who authored it?",
    "What is the oldest commit in the repository?",
    "How many open issues are currently in the repository?",
    "How many closed issues are currently in the repository?",
    "Which issue was created most recently?",
    "Which issue has been open for the longest time?",
    "Which issue has received the highest number of comments?",
    "How many issue comments exist in total?",
    "How many commit comments exist in total?",
    "Which project contains the largest number of issues?",
    "How many issue events are associated with issue #2?",
    "Which contributor most recently made a commit?",
    "Which issue was closed most recently?",
    "Which contributor has authored at least one commit comment?",
    "What are the top 5 contributors ranked by number of commits?",

    "What is the average number of commits per contributor?",
    "What percentage of issues are currently closed?",
    "What percentage of issues are currently open?",
    "What is the ratio of open issues to closed issues?",
    "What is the average number of comments per issue?",
    "What is the median number of comments per issue?",
    "What is the average number of issue events per issue?",
    "What is the average number of commit comments per commit?",
    "What percentage of contributors have made more than 10 commits?",
    "What is the standard deviation of commits across contributors?",
    "What is the average time (in days) taken to close an issue?",
    "What is the median time (in days) taken to close an issue?",
    "What percentage of issues were closed within 7 days of creation?",
    "What percentage of issues were closed within 30 days of creation?",
    "Which contributor has the highest commit-to-comment ratio?",
    "What percentage of all comments are issue comments versus commit comments?",
    "What is the average number of commits per month since the repository was created?",
    "What was the percentage growth or decline in commits from the previous month?",
    "What percentage of issues have at least one comment?",
    "Which contributor accounts for the highest percentage of all commits?",

    "Forecast the number of commits expected in the next month.",
    "Cluster issues into thematic groups based on their titles and descriptions.",
    "Summarize the most discussed technical topics across last 10 issues and comments.",
    "Assign issue #2 to contributor.",
    "Find duplicate or highly similar issues across 50 examples using semantic similarity.",
    "Forecast the number of issue comments the issue #2 will have.",
    "Cluster the last 10 commits in severity (Low, Medium, High) based on files changed.",
    "Estimate the probability issue #2 will be closed within 30 days.",
    "Based on last 10 commits of top 5 contributors, classify them in their specialty.",
    "Based on last 10 open issues, propose 5 sub teams to split workload."
]

def extract_token_count(response) -> int:
    """Extract total tokens from response usage metadata or response metadata."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return response.usage_metadata.get("total_tokens", 0)
    elif hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        return token_usage.get("total_tokens", 0)
    return 0

async def invoke_with_retry(model, messages, max_tries=10):
    """Asynchronous helper function to retry model invocation up to max_tries on any exception."""
    for attempt in range(1, max_tries + 1):
        try:
            return model.invoke(messages)
        except Exception as e:
            print(f"\n[!] Model invocation error (Attempt {attempt}/{max_tries}): {e}")
            if attempt == max_tries:
                print("[!] Max retries reached. Raising exception.")
                raise e
            wait_time = min(2 ** (attempt - 1), 60)
            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

async def run_agent_batch(repo_name: str, questions: list, output_csv: str = "repository_answers.csv"):
    _, aimodel = get_model()
    mcp_client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}
    
    bound_model = aimodel.bind_tools(tools)

    try:
        with open("system_prompts/system_prompt_brute_force.txt", "r", encoding="utf-8") as infile:
            system_prompt_text = infile.read().strip().replace("{repo_name}", repo_name)
    except FileNotFoundError:
        system_prompt_text = f"You are an AI assistant for {repo_name}."
    
    if repo_name == "none":
        print("Use --repo to select a repository.")
        return

    print(f"Running agent for repository '{repo_name}' across {len(questions)} questions...")
    results = []

    try:
        for question in questions:
            print(f"\n--- Processing Question: {question} ---")
            
            messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=question)
            ]
            
            total_tokens = 0
            tool_use_count = 0
            answer = ""
            
            try:
                response = await invoke_with_retry(bound_model, messages, max_tries=5)
                total_tokens += extract_token_count(response)
                messages.append(response)
                
                tool_iteration = 0
                max_tool_iterations = 10

                while response.tool_calls and tool_iteration < max_tool_iterations:
                    tool_iteration += 1
                    print(f"  [Tool Iteration {tool_iteration}/{max_tool_iterations}]")
                    
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_use_count += 1
                        
                        print(f"    -> Executing {tool_name} with args: {tool_args}")
                        
                        try:
                            result = await asyncio.wait_for(
                                tool_map[tool_name].ainvoke(tool_args), 
                                timeout=180.0
                            )
                            result_str = str(result)
                            if len(result_str) > 50000:
                                result_str = result_str[:50000] + "\n...[Output truncated due to size]..."
                                
                            messages.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))
                        except asyncio.TimeoutError:
                            error_msg = "Error: Tool execution timed out after 3 minutes (180s)."
                            print(f"    [!] {error_msg}")
                            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            print(f"    [!] {error_msg}")
                            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                    
                    try:
                        response = await invoke_with_retry(bound_model, messages, max_tries=3)
                        total_tokens += extract_token_count(response)
                        messages.append(response)
                    except Exception as invoke_err:
                        if "too long" in str(invoke_err).lower() or "status code: 400" in str(invoke_err):
                            print(f"  [!] Context window exceeded during tool loop. Escaping question...")
                            raise ValueError("Error: Context window limit exceeded. Prompt is too large.")
                        raise invoke_err
                
                if tool_iteration >= max_tool_iterations and response.tool_calls:
                    print("  [!] Reached maximum tool iteration limit (10). Forcing model response.")
                    response = await invoke_with_retry(aimodel, messages, max_tries=3)
                    total_tokens += extract_token_count(response)
                    messages.append(response)
                    
                answer = response.content
                print(f"AI Answer: {answer}")
                
            except Exception as inner_e:
                answer = f"Error processing question: {str(inner_e)}"
                print(f"  [!] {answer}")
            
            results.append({
                "question": question, 
                "answer": answer, 
                "token_count": total_tokens, 
                "tool_use": tool_use_count
            })

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl + C). Saving current progress...")

    if results:
        fieldnames = ["question", "answer", "token_count", "tool_use"]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nResults successfully saved to '{output_csv}' ({len(results)}/{len(questions)} questions answered).")
    else:
        print("\nNo results were collected to save.")


async def main():
    output_dir = Path("outputs_bf_gpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for repo in REPOS:
        output_file = output_dir / f"{repo}_answers.csv"
        print(f"\n==========================================")
        print(f"Processing repository: {repo}")
        print(f"==========================================")
        await run_agent_batch(repo, QUESTIONS, str(output_file))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully.")