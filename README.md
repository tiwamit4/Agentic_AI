# Agentic AI

This repository is a hands-on learning path for building agentic AI workflows with
LangGraph, LangChain, Groq-hosted LLMs, tools, memory, RAG, MCP, human-in-the-loop
approval, and subgraphs. The examples start with simple deterministic graphs and
gradually add LLM calls, branching, evaluation loops, tool execution, persistence,
retrieval, and multi-graph composition.

## What This Repository Covers

- LangGraph basics: `StateGraph`, `START`, `END`, nodes, edges, and typed state.
- Workflow state design with `TypedDict`, reducers, and message accumulation.
- Deterministic workflows for calculations and routing.
- LLM-powered workflows using `ChatGroq`.
- Prompt chaining across multiple LLM calls.
- Parallel branches that merge into a final summary or evaluation.
- Conditional edges for branching based on state.
- Iterative self-improvement loops with evaluator and optimizer nodes.
- Chatbot state with LangChain message objects.
- Short-term memory and checkpointing with LangGraph checkpointers.
- Tool calling with `ToolNode`, `tools_condition`, and `@tool`.
- External APIs through custom tools.
- Model Context Protocol integration with `MultiServerMCPClient`.
- Retrieval-augmented generation over a PDF using embeddings and FAISS.
- Human-in-the-loop interruption and resume flows.
- Subgraphs and shared state between parent and child graphs.

## Project Structure

```text
.
|-- 1_bmi_workflow.ipynb
|-- 2_llm_workflow.ipynb
|-- 3_prompt_chaining.ipynb
|-- 4_batsman_workflow.ipynb
|-- 5_UPSC_essay_workflow.ipynb
|-- 6_quadratic_equation_workflow.ipynb
|-- 7_review_reply_workflow.ipynb
|-- 8_X_post_generator.ipynb
|-- 9_basic_chatbot.ipynb
|-- 10_persistance.ipynb
|-- 11_tools.ipynb
|-- 12_mcp.py
|-- 13_rag.ipynb
|-- 14_hitl.ipynb
|-- 15_subgraphs.ipynb
|-- 15_subgraph_shared.ipynb
|-- chatbot_without_hitl.py
|-- chatbot_with_hitl.py
|-- intro-to-ml.pdf
|-- requirements.txt
`-- agentic_ai/
```

Note: `agentic_ai/` appears to be a local Python virtual environment rather than
application source code.

## File-by-File Concept Map

| File | Main concept | What it demonstrates |
| --- | --- | --- |
| `1_bmi_workflow.ipynb` | Basic LangGraph workflow | Builds a BMI calculator with typed state, a calculation node, a label node, and linear graph execution. |
| `2_llm_workflow.ipynb` | Single LLM node | Wraps a Groq LLM call inside a LangGraph node for question answering. |
| `3_prompt_chaining.ipynb` | Prompt chaining | Uses multiple LLM calls to create an outline, write a blog, and evaluate or improve generated content. |
| `4_batsman_workflow.ipynb` | Parallel branches | Calculates strike rate, balls per boundary, and boundary percentage in parallel, then combines them in a summary. |
| `5_UPSC_essay_workflow.ipynb` | Parallel LLM evaluation | Evaluates an essay across language, analysis, and depth of thought, then merges results into a final score. |
| `6_quadratic_equation_workflow.ipynb` | Conditional routing | Computes the discriminant and routes to real roots, repeated roots, or no real roots. |
| `7_review_reply_workflow.ipynb` | Sentiment-based routing | Detects review sentiment and routes positive reviews directly to a reply while diagnosing negative reviews before responding. |
| `8_X_post_generator.ipynb` | Iterative agent loop | Generates a post, evaluates it, and loops through optimization until approved. |
| `9_basic_chatbot.ipynb` | Chatbot graph | Uses LangChain message objects, `add_messages`, and memory to maintain chat state. |
| `10_persistance.ipynb` | Persistence and checkpoints | Uses in-memory checkpointing, thread IDs, state inspection, state replay, and crash recovery patterns. |
| `11_tools.ipynb` | Tool calling | Adds calculator, search, stock lookup, and random/custom tools through LangGraph's prebuilt tool node. |
| `12_mcp.py` | MCP tools | Connects a LangGraph chatbot to local and remote MCP servers and binds discovered tools to the LLM. |
| `13_rag.ipynb` | Retrieval-augmented generation | Loads `intro-to-ml.pdf`, splits text, embeds chunks, stores vectors in FAISS, and exposes retrieval as a tool. |
| `14_hitl.ipynb` | Human in the loop | Uses `interrupt` and `Command(resume=...)` to pause graph execution and resume after human approval. |
| `15_subgraphs.ipynb` | Subgraphs | Composes parent and child graphs where translation and answer generation are separated into reusable graph units. |
| `15_subgraph_shared.ipynb` | Shared-state subgraphs | Demonstrates parent and subgraph coordination when state keys are shared. |
| `chatbot_without_hitl.py` | Tool-using stock chatbot | CLI chatbot that can fetch stock prices and simulate stock purchases without human approval. |
| `chatbot_with_hitl.py` | HITL stock chatbot | CLI chatbot that pauses before a stock purchase and asks a human to approve or reject the action. |

## Core Concepts

### 1. LangGraph State

Most examples define a `TypedDict` state object. Nodes receive the current state
and return a partial state update.

Common patterns used in this repo:

- Simple scalar state, such as height, weight, BMI, and labels.
- LLM state, such as question and answer.
- Message state with `Annotated[list[BaseMessage], add_messages]`.
- Aggregated evaluation state using reducers such as `operator.add`.

### 2. Nodes and Edges

Each workflow is built by adding named nodes and connecting them with edges:

```python
graph = StateGraph(MyState)
graph.add_node("node_name", node_function)
graph.add_edge(START, "node_name")
graph.add_edge("node_name", END)
workflow = graph.compile()
```

The early notebooks use linear flows. Later notebooks use parallel edges,
conditional routing, loops, subgraphs, tools, and checkpoints.

### 3. LLM Workflows

The LLM examples use:

```python
llm = ChatGroq(model="llama-3.3-70b-versatile")
```

The model is called inside graph nodes to produce answers, outlines, essays,
reviews, summaries, post drafts, and structured evaluations.

### 4. Prompt Chaining

Prompt chaining appears when one LLM output becomes the next LLM input. For
example:

1. Generate a blog outline.
2. Use the outline to write the blog.
3. Evaluate the blog.
4. Improve it if needed.

This pattern is useful when a task is too large or nuanced for a single prompt.

### 5. Parallel Execution

Several workflows fan out from `START` into multiple nodes and then merge into a
final node. This is used for:

- Cricket batting metric calculations.
- UPSC essay evaluation dimensions.
- Independent analysis tasks that can be combined later.

### 6. Conditional Routing

Conditional edges route execution based on state. Examples include:

- Quadratic equations: route based on discriminant value.
- Review replies: route based on sentiment.
- X post generation: route based on evaluation approval.

### 7. Agentic Loops

The X post generator demonstrates a simple agentic loop:

```text
generate -> evaluate -> optimize -> evaluate -> ... -> END
```

This pattern is useful when the system needs to critique and refine its own
outputs before returning a final answer.

### 8. Memory and Persistence

The persistence notebook demonstrates:

- Checkpointers.
- Thread-specific graph state.
- Inspecting current state.
- Replaying from old checkpoints.
- Updating stored state.
- Recovering after interrupted or failed execution.

The chatbot examples use thread IDs to keep conversation state separate between
runs.

### 9. Tools

Tool examples use LangChain's `@tool` decorator and LangGraph's prebuilt
`ToolNode`.

Tools shown in the repo include:

- Calculator.
- DuckDuckGo search.
- Stock price lookup.
- Mock stock purchase.
- RAG retrieval tool.
- MCP-provided tools.

The common graph shape is:

```text
chat_node -> tools -> chat_node
```

`tools_condition` decides whether the assistant should call a tool or finish
with a normal response.

### 10. RAG

The RAG notebook builds a retrieval pipeline over `intro-to-ml.pdf`:

1. Load the PDF with `PyPDFLoader`.
2. Split pages into chunks with `RecursiveCharacterTextSplitter`.
3. Embed chunks with Hugging Face embeddings.
4. Store vectors in FAISS.
5. Retrieve relevant chunks for a query.
6. Expose retrieval as a LangGraph tool.
7. Let the LLM answer with retrieved context.

### 11. MCP

`12_mcp.py` uses `MultiServerMCPClient` to connect LangGraph to external MCP
servers. It demonstrates two styles:

- Local stdio MCP server.
- Remote streamable HTTP MCP server.

The discovered MCP tools are bound to the LLM and executed through `ToolNode`.

### 12. Human in the Loop

Human-in-the-loop examples use:

```python
decision = interrupt("Approve this action?")
```

The graph pauses and later resumes with:

```python
Command(resume=decision)
```

This is especially useful for high-impact actions such as purchases, approvals,
publishing, deletion, or sending messages.

### 13. Subgraphs

The subgraph notebooks demonstrate how larger workflows can be composed from
smaller graphs. This keeps complex applications easier to reason about and makes
common logic reusable.

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If you run `12_mcp.py`, you may also need the MCP adapter package:

```powershell
pip install langchain-mcp-adapters
```

### 3. Configure environment variables

Create a `.env` file in the project root. At minimum, the Groq examples need a
Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Some examples also use external services such as Alpha Vantage for stock prices.
Store those keys in `.env` rather than hardcoding them in source files.

## Running the Examples

### Run notebooks

Open the notebooks in Jupyter, VS Code, or another notebook environment:

```powershell
jupyter notebook
```

Then execute the notebooks in order if you want the intended learning path.

### Run the stock chatbot without HITL

```powershell
python chatbot_without_hitl.py
```

This starts a CLI chatbot that can call stock-related tools directly.

### Run the stock chatbot with HITL

```powershell
python chatbot_with_hitl.py
```

This starts a CLI chatbot that pauses before a simulated stock purchase and asks
for human approval.

### Run the MCP example

```powershell
python 12_mcp.py
```

Before running it, update the local MCP server path in `12_mcp.py` if needed.
The current example path is machine-specific.

## Suggested Learning Order

1. `1_bmi_workflow.ipynb`
2. `2_llm_workflow.ipynb`
3. `3_prompt_chaining.ipynb`
4. `4_batsman_workflow.ipynb`
5. `5_UPSC_essay_workflow.ipynb`
6. `6_quadratic_equation_workflow.ipynb`
7. `7_review_reply_workflow.ipynb`
8. `8_X_post_generator.ipynb`
9. `9_basic_chatbot.ipynb`
10. `10_persistance.ipynb`
11. `11_tools.ipynb`
12. `12_mcp.py`
13. `13_rag.ipynb`
14. `14_hitl.ipynb`
15. `15_subgraphs.ipynb`
16. `15_subgraph_shared.ipynb`

## Dependencies

Important libraries used across the repo:

- `langgraph`
- `langchain`
- `langchain-core`
- `langchain-groq`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-huggingface`
- `langgraph-checkpoint-sqlite`
- `python-dotenv`
- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf`
- `requests`
- `ddgs`

See `requirements.txt` for pinned versions.

## Notes and Caveats

- The folder name in the request appears as `AGENRIC_AI`, but this repository is
  located at `Agentic_AI`.
- `10_persistance.ipynb` is spelled with "persistance" in the filename; the
  concept is persistence/checkpointing.
- `agentic_ai/` is a virtual environment and should usually not be committed as
  project source.
- Do not commit real API keys. Keep secrets in `.env` and ensure `.env` remains
  ignored by Git.
- Some notebooks depend on live LLM/API calls, so outputs may vary between runs.
