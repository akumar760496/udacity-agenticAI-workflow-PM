# Instructions to Run the Project

## 1. Open the project directory

Open a terminal and move to the project root:

```bash
cd /Users/ajay/Documents/udacity_agenticai_project/PROJECT
```

## 2. Create or activate the virtual environment

If the virtual environment does not exist, create it:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

## 3. Install project dependencies

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file in the directory from which the script loads environment
variables. For the Phase 1 and Phase 2 scripts, place the file in the project
root:

```text
PROJECT/.env
```

Add the API key:

```env
OPENAI_API_KEY=your_api_key
```

For the Udacity Vocareum environment, use the provided Vocareum API key. The
agent clients use the Vocareum endpoint:

```text
https://openai.vocareum.com/v1
```

Do not commit `.env` or any API key to source control.

## 5. Check the Python environment

Verify that the required packages are available:

```bash
python -c "import openai, dotenv, numpy, pandas; print('Dependencies are available')"
```

## 6. Run Phase 1 scripts

Phase 1 contains one script for each agent. Run the scripts from the Phase 1
directory:

```bash
cd starter/phase_1
```

Run the direct prompt agent:

```bash
python direct_prompt_agent.py
```

Run the augmented prompt agent:

```bash
python augmented_prompt_agent.py
```

Run the knowledge-augmented prompt agent:

```bash
python knowledge_augmented_prompt_agent.py
```

Run the RAG knowledge prompt agent:

```bash
python rag_knowledge_prompt_agent.py
```

Run the evaluation agent:

```bash
python evaluation_agent.py
```

Run the routing agent:

```bash
python routing_agent.py
```

Run the action-planning agent:

```bash
python action_planning_agent.py
```

The RAG script creates chunk and embedding CSV files while it runs. These are
runtime outputs and should not be committed unless explicitly required.

## 7. Validate Phase 1 syntax

From the Phase 1 directory, run:

```bash
python -m compileall -q .
```

To check that no TODO markers remain in the Phase 1 implementation:

```bash
grep -RIn "TODO" --exclude-dir="__pycache__" .
```

## 8. Run Phase 2

Return to the project root:

```bash
cd /Users/ajay/Documents/udacity_agenticai_project/PROJECT
```

Run the project-management workflow:

```bash
python starter/phase_2/agentic_workflow.py
```

The workflow:

1. Loads `Product-Spec-Email-Router.txt`.
2. Extracts workflow steps with the action-planning agent.
3. Routes each step to the Product Manager, Program Manager, or Development
   Engineer agent.
4. Evaluates each generated response.
5. Prints a consolidated project plan containing:
   - Product Manager user stories.
   - Program Manager product features.
   - Development Engineer engineering tasks.

The product specification path is resolved relative to
`agentic_workflow.py`, so Phase 2 can be launched from the project root.

## 9. Validate Phase 2 syntax

From the project root:

```bash
python -m compileall -q starter/phase_2
```

## 10. Run all Phase 1 scripts sequentially

From the Phase 1 directory:

```bash
for script in \
  direct_prompt_agent.py \
  augmented_prompt_agent.py \
  knowledge_augmented_prompt_agent.py \
  rag_knowledge_prompt_agent.py \
  evaluation_agent.py \
  routing_agent.py \
  action_planning_agent.py
do
  echo "===== Running $script ====="
  python "$script" || exit 1
done
```

## 11. Common troubleshooting

### `ModuleNotFoundError`

Confirm that the virtual environment is activated and reinstall dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### `OPENAI_API_KEY` is missing

Confirm that `.env` exists in the project root and contains:

```env
OPENAI_API_KEY=your_api_key
```

### `401 invalid_api_key`

Confirm that the key matches the configured API endpoint. Vocareum keys must
use the Vocareum endpoint configured in the agent library rather than the
default OpenAI endpoint.

### `FileNotFoundError` for the product specification

Run the Phase 2 workflow from the project root:

```bash
python starter/phase_2/agentic_workflow.py
```

The required file must exist at:

```text
starter/phase_2/Product-Spec-Email-Router.txt
```

### `429` or quota errors

Check the available API quota and rate limits, then retry after confirming the
API key and endpoint configuration.
