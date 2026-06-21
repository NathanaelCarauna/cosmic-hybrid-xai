# Models

Author's own models for COSMIC functional size estimation.

## Structure

Each model has its own directory with configuration, pipeline code, and a runner:

```
models/
  hybrid-xai/           # Dissertation's hybrid + explainable AI approach
    __init__.py
    config.yaml         # Experiment parameters (LLM, prompts, features)
    pipeline.py         # Core model logic
```

## Adding / Developing a Model

1. **Create model directory**: `models/<model-name>/`

2. **Define configuration** in `config.yaml`:
   ```yaml
   method_name: my-model
   llm_model: gpt-4o
   prompts_version: v1
   features: [requirement_text, domain_context]
   hyperparameters:
     temperature: 0
   ```

3. **Implement the pipeline** in `pipeline.py`:
   - Must expose an `analyze(uc_name, uc_content, model) -> dict` method
   - Return dict must follow the BaseRunner contract (see `shared/base_runner.py`)

4. **Create a runner** in `scripts/<model>_runner.py`:
   ```python
   from shared.base_runner import BaseRunner

   class MyModelRunner(BaseRunner):
       def process_fp(self, fp_file):
           # call your pipeline here
           ...
   ```

5. **Run experiments**:
   ```bash
   python scripts/<model>_runner.py <input_dir> \
       --dataset <name> --results-dir results/
   ```

6. Results go to `results/<model-name>/<run_id>/` with the same format as replications.

## Current Models

| Name        | Description                          | Status      |
|-------------|--------------------------------------|-------------|
| hybrid-xai  | Hybrid + Explainable AI (MSc thesis) | Placeholder |
