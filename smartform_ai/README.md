# SmartForm AI MVP 🚀

Build minimal formwork sets, detect element repetitions, auto-generate BoQ, and simulate project timelines—all from standard schedule and structural element data.

## Features
- **Repetition Detection Module**: Groups structural elements by type and dimensions.
- **Optimization Module (PuLP)**: Linear programming reduces procurement needs based on the formwork reuse cycle.
- **Dynamic BoQ Generator**: Auto-calculates Formwork Areas based on column/slab/beam dimensions.
- **Inventory Simulation**: Daily tracking of usage vs available inventory.

## How to Run Locally

1. Setup and Install packages:
   ```bash
   cd d:\mahasangram\createch001\smartform_ai
   pip install -r requirements.txt
   ```
   
2. Generate mock data for testing (optional, fallback if no upload provided in the UI):
   ```bash
   python generate_mock_data.py
   ```
   
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- Python
- Streamlit (UI)
- Pandas (Data)
- PuLP (Optimization)
- Matplotlib (Viz)
