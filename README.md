# Infosys-internship_Aneesh-Jantikar

Author: Aneesh Jantikar (aneeshj-05)  
Last updated: 2026-01-13

## Project overview

This repository collects milestone notebooks created during the Infosys internship. The work spans two main areas:

- Deep learning **from-scratch** experiments implemented using NumPy and lightweight training loops (Milestone 1).
- Web scraping and a small data‑analysis/enrichment pipeline for the "Books to Scrape" demo site, implemented with Playwright and asynchronous scraping (Milestones 2–4). Milestone 3 includes a pricing simulation; Milestone 4 extends the pipeline with enrichment (ISBN/price lookups).

Each notebook is runnable in Google Colab or a local environment (with some system deps for Playwright). The notebooks store outputs to `output/` where applicable.

---

## Table of contents

- Quick start
- Requirements & environment
- How to run each notebook
- Notebook-by-notebook detailed description
  - MIlestone1.ipynb
  - MIlestone2.ipynb
  - Milestone3.ipynb
  - MIlestone4.ipynb
- Artifacts / output files
- Notes, runtime expectations & tips
- License & contact

---

## Quick start

1. Clone the repository:
   ```
   git clone https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar.git
   cd Infosys-internship_Aneesh-Jantikar
   ```

2. Open the notebook you want to run in Google Colab (recommended for Playwright setup) or locally in Jupyter.

3. For Playwright-based notebooks, run the install cells (they call `pip install playwright nest_asyncio` and `playwright install chromium`) and install the OS dependencies as required (the notebooks include apt-get lines when run in Colab).

---

## Requirements & environment

- Python 3.8+ (tested in Colab / Python 3.12 in some runs)
- Major Python libraries used across notebooks:
  - numpy, matplotlib, scikit-learn
  - tensorflow (tf.keras) for loading MNIST
  - pandas (used implicitly for display in some notebooks)
  - playwright, nest_asyncio, asyncio
  - playwright browser: Chromium (installed by `playwright install chromium`)
- System packages (for Playwright/Chromium when running locally or in certain hosted environments):
  - libatk1.0-0, libatk-bridge2.0-0, libatspi2.0-0, libxcomposite1, libxtst6, and others (the notebooks use `apt-get` to install these on Colab).
- Recommended: Use Google Colab (free) for convenience; it runs the Playwright install steps in the notebooks and provides adequate compute for the NumPy DNN training demonstration.

---

## How to run each notebook

- Open the notebook in Colab or Jupyter.
- Execute cells sequentially; the notebooks include setup cells for dependencies.
- Playwright notebooks: run the Playwright install cells first (they run `!pip install playwright nest_asyncio` and `!playwright install chromium`), then execute the scraper cell(s). On Colab the apt-get system installs are included.
- Milestone 1 (DNN from scratch): run top-to-bottom. Training a pure-NumPy 4-layer network is computationally expensive and can take many minutes/hours depending on training epochs and CPU (the notebook uses MNIST and prints epoch-by-epoch stats).
- Milestone 2/3/4 (scrapers & enrichment): run the Playwright setup cells first. The scrapers are asynchronous and will save CSV/JSON outputs under `output/`.

---

## Notebook-by-notebook detailed description

### 1) MIlestone1.ipynb
URL: https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar/blob/main/MIlestone1.ipynb

Purpose
- Implement and demonstrate a 4-layer fully connected neural network implemented from scratch using NumPy.
- Train and evaluate on the MNIST handwritten digits dataset.
- Provide a clear didactic implementation of forward pass, backward pass (backpropagation), and training loop.

Main components
- Data loading: uses `tensorflow.keras.datasets.mnist` to load MNIST (flattened to 784-dim inputs) and `to_categorical` for one-hot labels.
- DeepNeuralNetwork class:
  - Initialization: Xavier/He-style random initialization for weights (scaled).
  - Activation functions: `sigmoid()` and a numerically stabilized `softmax()`.
  - Loss: binary cross-entropy (with clipping for numerical stability), and a derivative calculation used during backprop.
  - Forward pass: computes activations for a 4-layer network (input → hidden1 → hidden2 → output).
  - Backward pass: computes weight updates via manual backpropagation using chain rule and outer-product updates.
  - `update_network_parameters`: simple SGD update rule.
  - `compute_accuracy`: iterates over validation samples and computes prediction accuracy.
  - `train`: training loop that iterates epochs and samples, calls forward/backward/update, computes and prints train/validation accuracy & loss, stores history lists, and plots loss/accuracy curves.
- Example training run included:
  - `dnn1 = DeepNeuralNetwork(sizes=[784, 128, 64, 10], epochs=80, l_rate=0.05)`
  - The notebook prints epoch-by-epoch accuracies and losses.
  - The sample training output shows training accuracies approaching high 90s (e.g., training acc ~98%, validation ~95% in one logged run) for long training schedules.
- Outputs:
  - The notebook uses `np.save('Data', self)` at the end to store the DNN object (so a file named `Data.npy` may be created when running).
  - Plots for training/validation loss and accuracy (matplotlib figures displayed inline).

Runtime and notes
- Training the pure-NumPy DNN is CPU-bound and can be slow for large epochs and dataset sizes. The example training printed very long runs (many minutes/hours).
- Numerical stability warnings may appear from exponentials, divisions; the implementation includes clipping and stable softmax but manual implementations can produce runtime warnings in some libraries/environments.
- The notebook is educational — it intentionally performs operations in NumPy rather than using a DL framework.

Suggestions / improvements
- Use mini-batches rather than single-sample updates to accelerate training and stabilize gradients.
- Replace manual softmax derivative/backprop quirks with cross-entropy + softmax combined gradient for numerical stability.
- Add checkpoint saving of weights and a small inference wrapper function.

---

### 2) MIlestone2.ipynb
URL: https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar/blob/main/MIlestone2.ipynb

Purpose
- Asynchronous web scraping demo: scrape the example site https://books.toscrape.com/ using Playwright in headless Chromium.
- Collect metadata for books (title, relative/product URL, price text, parsed price, stock availability, rating, UPC, description, image URL, and additional product table fields).

Main components
- Setup cells:
  - `!pip install playwright nest_asyncio`
  - `!playwright install chromium`
  - `!apt-get install ...` system libraries (for Colab / Debian/Ubuntu environments).
  - `nest_asyncio.apply()` to allow running async event loop in notebook.
- Parsing helpers:
  - `parse_price(s)` extracts numeric price floats using regex.
  - `parse_stock(s)` extracts integer stock count using regex.
  - `RATING_MAP` maps textual star-rating classes (`"One"`, `"Two"`, ...) to integers 1–5.
- Scraper function `scrape_all_books()`:
  - Navigates pages starting at base URL; uses `page.query_selector_all("article.product_pod")`.
  - For each book card, extracts title, product relative URL → absolute product_url, price text → numeric price, stock, rating (via class).
  - Opens a new page per product to fetch details from product page: product table rows (UPC, product information), description at selector `#product_description + p`, and image `src` resolution.
  - Appends a dictionary per book to `rows`.
  - Paging loop follows `li.next > a` until no next page.
- Execution:
  - The notebook runs the `scrape_all_books()` coroutine and writes results:
    - `output/books_all.csv`
    - `output/books_all.json`
  - Example run printed page-by-page progress and saved `1000` books (default site has 1000 items).
- Outputs:
  - `output/books_all.csv`
  - `output/books_all.json`

Runtime and notes
- Playwright + Chromium will download the browser binaries during first install. This can take time and requires disk space (the notebook shows the download progress).
- The scraper politely sleeps between page navigations (small delay) and prints progress every 20 books.
- The code is robust to missing description/image or product details by wrapping product page parsing in try/except and continuing.

Suggestions
- Add concurrency control if needed to speed detail-page fetches (careful with politeness and site rules).
- Add retry/backoff on transient navigation failures.

---

### 3) Milestone3.ipynb
URL: https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar/blob/main/Milestone3.ipynb

Purpose
- Extends the scraping work from Milestone2 and demonstrates further data-processing steps.
- Performs a pricing simulation on the scraped books to estimate revenue changes under a small discount scenario.

Main components & behavior observed in outputs
- Setup & scraping: the notebook repeats the Playwright install/setup and runs an equivalent scraper to produce the same `output/books_all.csv` / JSON artifacts (the same `scrape_all_books()` logic appears).
- Pricing simulation and analysis:
  - The notebook builds a DataFrame that contains columns such as:
    - `title`, `price`, `new_price`, `discount_pct`
    - `baseline_monthly_sales`, `expected_new_monthly_sales`
    - `baseline_monthly_revenue`, `new_monthly_revenue`, `delta_revenue`
  - The notebook displays a sample DataFrame of the pricing simulation results (10 sample rows shown in the execution output).
  - The simulation appears to:
    - Apply a discount percentage (e.g., 5%),
    - Estimate how monthly sales change (a simple model: small uplift from discount),
    - Compute baseline and new monthly revenue and delta revenue for each book.
  - The notebook prints messages like "Saved pricing simulation to output/books_pricing_simulation..." (the exact saved filename prefix appears in outputs).
- Outputs:
  - A pricing simulation DataFrame is displayed in the notebook and saved to `output/` (suggested naming: `books_pricing_simulation.csv` / `.json`).

Runtime and notes
- This notebook is meant for exploratory analysis: the simulation is a simple business model demonstrating how a small discount can change revenue.
- The notebook leverages the previously scraped book metadata and performs vectorized/pandas-style calculations (the notebook displays the resulting DataFrame).

Suggestions
- Document or parameterize the demand response model (how `expected_new_monthly_sales` is computed).
- Add plots summarizing total delta revenue, percent of books benefitting from discount, and top/bottom deltas.

---

### 4) MIlestone4.ipynb
URL: https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar/blob/main/MIlestone4.ipynb

Purpose
- Demonstrates a more advanced pipeline that scrapes the Books to Scrape site and then performs an enrichment step by looking up ISBNs and prices on external sources (e.g., BooksRun or similar price-lookup services).
- The pipeline processes scraped books and attempts to match/enrich them with ISBN and price information, logging successes and errors.

Main components & observed behavior
- Scraping phase:
  - The notebook logs progress: "Scraping page 1... page 2... ... page 50..." and prints "Total books scraped: 1000".
- Enrichment phase:
  - For each scraped book, the notebook attempts to find an ISBN and then fetch price information.
  - For many items it prints messages like:
    - `Processing (N/1000): <Title>`
    - On success: `✔ PRICE FOUND → ISBN <isbn>`
    - On some lookups: `BooksRun error for ISBN ...: 'list' object has no attribute 'get'` (indicates parsing/response format mismatch and logged error handling).
  - The enrichment step tries multiple lookups (and prints multiple `✔ PRICE FOUND` messages for books with more than one matched ISBN).
- Outputs:
  - The notebook prints progress to STDOUT and shows that enrichment succeeded for many entries.
  - The notebook likely saves an enriched dataset (e.g., `output/enriched_books.csv` or similar) — the run prints found ISBNs and errors; exact saved filenames should be confirmed by running the notebook (look for save/write lines).

Runtime and notes
- This notebook combines scraping + remote API calls and can be long-running (processing 1000 books and performing external lookups).
- Some errors indicate the external API returned data types not expected by the parsing code (e.g., a `list` where `.get()` was expected). The code catches and logs these errors and continues.
- Running this notebook may require API access/keys depending on which external services are used (the notebook's outputs suggest BooksRun or similar; if an API key is needed, add it to environment variables or notebook variables securely).

Suggestions
- Add retry logic and stricter response validation for the external API calls.
- Add a config cell at the top for API keys and rate-limit settings.
- Save intermediate results frequently to allow resuming.

---

## Artifacts / output files (observed in notebooks)

- `output/books_all.csv` — CSV of scraped books (Milestone2 / Milestone3 / Milestone4).
- `output/books_all.json` — JSON of scraped books.
- Pricing simulation outputs (Milestone3) — likely `output/books_pricing_simulation.csv` or similar (the notebook prints a message indicating savings to `output/`).
- DNN save in Milestone1: `Data.npy` (created by `np.save('Data', self)`), possibly additional saved artifacts if you run the notebook (e.g., `final_weights` commented out).
- Enriched output from Milestone4 (if present): an enriched CSV/JSON with ISBN and price fields (please run the notebook to confirm exact filename).

---

## Notes, runtime expectations & troubleshooting

- Playwright first run: the `playwright install chromium` step downloads binaries (100+ MB) and may take time; the notebooks already include these cells.
- Running the NumPy DNN training on a full MNIST dataset and many epochs is slow on CPU; expect training to take a long time if you run many epochs. Use fewer epochs for experimentation or convert to mini-batch updates.
- When running on Colab:
  - Let the Playwright install cells run to completion.
  - Colab sometimes blocks or requires additional apt packages — the notebooks include apt-get lines that have been used in the runs.
- If the external enrichment API returns unexpected structure (e.g., lists rather than dicts), add defensive parsing and log the raw response for debugging.

---

## Contact & attribution

- Repository: https://github.com/aneeshj-05/Infosys-internship_Aneesh-Jantikar
- Author: Aneesh Jantikar (aneeshj-05)

---

## License

Add a license file as appropriate (MIT, Apache-2.0, etc.). No license file was found in the repository snapshot; consider adding one if you intend the work to be shared.
