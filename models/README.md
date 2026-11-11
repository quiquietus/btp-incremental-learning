# models/

Empty until the Kaggle notebook has run. Install its results with

    python scripts/install_outputs.py path/to/outputs.zip

which fills this folder with `global.npz`, `fedadam_state.npz`, `personal_{1,2,3}.npz`,
`central.npz`, `fisher.npz`, `config.json`, `fl_history.json` and `node_data/`.
Without them the system still starts, using a randomly initialised placeholder model
(the pages show a "placeholder model" badge).
