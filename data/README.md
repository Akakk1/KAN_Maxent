# Data

**`data/nceas/` is not MIT.** It remains [CC BY-NC 4.0](nceas/LICENSE)
(Elith et al. 2020). See also [`LICENSE`](LICENSE) in this folder.

## In this package (tracked)

### `nceas/` — primary benchmark

Six regions (AWT, CAN, NSW, NZ, SA, SWI) with PO, PA, environment tables, and 50k background points derived from the NCEAS / Valavi benchmark layout.

| Region | Typical files |
|:-------|:--------------|
| each | `{REG}_po.csv`, `{REG}_pa*.csv`, `{REG}_env*.csv`, `{REG}_bg_50k.csv`, `MANIFEST.json` |

**Protocol:** train on PO + random 50k BG; evaluate on independent PA; never use PA for model selection.

**Skip rule:** species with n_PO &lt; 5 are skipped (recorded in metrics).

**Licence:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(Elith et al. 2020). Non-commercial use only; keep attribution.

**Cite:** Elith et al. (2020) *Biodiversity Informatics*,
doi:10.17161/bi.v15i2.13384; Valavi et al. (2022) *Ecological Monographs*.

Ginkgo occurrence tables are **not** shipped (not used for the NCEAS
analyses in this package).

## Not in this package (download if needed)

| Asset | Purpose | Notes |
|:------|:--------|:------|
| Valavi DataS1 zip / `background_50k` originals | Provenance / re-extract | Large; lab backup only |
| OSF MaxNet prediction CSVs | Pipeline alignment checks | Optional `compare_valavi_maxnet.py` |

## Integrity

After clone, NCEAS tables should load via:

```bash
python -c "from kanmaxent.data.nceas_io import load_region; print(load_region('CAN').region)"
```

See also `docs/data_manifest.md`.
