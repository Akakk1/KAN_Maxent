# Data

## In this package (tracked)

### `nceas/` — primary benchmark

Six regions (AWT, CAN, NSW, NZ, SA, SWI) with PO, PA, environment tables, and 50k background points derived from the NCEAS / Valavi benchmark layout.

| Region | Typical files |
|:-------|:--------------|
| each | `{REG}_po.csv`, `{REG}_pa*.csv`, `{REG}_env*.csv`, `{REG}_bg_50k.csv`, `MANIFEST.json` |

**Protocol:** train on PO + random 50k BG; evaluate on independent PA; never use PA for model selection.

**Skip rule:** species with n_PO &lt; 5 are skipped (recorded in metrics).

**Cite:** Elith et al. (2020) *Biodiversity Informatics*; Valavi et al. (2021/2022) *Ecological Monographs* NCEAS reanalysis and DataS1.

### `ginkgo/` — optional legacy

Single-species table for historical spatial-CV experiments. **Not required** for main NCEAS claims or for the default smoke script’s scientific narrative (tests may still reference it if present).

## Not in this package (download if needed)

| Asset | Purpose | Notes |
|:------|:--------|:------|
| Valavi DataS1 zip / `background_50k` originals | Provenance / re-extract | Large; lab backup only |
| OSF MaxNet prediction CSVs | Pipeline alignment checks | Optional `compare_valavi_maxnet.py` |
| WorldClim rasters | Ginkgo maps | Not used for NCEAS tables |

## Integrity

After clone, NCEAS tables should load via:

```bash
python -c "from kanmaxent.data.nceas_io import load_region; print(load_region('CAN').region)"
```

See also `docs/data_manifest.md`.
