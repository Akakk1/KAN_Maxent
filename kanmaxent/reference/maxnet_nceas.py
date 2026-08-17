"""Valavi-style maxnet for NCEAS (factors + optional feature class by n_po)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

R_SCRIPT = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly=TRUE)
    train_csv <- args[[1]]
    test_csv  <- args[[2]]
    out_json  <- args[[3]]
    regmult   <- as.numeric(args[[4]])
    fclasses  <- args[[5]]
    cat_json  <- args[[6]]

    suppressPackageStartupMessages({
      library(maxnet)
      library(jsonlite)
    })
    cats <- fromJSON(cat_json)

    train <- read.csv(train_csv, check.names=FALSE, stringsAsFactors=FALSE)
    test  <- read.csv(test_csv, check.names=FALSE, stringsAsFactors=FALSE)
    for (c in cats) {
      if (c %in% names(train)) {
        train[[c]] <- as.factor(as.character(train[[c]]))
        test[[c]]  <- as.factor(as.character(test[[c]]))
        # expand test levels to train levels
        test[[c]] <- factor(as.character(test[[c]]), levels=levels(train[[c]]))
      }
    }

    resp <- train$occ
    preds <- train[, setdiff(names(train), c("occ")), drop=FALSE]
    te <- test[, setdiff(names(test), c("occ")), drop=FALSE]

    ok <- TRUE
    err <- ""
    used_classes <- fclasses
    link <- rep(NA_real_, nrow(te))
    n_betas <- NA_integer_
    glmnet::glmnet.control(maxit=200000)
    fit_one <- function(cls) {
      maxnet::maxnet(
        p = resp,
        data = preds,
        f = maxnet::maxnet.formula(p = resp, data = preds, classes = cls),
        regmult = regmult
      )
    }
    mod <- NULL
    for (cls in unique(c(fclasses, "l", "lq"))) {
      tryCatch({
        mod <<- fit_one(cls)
        used_classes <<- cls
      }, error = function(e) {
        err <<- conditionMessage(e)
        mod <<- NULL
      })
      if (!is.null(mod)) break
    }
    if (is.null(mod)) {
      ok <- FALSE
    } else {
      link <- as.numeric(predict(mod, te, type = "link"))
      n_betas <- length(mod$betas)
    }
    write_json(list(
      ok=ok, error=err,
      maxnet_version=as.character(packageVersion("maxnet")),
      n_betas=n_betas, link=link, classes=used_classes
    ), out_json, auto_unbox=TRUE, digits=NA, na="null")
    """
)


def fit_predict_maxnet_nceas(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    covars: Sequence[str],
    categorical: Sequence[str],
    *,
    n_presence: int,
    regmult: float = 1.0,
    rscript: str = "Rscript",
    timeout_s: int = 900,
) -> Tuple[np.ndarray, Dict]:
    """Fit maxnet on train_df[occ+covars]; predict link on test_df[covars]."""
    fclasses = "l" if n_presence <= 10 else "lq"
    cols = list(covars)
    tr = train_df[cols + ["occ"]].copy()
    te = test_df[cols].copy()
    te["occ"] = 0

    with tempfile.TemporaryDirectory(prefix="maxnet_nceas_") as td:
        td = Path(td)
        tr_path, te_path = td / "train.csv", td / "test.csv"
        out_json, script = td / "out.json", td / "run.R"
        tr.to_csv(tr_path, index=False)
        te.to_csv(te_path, index=False)
        script.write_text(R_SCRIPT, encoding="utf-8")
        cat_json = json.dumps(list(categorical))
        try:
            proc = subprocess.run(
                [
                    rscript,
                    str(script),
                    str(tr_path),
                    str(te_path),
                    str(out_json),
                    str(regmult),
                    fclasses,
                    cat_json,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except Exception as e:
            return np.full(len(te), np.nan), {"ok": False, "error": str(e)}

        if not out_json.is_file():
            return np.full(len(te), np.nan), {
                "ok": False,
                "error": proc.stderr[-800:] if proc.stderr else "no json",
                "returncode": proc.returncode,
            }
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        raw_link = payload.get("link", [])
        link = np.array(
            [np.nan if (v is None or v == "NA" or v == "NaN") else float(v) for v in raw_link],
            dtype=np.float64,
        )
        if link.size != len(te):
            link = np.full(len(te), np.nan)
        ok = bool(payload.get("ok", False)) and bool(np.isfinite(link).any())
        return link, {
            "ok": ok,
            "error": payload.get("error", ""),
            "maxnet_version": payload.get("maxnet_version"),
            "n_betas": payload.get("n_betas"),
            "classes": payload.get("classes", fclasses),
            "stderr": (proc.stderr or "")[-400:],
        }

