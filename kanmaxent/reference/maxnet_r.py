"""R maxnet bridge via subprocess (raw env, regmult=1.0)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

R_SCRIPT = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly=TRUE)
    train_csv <- args[[1]]
    test_csv  <- args[[2]]
    out_json  <- args[[3]]
    regmult   <- as.numeric(args[[4]])

    suppressPackageStartupMessages({
      library(maxnet)
      library(jsonlite)
    })

    train <- read.csv(train_csv, check.names=FALSE)
    test  <- read.csv(test_csv, check.names=FALSE)
    non_env <- c("label", "fold", "decimalLatitude", "decimalLongitude")
    env_cols <- setdiff(names(train), non_env)
    p <- train$label
    env_tr <- train[, env_cols, drop=FALSE]
    env_te <- test[, env_cols, drop=FALSE]

    ok <- TRUE
    err <- ""
    link <- rep(NA_real_, nrow(test))
    cloglog <- rep(NA_real_, nrow(test))
    n_betas <- NA_integer_

    tryCatch({
      glmnet::glmnet.control(maxit=10000)
      model <- maxnet(p = p, data = env_tr, regmult = regmult)
      link <- as.numeric(predict(model, env_te, type = "link"))
      cloglog <- as.numeric(predict(model, env_te, type = "cloglog"))
      n_betas <- length(model$betas)
    }, error = function(e) {
      ok <<- FALSE
      err <<- conditionMessage(e)
    })

    ver <- as.character(packageVersion("maxnet"))
    payload <- list(
      ok = ok,
      error = err,
      maxnet_version = ver,
      n_betas = n_betas,
      link = link,
      cloglog = cloglog
    )
    write_json(payload, out_json, auto_unbox=TRUE, digits=NA)
    """
)


def fit_predict_maxnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    env_names: list,
    *,
    regmult: float = 1.0,
    rscript: str = "Rscript",
    timeout_s: int = 600,
) -> Tuple[np.ndarray, Dict]:
    """Fit maxnet on train (raw env) and predict link scores on test.

    Returns
    -------
    scores : ndarray
        type='link' predictions (length n_test); NaN if failed.
    info : dict
        ok, error, maxnet_version, n_betas, cloglog optional.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="maxnet_") as td:
        td = Path(td)
        train_df = pd.DataFrame(X_train, columns=env_names)
        train_df["label"] = y_train
        test_df = pd.DataFrame(X_test, columns=env_names)
        test_df["label"] = 0  # unused
        train_csv = td / "train.csv"
        test_csv = td / "test.csv"
        out_json = td / "out.json"
        script = td / "run_maxnet.R"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        script.write_text(R_SCRIPT, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    rscript,
                    str(script),
                    str(train_csv),
                    str(test_csv),
                    str(out_json),
                    str(regmult),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            n = X_test.shape[0]
            return np.full(n, np.nan), {
                "ok": False,
                "error": str(e),
                "maxnet_version": None,
                "n_betas": None,
                "stderr": "",
            }

        if not out_json.is_file():
            n = X_test.shape[0]
            return np.full(n, np.nan), {
                "ok": False,
                "error": f"no output json; stderr={proc.stderr[:500]}",
                "maxnet_version": None,
                "n_betas": None,
                "returncode": proc.returncode,
            }

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        link = np.asarray(payload.get("link", []), dtype=np.float64)
        if link.size != X_test.shape[0]:
            link = np.full(X_test.shape[0], np.nan)
        info = {
            "ok": bool(payload.get("ok", False)),
            "error": payload.get("error", ""),
            "maxnet_version": payload.get("maxnet_version"),
            "n_betas": payload.get("n_betas"),
            "cloglog": np.asarray(payload.get("cloglog", []), dtype=np.float64),
            "stderr": proc.stderr[-500:] if proc.stderr else "",
        }
        return link, info
