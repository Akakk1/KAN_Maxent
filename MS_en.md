## Do deep interactions in Kolmogorov–Arnold networks improve presence-only species distribution models?

**Manuscript version v7.2.** Supporting-information pair: live `SI/SI.md` + `SI/figures/` (frozen copy: `SI/versions/v7.2/`). Plotting script: `benchmarks/plot_ms_closure_v7.2.py`. The earlier **v7** supporting information (four-model/BCE Fig. S1 and the former palette) is retained in `SI/versions/v7/`, with `benchmarks/generate_si_v7.py`.

## Abstract

Kolmogorov–Arnold networks (KANs) provide learnable univariate functions on edges with layered compositions that can represent covariate interactions. We evaluated whether these components improve single-species distribution modelling from presence-only records. Across 225 species from six NCEAS regions, we compared end-to-end standard KANs, additive B-spline IPP, same-basis generalised additive models (GAMs), maxnet, and residual KAN interaction models using independent presence–absence evaluation data. End-to-end KAN achieved a mean area under the receiver operating characteristic curve (AUC) of 0.642 and was 0.072 AUC units below maxnet. Additive B-spline models closely matched same-basis generalised additive models (species-level AUC correlation: 0.9999) and exceeded maxnet by 0.0089 AUC units on average. In contrast, adding residual KAN interactions reduced AUC by 0.0048–0.0390, with larger reductions in deeper models. Thus, the improvement of residual models over end-to-end KAN arose from retaining additive main effects, not from learned interactions. Learnable additive functions offer an interpretable alternative alongside maxnet and generalised additive models, whereas multilayer KANs and residual deep interactions are not supported as default models.

**Keywords:** species distribution modelling; Kolmogorov–Arnold networks; MaxEnt; presence-only; B-splines; model complexity

## Highlights

- Additive B-spline models matched GAMs and slightly exceeded maxnet.
- End-to-end KAN underperformed maxnet across 225 species in six regions.
- A fair residual protocol separated main-effect retention from interaction gains.
- Deeper residual interactions caused progressively larger AUC losses.

## Introduction

More model complexity does not automatically mean better prediction. Can deep interactions improve presence-only species distribution models (PO-SDMs)? Kolmogorov–Arnold networks (KANs) bring two promises to this question: learnable univariate functions on connection edges, and layered compositions that can represent covariate interactions (Liu and Tegmark, 2025; https://doi.org/10.1103/4t7t-v19l). The first promise is about response shape; the second is about whether interactions learned from presence-only data generalise. The two are independent, and we evaluate them separately.

The question matters because the current benchmark is fixed-shape and the alternatives are hard to read. MaxEnt remains among the most widely used and robust methods for PO records (Elith et al., 2006; https://doi.org/10.1111/j.2006.0906-7590.04596.x; Elith et al., 2011; https://doi.org/10.1111/j.1472-4642.2010.00725.x; Valavi et al., 2022; https://doi.org/10.1002/ecm.1486). Its intensity is a linear combination of pre-specified feature classes, such as linear, quadratic, product, threshold, and hinge features (Phillips et al., 2006; https://doi.org/10.1016/j.ecolmodel.2005.03.026; Merow et al., 2013; https://doi.org/10.1111/j.1600-0587.2013.07872.x), which constrains univariate response shapes and represents interactions through explicitly specified feature combinations. Deep learning offers more capacity: deep architectures for SDMs have shown competitive performance in large-scale benchmarks (Zbinden et al., 2024; https://doi.org/10.1016/j.ecoinf.2024.102623; Ryckewaert et al., 2026; https://doi.org/10.1111/2041-210X.70262; Kellenberger et al., 2026; https://doi.org/10.1111/geb.70184). But as complexity rises, predictive scores become harder to trace back to the environmental dimensions and functional forms that matter.

KANs separate the two promises. A KAN places a learnable univariate function on each edge and composes them in layers (Liu and Tegmark, 2025; https://doi.org/10.1103/4t7t-v19l), separating two questions that other architectures blur: can a learnable univariate edge function represent response curves more effectively than MaxEnt's fixed feature classes, and can deeper compositions learn generalisable environmental interactions beyond those functions? Systematic benchmark-based decompositions for ecological PO-SDMs remain limited.

Here, we use the six-region NCEAS dataset (Elith et al., 2020; https://doi.org/10.17161/bi.v15i2.13384) and follow the benchmark protocol of PO training and independent PA evaluation (Valavi et al., 2022; https://doi.org/10.1002/ecm.1486). We address three questions in sequence: whether standard end-to-end multilayer KAN can reliably outperform maxnet; whether an additive B-spline representation agrees with a same-basis GAM and how it compares with maxnet and BCE; and whether Deep-2 and Deep-3 provide consistent gains under a fair residual protocol when added to fitted additive main effects. Under this protocol, the fitted additive main effects are frozen and only the interaction mixer is optimised. 

## 1 Results

Unless otherwise stated, the primary analysis included 225 species across the six regions with at least $n_{\mathrm{PO}}\ge5$ training PO records. 

### 1.1 Performance of standard KAN across six regions

![[figures/ms_results/v7/Fig1_standard_kan_e2e_maxnet.png]]

**Fig. 1.** Species-level paired comparison of end-to-end standard KAN (intention-to-treat, after the predefined rescue protocol) and maxnet across the six NCEAS regions. Each point is one species; fill colour denotes region (AWT, CAN, NSW, NZ, SA, SWI). The dashed line is the line of equality. Open circles mark species that still failed the success criterion after the second rescue round (R2). The panel title reports the paired mean $\Delta$AUC (end-to-end KAN minus maxnet) and its bootstrap 95% confidence interval.

We first examined the end-to-end performance of standard multilayer KANs for all species in the six regions. The model had width $[P,4,1]$. Continuous covariates were scaled to $[-1,1]$ using the 1st and 99th percentiles of the training PO and background samples. Categorical covariates entered through the linear term $X_{\mathrm{cat}}\beta$. The model contained neither an additive skip connection nor an additive warm start; all parameters were jointly trained from random initialisation under the IPP objective. For failed fits, we applied up to two rounds of rescue in a pre-specified order. R1 lowered the learning rate, extended training, and used a different seed; R2 lowered the learning rate further and disabled SiLU. We retained the final usable prediction according to an intention-to-treat principle.

The mean AUC of end-to-end standard KAN was **0.642** (SD 0.124; median 0.616), compared with **0.714** for maxnet using a background cap of 10,000. The paired mean $\Delta$AUC (end-to-end KAN minus maxnet) was **$-0.0717$**, with a bootstrap 95% CI of $[-0.0859,-0.0571]$. The 95% CI excluded zero. Standard KAN was therefore systematically inferior to maxnet in this end-to-end setting. Nine species still failed to meet the predefined success criterion after rescue; their full status is reported in the Supporting Information. Regional paired differences are shown in Fig. S1.

To assess the reproducibility of the maxnet pipeline, we compared its predictions region by region with the MaxNet outputs made available by Valavi et al. (2022) on OSF. Pearson $r$ reached 0.983 in CAN and was at least 0.91 in every region. The poorer performance of end-to-end KAN is therefore unlikely to be caused by a mismatch in the maxnet pipeline.

### 1.2 Discriminative performance of additive models on independent PA data

![[figures/ms_results/v7/Fig2_additive_performance.png|680]]

**Fig. 2.** Discriminative performance on independent presence–absence data at species-specific $\lambda_s^\star$ ($n=225$). Navy, plum, and brick identify B-spline IPP, same-basis GAM-IPP, and maxnet; these method colours are distinct from the regional pastel palette used in (C), (D), and (F). (A) Species-level AUC distributions. Boxes show the interquartile range and median, whiskers are Tukey fences, light points are individual species, and black diamonds are means. White numerals give the median to three decimal places. (B) Mean AUC ± standard error across species, with the mean printed to the right of each bar. (C) Paired species scatter of B-spline IPP against same-basis GAM-IPP; the dashed line is 1:1 and points are coloured by region. (D) The same comparison against maxnet with a background cap of 10,000. (E) Paired $\Delta$AUC for the two additive contrasts; the dashed line is zero. Numerals are medians and are placed outside the box when the median lies near zero. (F) Species-level $\Delta$AUC (B-spline IPP minus maxnet) by region, with the same zero line and median annotation. Deep residual results are shown in Fig. 5.

After selecting the smoothing penalty $\lambda_s^\star$ using species-level PO cross-validation, all additive models showed better than random discrimination (Fig. 2A,B). Mean AUC was 0.723 for B-spline IPP (SD 0.138; median 0.726), and the same for same-basis GAM-IPP. Maxnet with a background cap of 10,000 gave 0.714 (SD 0.147; median 0.725), and the weighted BCE control under the same additive parameterisation was approximately 0.716. Species with sufficient records selected $\lambda_s^\star$ through five-fold PO cross-validation. Species with $5\le n_{\mathrm{PO}}<30$ used the fixed value $\lambda_s=10^{-2}$ to avoid unstable tuning. AUC was the primary summary metric in the following analyses; AUPRC and PRG are reported in Table S1, and the multi-metric plane is shown in Fig. S5.

Differences among regions primarily reflected intrinsic variation in predictability (Fig. S2). Regional mean AUC for B-spline IPP was 0.805 in SWI, 0.794 in SA, 0.738 in NZ, 0.704 in NSW, 0.672 in AWT, and 0.602 in CAN. B-spline IPP and same-basis GAM-IPP were numerically equivalent at the species level: the mean $\Delta$AUC (B-spline IPP minus GAM-IPP) was **$-0.0002$**, with a bootstrap 95% CI of $[-0.0006,+0.0000]$ and Pearson $r=0.9999$ (Fig. 2C). Given the same B-spline basis, knots, and penalty, the two implementations produced virtually identical fits. Regional auxiliary metrics and the species-by-model long table are provided in the Supporting Information (Tables S2–S4 and S12; Fig. S3).

The advantage of B-spline IPP over maxnet was modest but directionally consistent (Fig. 2D). The mean $\Delta$AUC was **$+0.0089$**, with a 95% CI of $[+0.0023,+0.0152]$; 57.8% of species had $\Delta>0$, 101 species exceeded $+0.01$, and 66 were below $-0.01$ (Fig. 2E). Individual species at the two extremes of the $\Delta$AUC distribution are shown in Fig. S4. The additive advantage was most apparent in SA ($+0.032$), CAN ($+0.020$), and SWI ($+0.017$), whereas NSW and AWT were close to parity (Fig. 2F). Raising the maxnet background cap from 10,000 to 50,000 changed the paired mean $\Delta$AUC by only approximately $+0.0006$, with a confidence interval crossing zero. Background truncation was therefore not the main source of the difference. The mean $\Delta$AUC of B-spline IPP relative to BCE was approximately $+0.006$, with the same direction but a narrower magnitude (Table S5).

### 1.3 Consistency of species ranking and effect of training sample size

![[figures/ms_results/v7/Fig3_additive_robustness.png]]

**Fig. 3.** Consistency of species ranking and the effect of training sample size at $\lambda_s^\star$. (A) The 225 species ordered by increasing B-spline IPP AUC (navy), with maxnet overlaid (brick). Shading marks local intervals in which one method exceeds the other. (B) Species-level $\Delta$AUC (B-spline IPP minus maxnet), ordered by magnitude; bar colours follow the regional palette in Fig. 1, and the dashed line is zero. (C) Independent-PA AUC of B-spline IPP versus the number of training presence-only records $n_{\mathrm{PO}}$ (log scale). Points are species coloured by region; the black curve is a spline smoother. (D) The corresponding $\Delta$AUC versus $n_{\mathrm{PO}}$, with a dashed zero line and the same smoother.

B-spline IPP and maxnet largely agreed on which species were easy or hard to model (Fig. 3A). The two models use different features and different optimisation paths, but the ordering barely changed. Positive differences were spread across regions and were more common in CAN and SA (Fig. 3B).

The number of training PO records covered a broad range, but it primarily affected estimation variance for individual species rather than the overall ordering of models. For $n_{\mathrm{PO}}<30$, B-spline IPP AUC was lower and more dispersed; the trend became more stable once record numbers exceeded approximately 30 (Fig. 3C; Table S6). The smoothed trend of $\Delta$AUC over $n_{\mathrm{PO}}$ remained close to zero (Fig. 3D). Species with enough records for cross-validated tuning had a slightly larger advantage over maxnet than species with few records and fixed λs. The difference was small.

### 1.4 Agreement of response curves

![[figures/ms_results/v7/Fig4_response_curves.png]]

**Fig. 4.** Agreement of univariate response curves between B-spline IPP and same-basis GAM-IPP. (A) Empirical cumulative distribution of Pearson $r$ across 159 feature-by-species curve pairs, with the 50th, 90th, and 95th percentiles marked. (B) Regional mean $r$; bars use the same regional colours as Fig. 1–3. The vertical axis is truncated at 0.90 to show among-region differences; values are printed on the bars. (C) Three more divergent curve pairs (left) and three highly concordant pairs (right). Solid lines are B-spline IPP and dashed lines are GAM-IPP; each panel is labelled with the region, species, covariate, and $r$.

The agreement between B-spline IPP and same-basis GAM-IPP extended from species-level AUC to univariate response curves. Eighteen stratified sampled species generated 159 feature-by-species curve pairs; the median Pearson $r$ was 0.9997, the mean was 0.988, and the 5th percentile was 0.947 (Fig. 4A). Only 10 curve pairs had $r<0.95$, including 5 below 0.90; the mean $r$ was above 0.95 in every region (Fig. 4B). The larger differences were mainly in local amplitude; the direction of response stayed the same (Fig. 4C). Regional summaries are provided in Table S11.

These curve comparisons checked whether the two same-basis implementations recovered similar functional shapes. They were not intended to interpret model components as causal effects. In regions with highly correlated covariates, an individual edge function should still be read as one part of the fitted response, conditional on the rest of the model.

### 1.5 Deep residual interaction ablation

![[figures/ms_results/v7/Fig5_deepkan_ablation.png]]

**Fig. 5.** Fair residual deep ablation. In (A) and (B), each region has a pair of boxes: solid navy for the mixer that takes edge-function outputs ($R_\phi$) and hatched brick for the mixer that takes scaled covariates ($R_x$). Boxes are species-mean $\Delta$AUC relative to the additive baseline, averaged over three seeds. The dashed line is zero. The two numerals above each region are the pair of medians (upper $R_\phi$, lower $R_x$), coloured to match the boxes and placed above the whiskers so that they are not read as labels on the zero line. The legend reports the global mean $\Delta$AUC. (A) Deep-2. (B) Deep-3. (C) $\Delta$AUC of Deep-2 $R_\phi$ relative to end-to-end standard KAN (intention-to-treat), by region, using the regional colour palette. Numerals are medians (inside the box when space allows, otherwise outside). This comparison reflects the benefit of retaining additive main effects, not an independent interaction benefit.

Under the fair residual protocol, we added a KAN mixer to the additive model at $\lambda_s^\star$. The additive fit served as a warm start; edge functions and categorical coefficients were frozen, and only the mixer was trained with L-BFGS, without SiLU. The residual form means that predictions return to the additive baseline when the mixer output is near zero. Deep-2 was $\sum_p\phi_p+\Phi(u)$ and Deep-3 was $\sum_p\phi_p+\Psi(\Phi(u))$, both of width $[P,4,1]$. The mixer input $u$ was either the edge outputs $\phi$ or the quantile-scaled covariates $T(\mathbf{x})$. All six regions and three seeds were covered; we averaged over seeds within species before summarizing across species.

Relative to the additive baseline, Deep-2 ($R_\phi$) lost **$-0.0048$** (95% CI $[-0.0099,-0.0001]$; median $-0.0008$; 48 species were above $+0.01$ and 71 below $-0.01$), and Deep-2 ($R_x$) lost **$-0.0085$** (CI $[-0.0137,-0.0035]$). Deep-3 lost more: **$-0.0211$** (CI $[-0.0270,-0.0155]$) with edge-output input and **$-0.0390$** (CI $[-0.0463,-0.0321]$) with scaled-covariate input. Mean AUC fell from 0.723 for the additive model to 0.718 (Deep-2 $R_\phi$), 0.714 (Deep-2 $R_x$), 0.702 (Deep-3 $R_\phi$), and 0.684 (Deep-3 $R_x$). Within-species variation across the three seeds was small: the standard deviation of species-level AUC averaged 0.0042 (median 0.0025).

Region by region, Deep-2 ($R_\phi$) stayed close to the additive baseline: mean $\Delta$AUC was $+0.0001$ in AWT, $+0.0053$ in CAN, $-0.0097$ in NSW, $-0.0099$ in NZ, $-0.0051$ in SA, and $-0.0003$ in SWI. Even in CAN, where the mean was the largest positive value, the confidence interval never stayed positive. Switching the mixer input from edge outputs to scaled covariates did not recover the lost performance. 

Residual models improved a lot over end-to-end KAN, but the relevant benchmark is the additive baseline. The gap between residual and end-to-end models shows the value of keeping the additive main effects intact. The gap between residual and additive models shows whether interactions add anything. Deep-2 and Deep-3 gave no transferable positive gain, and deeper models did worse.

### 1.6 Target-group background sensitivity

For 12 species in NSW, we compared the regional random background with a target-group background (TGB), built from PO locations of other species in the same taxonomic group. For B-spline IPP and same-basis GAM-IPP, the paired mean $\Delta$AUC (TGB minus random background) was $-0.0010$ (95% CI $[-0.0432,+0.0432]$) and $-0.0014$ (CI $[-0.0436,+0.0429]$). BCE gave $-0.0096$ (CI $[-0.0506,+0.0296]$). All three intervals included zero. Background choice therefore did not materially change performance in this subset, even though background-point settings are known to affect SDM performance (Whitford et al., 2024; https://doi.org/10.1016/j.ecolmodel.2023.110604). Maxnet yielded usable fits for only five species under TGB, so its subset result should not be interpreted as stable. Full values are reported in Tables S8–S9.

## 2 Discussion

The two capabilities of KANs, learnable univariate edge functions and deep composition, should be evaluated separately in single-species PO-SDMs. Standard end-to-end multilayer KAN was substantially inferior to maxnet on independent PA evaluation across six regions, showing that greater functional capacity alone does not guarantee generalisable distribution predictions. Large benchmark studies also find limited performance differences between maxnet and complex machine-learning models on independent PA evaluation (https://doi.org/10.1002/ecm.1486; Chollet Ramampiandra et al., 2023; https://doi.org/10.1016/j.ecolmodel.2023.110353).

When the model was restricted to additive B-spline edge functions, the result was more readily interpretable. B-spline IPP and same-basis GAM-IPP had almost identical AUCs and highly concordant response curves, as expected from their shared function class and penalty structure. B-spline IPP exceeded maxnet by a mean AUC of 0.0089. Although the effect was small, its direction was reproducible across species. The modest advantage reflects a more efficient encoding of the information already present in the data. MaxEnt relies on pre-specified feature classes and their combinations (https://doi.org/10.1016/j.ecolmodel.2005.03.026; https://doi.org/10.1111/j.1472-4642.2010.00725.x), whereas learnable B-splines estimate the nonlinear mapping between environmental variables and species response directly from the data. If the true ecological process is mainly governed by nonlinear constraints of temperature, precipitation, elevation, and related factors, this additional univariate flexibility can be used effectively, producing a slight improvement in ranking ability. On this benchmark, the additional freedom brought limited marginal benefit.

The deep residual ablation further tested whether deep interactions could add value beyond the additive representation. Once additive main effects were retained, residual models were much better than end-to-end KAN trained from random initialization; yet relative to the same additive baseline, neither Deep-2 nor Deep-3 produced a stable improvement. Comparing only residual and end-to-end models would conflate the benefit of retaining main effects with evidence that deep interactions work. In the present single-species PO data, the smoothly varying univariate mappings have already absorbed most stably usable signal, while additional interactions are more prone to fitting sampling bias and noise (https://doi.org/10.1890/07-2153.1; https://doi.org/10.1111/2041-210X.12242; Chollet Ramampiandra et al., 2023; https://doi.org/10.1016/j.ecolmodel.2023.110353). Once smooth univariate responses have been extracted, increasing model freedom expands the function space without providing sufficiently stable and rich higher-order signal to constrain the additional parameters. Model expressivity and learnable ecological information then become decoupled, and model complexity is no longer the principal limitation on predictive performance.

From a statistical-learning perspective, improved performance depends on the amount of information that data can estimate reliably, not on the size of the parameter space itself (Hastie et al., 2009; https://doi.org/10.1007/978-0-387-84858-7; Chollet Ramampiandra et al., 2023; https://doi.org/10.1016/j.ecolmodel.2023.110353). Additive B-splines use more flexible univariate mappings to extract the primary associations already present in the data. Deep interaction structures, in contrast, assume stable and reproducible combined effects. When such interactions are weaker than noise, the data cannot distinguish among many candidate relationships in high-dimensional space. That noise arises from sampling error, spatial bias, and unobserved variables (Nolan et al., 2022; https://doi.org/10.1111/jbi.14268). Under these conditions, greater complexity adds uncertain parameter space and little useful predictive information.

Deep interactions offered no gain under either mixer input. If building interactions on edge outputs $\phi$ had created an information bottleneck, direct use of scaled covariates should have produced a systematic improvement, but the results do not support this. The study nevertheless has three limitations. First, we did not systematically compare an explicit-interaction GAM with matched capacity, such as one using tensor-product smooths (https://doi.org/10.1111/j.1467-9868.2010.00749.x). Second, the fair residual protocol froze additive main effects and did not examine joint fine-tuning of main effects and interactions. Third, all models were fitted independently by species and did not use information sharing across species. Interpretable interaction models such as GA$^2$M provide a methodological precedent for residual decomposition (https://doi.org/10.1145/2487575.2487579; Yang et al., 2021; https://doi.org/10.1016/j.patcog.2021.108192), but their ecological applicability still requires evaluation under more explicit interaction hypotheses.

Multispecies joint deep models can benefit from borrowing information across species, through shared hidden layers, as in DeepMaxEnt (https://doi.org/10.1016/j.ecoinf.2024.102623; https://doi.org/10.1111/2041-210X.70262; https://doi.org/10.1111/geb.70184). This increases the information available to each species, whereas single-species residual KAN lacks such cross-species information. For individual or information-poor species, future work should increase the information structure available to the model, for example through functional-trait constraints (https://doi.org/10.1111/j.1600-0587.2011.07085.x), joint species distribution models (JSDMs; https://doi.org/10.1111/2041-210X.12180), hierarchical community-model frameworks (https://doi.org/10.1111/2041-210X.13345), or priors based on historical climate trajectories, or integrated models that combine presence-only and presence-absence data (Morera-Pujol et al., 2023; https://doi.org/10.1111/ecog.06451). Model complexity should be matched to the amount and structure of information available.

## 3 Conclusions

KANs provide both learnable nonlinear edge functions and deep compositional structure, but these components do not make equal contributions to single-species presence-only SDMs. End-to-end standard multilayer KAN was systematically inferior to maxnet across the six regions. Once the representation was decomposed, additive B-spline edge functions at species-specific $\lambda_s^\star$ aligned with the same-basis GAM and were slightly better than maxnet on average. Residual Deep-2 and Deep-3 models added no transferable gain to this baseline, regardless of input type, and deeper models performed worse.

Thus, flexible univariate response functions captured most of the reliably usable signal in this benchmark, and an additive model is sufficient to exploit it. It can serve as an interpretable tool alongside maxnet and GAMs; multilayer KANs and single-species residual deep interactions are not suitable as defaults.

---

## 4 Materials and Methods

### 4.1 Data

We used the NCEAS species-distribution benchmark data (Elith et al., 2020; https://doi.org/10.17161/bi.v15i2.13384). The PO-training and independent-PA-evaluation protocol followed Valavi et al. (2022; https://doi.org/10.1002/ecm.1486). The data include 226 species identifiers across six regions, AWT (40 species), CAN (20), NSW (54), NZ (52), SA (30), and SWI (30). The primary analysis included 225 species with $n_{\mathrm{PO}}\ge5$; one species with insufficient records was excluded. Each region provides an independent PA test set, 50,000 random background points, and region-specific environmental covariates.

Regions contained 6–13 continuous covariates; some also included categorical covariates, for example vegetation type in CAN, vegetation system in NSW, age and toxicity classes in NZ, and lithology class in SWI. Continuous covariates were standardised using the combined PO and background samples in the B-spline, GAM, and BCE pipelines, whereas maxnet used the original scale. Categorical covariates were one-hot encoded and entered the model through the linear term $X_{\mathrm{cat}}\beta$, with an $\ell_2$ penalty on $\beta$. PA labels were used only for final evaluation and were not used for fitting, selecting $\lambda_s$, or scaling covariates.

### 4.2 Statistical framework

All models estimated the relative log-intensity $\eta(\mathbf{x})$ conditional on environmental covariates $\mathbf{x}$. We treated PO records as observations under the conditional likelihood of a spatially discretised inhomogeneous Poisson point process (IPP). The normalisation term was computed over the training support (PO plus background) with log-sum-exp (Renner and Warton, 2013; https://doi.org/10.1111/j.2041-210x.2012.00245.x; Renner et al., 2015; https://doi.org/10.1111/2041-210X.12352). When $\eta$ is a linear function of features, this objective corresponds to MaxEnt (Fithian and Hastie, 2013; https://doi.org/10.1214/13-AOAS667). Our primary comparison therefore varies the parameterisation of $\eta$ while keeping the training objective fixed.

Categorical covariates are written uniformly as $X_{\mathrm{cat}}\beta$; this term is omitted in regions without categorical variables. We use $\eta_{\mathrm{cont}}$ for the continuous-covariate component, such that the complete predictor is $\eta=\eta_{\mathrm{cont}}+X_{\mathrm{cat}}\beta$.

### 4.3 Representations of the intensity function

#### 4.3.1 Additive family (primary benchmark)

**B-spline IPP.** The continuous-covariate component is defined as a sum of additive edge functions:

$$
\eta_{\mathrm{cont}}(\mathbf{x})=\sum_{p=1}^{P}\phi_p(x_p),
$$

where each $\phi_p$ is a linear combination of cubic B-spline basis functions (number of intervals $G=6$, order $K=3$; Wood, 2011; https://doi.org/10.1111/j.1467-9868.2010.00749.x). The global intercept was fixed at 0, and edge functions were centred during training. The smoothing-penalty coefficient was $\lambda_s$, and the coefficient ridge penalty was $\lambda_r=10^{-6}$.

**Selection of $\lambda_s^\star$.** We selected $\lambda_s^\star$ by random five-fold cross-validation over PO occurrence indices (Hastie et al., 2009; https://doi.org/10.1007/978-0-387-84858-7), ranking fold-level performance by the AUC of held-out PO relative to background (Phillips et al., 2006; https://doi.org/10.1016/j.ecolmodel.2005.03.026), and chose from ${10^{-4},10^{-3},10^{-2},10^{-1},1}$. For species with $5\le n_{\mathrm{PO}}<30$, we fixed $\lambda_s=10^{-2}$, because cross-validated tuning is unstable with so few records (Merow et al., 2013; https://doi.org/10.1111/j.1600-0587.2013.07872.x); species with $n_{\mathrm{PO}}<5$ were excluded from the primary analysis. After selection, models were refitted using all PO records and 50,000 background points. Unless otherwise stated, the primary analysis uses $\lambda_s^\star$.

**Same-basis GAM-IPP.** GAM-IPP used exactly the same B-spline basis, knot boundaries, $\lambda_s^\star$, and $\lambda_r$ as B-spline IPP, differing only in the solution implementation (scipy L-BFGS-B; Byrd et al., 1995; https://doi.org/10.1137/0916069). It served as a numerical cross-check.

**maxnet.** We used R `maxnet` 0.1.4 (the glmnet implementation of MaxEnt; Phillips et al., 2017; https://doi.org/10.1111/ecog.03049; Friedman et al., 2010; https://doi.org/10.18637/jss.v033.i01), with `l` or `lq` feature classes (Phillips et al., 2006; https://doi.org/10.1016/j.ecolmodel.2005.03.026; Elith et al., 2011; https://doi.org/10.1111/j.1472-4642.2010.00725.x). The primary comparison used a cap of 10,000 background points, following Valavi et al. (2022); results with 50,000 background points are also reported.

**BCE control.** Under the same additive parameterisation as B-spline IPP, we replaced the loss with weighted Bernoulli cross-entropy, with a positive-class weight of $n_{\mathrm{BG}}/n_{\mathrm{PO}}$. BCE was used only to check whether the loss function mattered (Fithian and Hastie, 2013; https://doi.org/10.1214/13-AOAS667).

The additive primary analysis covered 225 eligible species in all six regions, with random seed fixed to 0.

#### 4.3.2 Residual deep interactions (fair residual protocol)

We added a residual mixer to the additive edge functions, so that $\eta_{\mathrm{cont}}$ represents limited interactions beyond the main effects. The additive model was fitted first, and its parameters were frozen before the mixer was optimized. This decomposition of main effects and interactions follows the GA$^2$M framework (Lou et al., 2013; https://doi.org/10.1145/2487575.2487579).

**Deep-2:**

$$
\eta_{\mathrm{cont}}(\mathbf{x})=\sum_{p=1}^{P}\phi_p(x_p)+\Phi(u),
$$

where $\Phi:\mathbb{R}^P\rightarrow\mathbb{R}$ was implemented with pykan 0.2.8 (Liu et al., 2025; https://doi.org/10.48550/arXiv.2404.19756), with the knot grid frozen during training.

**Deep-3:** $\eta_{\mathrm{cont}}=\sum_p\phi_p+\Psi(\Phi(u))$, with layer width $[P,4,1]$.

Two mixer inputs $u$ were considered. The first was the edge-function output $\phi$, scaled by its training-set standard deviation (denoted $R_\phi$). The second was the quantile-scaled covariates $T(\mathbf{x})$, mapped to $[-1,1]$ using the 1st and 99th percentiles of the training PO and background samples (denoted $R_x$). Training warm-started from the additive fit at $\lambda_s^\star$, then froze the edge functions and categorical coefficients and optimized only the mixer with L-BFGS (Liu and Nocedal, 1989; [https://doi.org/10.1007/BF01589116](https://doi.org/10.1007/BF01589116)). The SiLU base activation was disabled (Elfwing et al., 2018; [https://doi.org/10.1016/j.neunet.2017.12.012](https://doi.org/10.1016/j.neunet.2017.12.012)). The residual structure ensures a return to the additive prediction when the mixer approaches zero. Both Deep-2 and Deep-3 covered all six regions, 225 species, and seeds $\{0,1,2\}$; results were first averaged over seeds within species and then summarized across species. The mixer received an additional $\ell_2$ penalty of $\lambda_\phi=10^{-4}$.

#### 4.3.3 Standard multilayer KAN (end to end, six regions)

Standard multilayer KANs were fitted end to end for all species in all six regions (Liu et al., 2025; https://doi.org/10.48550/arXiv.2404.19756; Liu and Tegmark, 2025; https://doi.org/10.1103/4t7t-v19l):

$$
\eta_{\mathrm{cont}}(\mathbf{x})=\mathrm{KAN}_{[P,h,1]}\big(T(\mathbf{x}_{\mathrm{cont}})\big),
$$

where $T$ is as defined above and $h=4$. Edge functions used $G=6$ intervals and order $K=3$; SiLU was enabled by default, the knot grid was frozen, and the package’s default entropy and sparsity penalties were disabled. The model included neither an additive skip connection nor an additive warm start. KAN parameters and categorical coefficients $\beta$ were jointly trained under the IPP objective from random initialization (seed 0).

If a fit produced non-finite loss or non-finite predictions, failed to converge, or had independent PA AUC below 0.55 with an AUC deficit greater than 0.15 relative to maxnet, we applied the pre-specified rescue sequence: R1 lowered the learning rate, extended training, and used a different seed; R2 lowered the learning rate further and disabled SiLU. The primary analysis used the first successful result in the rescue chain according to an intention-to-treat principle; if all attempts failed, the final prediction was retained and marked as failed.

### 4.4 Training and regularisation

Additive B-spline/GAM models used $\lambda_s^\star$ and $\lambda_r=10^{-6}$. B-spline knot boundaries were determined from the training data and then fixed; optimisation was primarily performed with L-BFGS (Liu and Nocedal, 1989; https://doi.org/10.1007/BF01589116). Maxnet was fitted with the feature classes and background protocol described in Section 4.3.1.

Standard multilayer KAN was first trained with Adam (learning rate 0.03, default 150 steps, gradient-norm clipping; Kingma and Ba, 2015; https://arxiv.org/abs/1412.6980), followed by L-BFGS refinement (10 outer steps); the number of training steps was adapted according to $n_{\mathrm{PO}}$. The KAN regularization coefficient was $\lambda_{\mathrm{kan}}=10^{-4}$, and the categorical-term ridge penalty was $\lambda_r=10^{-6}$. Training settings for residual deep models are described above under the fair residual protocol.

### 4.5 Evaluation and implementation

The primary evaluation metric was ROC AUC on independent PA data (Hanley and McNeil, 1982; https://doi.org/10.1148/radiology.143.1.7063747); we also report AUPRC (Sofaer et al., 2019; https://doi.org/10.1111/2041-210X.13140), precision–recall gain (PRG; Flach and Kull, 2015; https://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right), and the Pearson correlation between prediction scores and labels (COR). Models were compared with species as the statistical unit, reporting paired $\Delta$AUC and bootstrap 95% confidence intervals (Efron and Tibshirani, 1993; https://doi.org/10.1201/9780429246593).

To assess the effect of background definition, we used 12 NSW species. At the same $\lambda_s^\star$, we fitted B-spline IPP, same-basis GAM-IPP, BCE, and maxnet with background caps of 10,000 and 50,000, using either the 50,000 regional random background points or a target-group background. The target-group background comprised PO locations of other species in the same taxonomic group, with the focal species excluded and a cap of 50,000. We evaluated each model on independent PA data. Paired $\Delta$AUC was defined as the TGB result minus the random-background result, and model-level means were accompanied by bootstrap confidence intervals. If maxnet did not yield a usable prediction, it was recorded as non-convergent and the number of completed species was reported (for background definition and sampling bias, see Phillips et al., 2009; https://doi.org/10.1890/07-2153.1).

Analyses were implemented in Python/PyTorch (Paszke et al., 2019; https://doi.org/10.5555/3454287.3455008). The deep components used pykan 0.2.8 (Liu et al., 2025; https://doi.org/10.48550/arXiv.2404.19756), and maxnet was called through the R `maxnet` package (Phillips et al., 2017; https://doi.org/10.1111/ecog.03049).

---

## Supporting Information

The Supporting Information is available with the online version of this article (English file: Supporting Information). Unless otherwise stated, all summaries use the primary analysis at species-specific $\lambda_s^\star$ on independent presence–absence evaluation for the 225 species with $n_{\mathrm{PO}}\ge5$.

**Figures.** Figure S1 shows the regional distribution of species-level $\Delta$AUC for end-to-end standard KAN (intention-to-treat) relative to maxnet. Figure S2 shows regional AUC boxplots for B-spline IPP, same-basis GAM-IPP, and maxnet. Figure S3 is a species-by-model AUC heatmap within each NCEAS region. Figure S4 highlights the species with the most negative and most positive $\Delta$AUC (B-spline IPP minus maxnet). Figure S5 places the three additive models on multi-metric planes of mean AUC versus mean COR and mean AUPRC (± SE).

**Tables.** Table S1 reports global species-level metrics for additive, residual deep, and end-to-end models. Tables S2–S4 give regional means of AUC, AUPRC, and PRG. Table S5 summarises key paired $\Delta$AUC contrasts from the primary analysis. Table S6 stratifies AUC by training presence-only sample size. Table S7 lists species-by-model AUC for the CAN region. Tables S8–S9 report the target-group background sensitivity analysis at the model and species levels. Table S10 gives regional $\Delta$AUC for fair residual Deep-2 and Deep-3 models. Table S11 summarises response-curve agreement by region. Tables S12–S14 provide machine-readable long tables of species-by-model AUC and COR, species-specific $\lambda_s^\star$, and end-to-end intention-to-treat outcomes. Rendered SI figures and CSV tables are also archived with the code repository (https://github.com/Akakk1/KAN_Maxent).

---

## Data availability

Code for model fitting, evaluation, and figure generation is available at https://github.com/Akakk1/KAN_Maxent (commit cdfdb65; MIT License). The repository includes the `kanmaxent` Python package, experiment scripts, unit tests, a short smoke-check entry point, pre-computed metrics tables under `results/`, and the rendered main-text and Supporting Information figures. Full multi-region re-runs are optional for recovering the published summary statistics; they write to a local `outputs/` directory when executed.

The primary empirical analyses use the six-region NCEAS presence-only / independent presence–absence benchmark compiled by Elith et al. (2020) and analysed as a community benchmark by Valavi et al. (2022). Region-level occurrence and environmental tables used in this study are distributed with the code repository under `data/nceas/`. For pipeline checks only, we compared our maxnet predictions with the MaxNet outputs released by Valavi et al. (2022) on the Open Science Framework (OSF); those third-party prediction files are not redistributed here and should be obtained from the original OSF deposit. Optional R `maxnet` baselines require the `maxnet` package (Phillips et al., 2017).

---

## References

Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*. https://doi.org/10.1137/0916069

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. https://doi.org/10.1201/9780429246593

Elfwing, S., Uchibe, E., & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. *Neural Networks*. https://doi.org/10.1016/j.neunet.2017.12.012

Elith, J., et al. (2006). Novel methods improve prediction of species distributions from occurrence data. *Ecography*. https://doi.org/10.1111/j.2006.0906-7590.04596.x

Elith, J., et al. (2011). A statistical explanation of MaxEnt for ecologists. *Diversity and Distributions*. https://doi.org/10.1111/j.1472-4642.2010.00725.x

Elith, J., et al. (2020). Presence-only and presence-absence data for comparing species distribution modeling methods. *Biodiversity Informatics*. https://doi.org/10.17161/bi.v15i2.13384

Fithian, W., & Hastie, T. (2013). Finite-sample equivalence in statistical models for presence-only data. *Annals of Applied Statistics*. https://doi.org/10.1214/13-AOAS667

Fithian, W., Elith, J., Hastie, T., & Keith, D. A. (2015). Bias correction in species distribution models: pooling survey and collection data for multiple species. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.12242

Flach, P., & Kull, M. (2015). Precision-recall-gain curves: PR analysis done right. *Advances in Neural Information Processing Systems*. https://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*. https://doi.org/10.18637/jss.v033.i01

Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*. https://doi.org/10.1148/radiology.143.1.7063747

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

Kellenberger, B., Winner, K., & Jetz, W. (2026). The performance and potential of deep learning for predicting species distributions. *Global Ecology and Biogeography*. https://doi.org/10.1111/geb.70184

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*. https://doi.org/10.1007/BF01589116

Liu, Z., et al. (2025). KAN: Kolmogorov–Arnold Networks. *arXiv*. https://doi.org/10.48550/arXiv.2404.19756

Liu, Z., & Tegmark, M. (2025). Kolmogorov–Arnold networks meet science. *Physical Review X*. https://doi.org/10.1103/4t7t-v19l

Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *KDD*. https://doi.org/10.1145/2487575.2487579

Merow, C., Smith, M. J., & Silander, J. A. (2013). A practical guide to MaxEnt for modeling species distributions. *Ecography*. https://doi.org/10.1111/j.1600-0587.2013.07872.x

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*. https://doi.org/10.5555/3454287.3455008

Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*. https://doi.org/10.1016/j.ecolmodel.2005.03.026

Phillips, S. J., et al. (2009). Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. *Ecological Applications*. https://doi.org/10.1890/07-2153.1

Phillips, S. J., et al. (2017). Opening the black box: an open-source release of Maxent. *Ecography*. https://doi.org/10.1111/ecog.03049

Pollock, L. J., Morris, W. K., & Vesk, P. A. (2012). The role of functional traits in species distributions revealed through a hierarchical model. *Ecography*. https://doi.org/10.1111/j.1600-0587.2011.07085.x

Pollock, L. J., et al. (2014). Understanding co-occurrence by modelling species simultaneously with a joint species distribution model (JSDM). *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.12180

Renner, I. W., & Warton, D. I. (2013). Equivalence of MAXENT and Poisson point process models for species distribution modeling in ecology. *Methods in Ecology and Evolution*. https://doi.org/10.1111/j.2041-210x.2012.00245.x

Renner, I. W., et al. (2015). Point process models for presence-only analysis. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.12352

Ryckewaert, M., et al. (2026). Applying the maximum entropy principle to neural networks enhances multi-species distribution models. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.70262

Sofaer, H. R., Hoeting, J. A., & Jarnevich, C. S. (2019). The area under the precision-recall curve as a performance metric for rare binary events. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.13140

Tikhonov, G., et al. (2020). Joint species distribution modelling with the R-package Hmsc. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.13345

Valavi, R., et al. (2022). Predictive performance of presence-only species distribution models: a benchmark study with reproducible code. *Ecological Monographs*. https://doi.org/10.1002/ecm.1486

Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*. https://doi.org/10.1111/j.1467-9868.2010.00749.x

Zbinden, R., et al. (2024). On the selection and effectiveness of pseudo-absences for species distribution modeling with deep learning. *Ecological Informatics*. https://doi.org/10.1016/j.ecoinf.2024.102623
