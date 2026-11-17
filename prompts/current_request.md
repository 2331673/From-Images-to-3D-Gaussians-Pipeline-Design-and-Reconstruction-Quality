# Image-to-3DGS: English project prompts

The prompts are organized by project stage using a November 4–17 presentation schedule.

## November 4

### P01. Translate the assignment into a research question

Develop a reproducible image-to-3D reconstruction pipeline using the cleaned data from Assignment 1 and 3D Gaussian Splatting. Frame the central question as: which decisions between input preparation and final rendering most strongly influence reconstruction quality? Separate deliverables from scientific questions. Require ten object reconstructions, one orbit GIF per object, retained point correspondences, executable code, and a report explaining the evidence behind the conclusions.

Inspect the existing implementation and artifacts before proposing additional work. Produce a requirement-to-artifact checklist and identify missing evidence. Distinguish completed experiments from proposed comparisons. Treat reconstruction quality as a combination of held-out image fidelity, foreground appearance, geometric initialization, and stability across viewpoints; do not reduce it to one aggregate score.

### P02. Audit the cleaned data and camera conventions

Inspect the cleaned asset manifest and select ten objects with useful variation in appearance and matching difficulty. Record asset identifiers, image dimensions, view counts, alpha masks, and camera metadata. Explain how the selection relates to the cleaned pool and why ten selected objects cannot establish dataset-wide performance.

For the available 30-view, 256-by-256 RGBA inputs, verify pose direction, camera axes, intrinsics, and scene scale before training. Use projection checks to detect convention errors. Keep views 000, 008, 016, and 024 for evaluation and the remaining 26 for training. Trace whether held-out images enter matching, initialization, or optimization; report any leakage found rather than assuming the split is enforced. Specify consistent alpha compositing and background treatment.

## November 5

### P03. Select and document a reproducible implementation

Review the existing gsplat-based implementation and document the dependency versions, hardware assumptions, and executable entry points needed to reproduce it. Establish the role of gsplat v1.5.3 in the available implementation. Prefer a version-pinned working configuration and preserve relevant upstream licenses.

Map the pipeline into data preparation, feature matching, geometric filtering, triangulation, Gaussian initialization, optimization, evaluation, and orbit rendering. For every transition, identify the input artifact, output artifact, and a practical correctness check. Distinguish known-camera triangulation from full structure-from-motion: the current camera poses are provided, so do not describe them as estimated by the pipeline. Deliver a concise implementation map and a dependency record without machine-specific credentials.

## November 6

### P04. Construct reliable point correspondences

Implement or audit SIFT matching on the training views with bidirectional ratio filtering at 0.8. Restrict features to the foreground using alpha greater than 250 and a 3-by-3 erosion. Apply the implemented epipolar threshold of 1.5 pixels, minimum parallax of 2 degrees, and reprojection threshold of 1.5 pixels. Confirm the order and definitions of these filters against the code.

Save both visual matching examples and machine-readable retained correspondences for every object. Explain the distinct failure modes addressed by descriptor filtering, foreground masking, epipolar geometry, parallax, and reprojection. Discuss the trade-off between point count and geometric reliability. Do not present these thresholds as optimal without a sensitivity experiment; use the retained-point statistics to identify objects for deeper inspection.

## November 7

### P05. Inspect triangulation and initialization failures

Use the known poses to triangulate accepted matches and export the sparse reconstruction in the format expected by the trainer. Check positive depth, reprojection consistency, color assignment, and coordinate compatibility. Record the retained point count and the initialization method for each of the ten objects.

Audit the rule that uses triangulated points when at least 20 reliable points are available and otherwise initializes 10,000 random points. Inspect objects 03 and 05, whose recorded counts are 6 and 18, and explain why the fallback is triggered. Consider low texture, repeated appearance, silhouette features, and viewpoint overlap as possible explanations, then distinguish what the images support from what remains a hypothesis. Do not equate a sparse point count with poor final rendering automatically.

## November 8

### P06. Establish a controlled training protocol

Document and verify the shared 3DGS training protocol for the ten objects: 6,000 optimization steps, seed 42, fixed train/test split, and the recorded H100 hardware environment. Read the actual run configurations for optimizer, loss, densification, pruning, and rendering settings rather than supplying conventional defaults from memory.

Preserve checkpoints, runtime statistics, Gaussian counts, evaluation images, and run configurations. Check that all objects reach the intended training budget and that each result resolves to the correct input object. Explain how densification can increase representational capacity while also increasing cost and potentially fitting view-specific detail. Treat this as a mechanism to examine, not proof that a larger Gaussian count improves generalization.

## November 9

### P07. Isolate the effect of geometric initialization

Evaluate the available initialization comparisons for objects 01 and 09. Compare triangulated-point initialization with random initialization while checking that the image split, optimization budget, seed, and other recorded training settings match. Identify any remaining differences in initial point count or point distribution that limit how narrowly the effect can be attributed.

Extract held-out PSNR, SSIM, LPIPS, runtime, and final Gaussian count wherever available. Compute differences from saved outputs rather than copying a narrative summary. Explain whether the evidence supports better final fidelity, faster convergence, or both; a final-checkpoint comparison alone cannot establish faster convergence. Limit the conclusion to these two tested objects and distinguish this ablation from the automatic fallbacks in the main ten-object experiment.

## November 10

### P08. Produce comparable visual reconstructions

Render a consistent orbit animation for each trained object using the available 96-frame trajectory. Verify camera orientation, orbit center, framing, resolution, background, and playback settings. Decode every GIF to confirm that it contains a valid animation and corresponds to the intended checkpoint.

Inspect silhouettes, fine structures, texture continuity, floaters, and changes across viewpoints. Relate suspicious views to the input coverage and matching evidence where possible. Use orbit GIFs to communicate view consistency, while retaining held-out image evaluation for quantitative comparisons. An orbit without reference images is a qualitative inspection tool, so avoid treating it as an independent numerical measure of geometric accuracy.

## November 11

### P09. Define evaluation precisely

Write an evaluation specification for PSNR, SSIM, LPIPS, and foreground error using the implementation actually executed. Include the MSE-to-PSNR formula and image value range; define SSIM terms and LPIPS feature-distance notation. Identify the LPIPS backbone, preprocessing, and library settings from the code. State the direction of improvement for each metric.

Define the foreground mask and its normalization explicitly, and distinguish full-image evaluation from foreground-only evaluation. Describe how scores are averaged across held-out views and then across objects. Recompute the same-protocol full-image versus foreground PSNR comparison from the PNG predictions so that image encoding does not confound the interpretation. Explain why a large easy background can obscure foreground reconstruction errors.

## November 12

### P10. Analyze variation across objects

Build the ten-object results table from the saved evaluation records and verify the reported macro averages. Analyze variation rather than restating every table row. Ask which objects are difficult, whether their errors are concentrated on the foreground, and whether matching evidence and rendered views suggest plausible explanations.

Separate measured associations from causal conclusions. A relationship among point count, final Gaussian count, and image quality does not by itself identify the mechanism. Use object-specific evidence to discuss texture, thin structures, view coverage, and initialization, and state when a proposed explanation would require an additional controlled experiment. Keep numerical claims traceable to the saved metrics and analysis outputs.

## November 13

### P11. Test whether additional optimization helps consistently

Compare the 2,000-step and 6,000-step evaluations within the same runs. Compute each object's PSNR change, the mean and median change, and the mean after excluding object 05. Quantify object 05's share of the net increase and inspect LPIPS changes alongside PSNR. Compare Gaussian-count growth with the quality changes.

Use these checks to determine whether the mean gain reflects broad improvement or one influential case. Explain why a positive mean does not imply every object benefits, and why disagreement between PSNR and LPIPS deserves interpretation. Distinguish the observed checkpoint comparison from an early-stopping policy: selecting checkpoints on held-out test performance would require a separate validation protocol before claiming an unbiased improvement.

## November 14

### P12. Synthesize stage-by-stage influences on quality

Organize the analysis around the mechanisms connecting pipeline decisions to reconstruction quality. Cover data and view coverage, camera correctness, foreground handling, correspondence filtering, triangulation, initialization, optimization duration, densification, evaluation region, and rendering trajectory. For each stage, identify its likely effect and the evidence available in this project.

Use controlled evidence from the two initialization comparisons, observational evidence from the ten-object results, and the within-run checkpoint comparison without merging their evidential strength. Prioritize the findings that change how the pipeline should be used. Explain which checks prevent catastrophic failures, which choices measurably affect the tested reconstructions, and which ideas remain untested. Avoid describing unperformed threshold, pose-noise, or coverage ablations as completed experiments.

## November 15

### P13. Write an analysis-centered course paper

Write an English course paper in the NeurIPS template with a clear research question, a concise pipeline description, reproducible experimental settings, and an analysis-centered results section. The paper should explain why outcomes differ and what the evidence supports, rather than narrating implementation steps or repeating numbers already visible in figures and tables.

Structure the argument around geometric initialization, heterogeneous responses to longer training, and foreground-aware evaluation. Include the metric equations and a 3DGS workflow diagram consistent with the actual implementation. Keep observations, interpretations, and recommendations distinguishable. Use direct academic prose, remove repeated defensive language, consolidate material limitations into a focused discussion, and omit Appendix C. Include the required group information and contributions only in the separately submitted report.

## November 16

### P14. Check the argument against the artifacts

Review every quantitative statement in the report against the stored results. Recalculate initialization deltas, aggregate scores, checkpoint differences, and foreground comparisons. Ensure that the discussion does not infer geometric correctness from rendering metrics alone or generalize two initialization comparisons to all possible objects.

Check that each figure answers a specific analytical question, captions define the comparison, and prose adds interpretation instead of repeating labels. Verify that the workflow diagram represents known poses, matching, triangulation, fallback initialization, Gaussian optimization, and rendering accurately. Improve transitions from observations to explanations, and remove claims that have no supporting artifact. Do not alter experiment outputs to make the narrative more consistent.

## November 17

### P15. Prepare the reproducible repository

Package the implementation, cleaned inputs used by the experiment, experiment configurations, model artifacts, numerical analysis code, matching results for all ten objects, and ten rendered GIFs. Provide a README with environment setup, artifact extraction, reproduction commands, and a map from outputs to analysis. Verify archive integrity and preserve third-party license notices.

Keep the final report PDF and all report-generation code outside the repository. Remove member names, student identifiers, private machine paths, credentials, and API keys from published files and packaged text metadata. Preserve the actual experimental timestamps and numerical outputs. Organize the project prompts by stage using the November 4–17 presentation schedule.

### P16. Audit submission completeness

Perform a final requirement-by-requirement audit: the complete image-to-3DGS pipeline, analysis of quality-affecting stages, ten objects, ten GIFs, point-matching artifacts for every object, executable code, prompt documentation, and a separate final report containing the repository link and required authorship information.

Verify that uploaded files match the prepared release and that the repository visibility matches the owner's choice. For a private repository, identify reviewer access as a separate submission dependency. Produce a short checklist of verified deliverables and genuinely unresolved items; do not treat repository upload alone as completion of course submission.
