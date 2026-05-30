# Project Milestones and Goals

This document is the running reference for major project goals, active milestones, decisions, and deferred ideas. Add new milestones to the top of the active list while they are in progress, then move them to the completed section once shipped.

## How to Use This File

- Capture one milestone per section.
- Keep goals, decisions, acceptance criteria, and next steps together.
- Prefer updating an existing milestone over creating duplicate entries.
- Move shipped work into `Completed Milestones` and keep future ideas in `Backlog`.

## Active Milestones

### 2026-04-28: Old Television Overlay Asset Pack

**Status:** In Progress

**Goal:**
Create a reusable visual overlay pack that makes generated scene sequences look like they are playing inside an old television set, without depending on per-render manual layout work.

**Decision:**
Use a hybrid workflow:

- Generate television concepts with the existing image-generation stack.
- Convert the selected concept into clean layered static assets manually.
- Keep scanlines, vignette, light noise, and similar analog treatments procedural in ffmpeg.

**Why This Approach:**

- Image models are useful for exploring television styles quickly.
- Final overlay assets need exact transparency and stable screen geometry.
- ffmpeg is better suited for repeatable, tunable effects like scanlines, mild noise, contrast shaping, and vignette.
- A reusable asset pack avoids regenerating fragile overlay art for every project.

**Recommended Asset Pack:**

- `tv_bezel_clean.png`: primary television frame with a transparent screen cutout.
- `tv_glare_soft.png`: optional reflective glass overlay with transparency.
- `tv_dust.png`: optional subtle texture overlay for age and wear.
- `tv_mask.png`: optional mask for rounded or curved screen shaping.
- `tv_overlay.json`: geometry metadata for the active screen area.

**Recommended Authoring Specs:**

- Author master assets at `2560x1440` minimum.
- Prefer `3840x2160` if the television frame contains fine texture.
- Export as PNG with alpha for all visual overlays.
- Keep the screen opening centered and measured precisely.
- Design for front-facing or near-front-facing perspective only.

**Screen Geometry Metadata Example:**

```json
{
  "name": "portable-crt-01",
  "canvas": {"width": 3840, "height": 2160},
  "screen": {"x": 812, "y": 446, "width": 2208, "height": 1242},
  "notes": "Generated scene montage should be scaled into this rectangle before bezel/glare overlays are applied."
}
```

**Generation Workflow:**

1. Generate concept art for multiple television directions using the current image workflow.
2. Select one direction based on readability, mood, and how much of the frame it leaves for the playable screen.
3. Clean up the chosen concept in a design tool such as Photoshop, GIMP, Affinity Photo, or Figma.
4. Separate the concept into reusable RGBA layers.
5. Measure and store exact screen bounds in metadata.
6. Use ffmpeg to place the stitched scene video inside the stored screen rectangle.
7. Apply bezel and glare overlays after scene scaling.
8. Add analog treatments procedurally in ffmpeg rather than baking them into the PNGs.

**Suggested Prompt Direction for Concept Generation:**

Base prompt:

```text
front-facing vintage cathode ray tube television, centered and symmetrical, isolated product shot, neutral studio background, worn black plastic bezel, curved glass screen, subtle reflections, realistic proportions, high detail, no logo, no text, no cables, straight-on camera
```

Negative prompt additions:

```text
text, watermark, logo, angled perspective, duplicate television, cluttered background, extra devices, blurry edges, warped proportions, cropped frame, asymmetrical geometry
```

Style variants worth testing:

- Realistic compact CRT monitor
- 1970s woodgrain television cabinet
- Portable analog TV with knobs and heavy bezel

**Implementation Guidance:**

- Apply the overlay once to the stitched scene montage rather than per scene clip.
- Keep the cinematic intro separate unless there is a specific reason to show the intro card inside the television frame.
- Preserve the existing audio mux step as a pure mux if possible; add the new visual processing before final audio attachment.

**Incremental Progress:**

- Added a CLI and pipeline toggle for procedural ffmpeg-only television overlay effects.
- Added a post-stitch media pass that generates scanlines, vignette, noise, bezel shading, and screen padding without depending on final television assets yet.
- Added focused tests covering CLI wiring, pipeline manifest propagation, and media filter construction.

**Acceptance Criteria:**

- The television overlay assets are reusable across `from-text` and `from-audio`.
- The playable screen area is defined in metadata rather than hard-coded ad hoc.
- The resulting video keeps the generated scene content readable inside the frame.
- The overlay pipeline remains deterministic and does not require model regeneration during render.
- Visual effects can be tuned in ffmpeg without re-exporting the base television assets.

**Next Implementation Step:**

Add an initial bundled overlay pack under project assets, replace the temporary procedural-only layout with asset-backed screen geometry, then apply bezel/glare layers before final muxing.

## Backlog

### Candidate Milestones

- Add support for multiple display frame themes such as CRT, projector, film gate, and smartphone playback.
- Add per-project overlay selection through CLI or config.
- Add documentation and preview renders for bundled visual overlay packs.

## Completed Milestones

_No completed milestones recorded yet._

## Milestone Template

Copy this block for new milestones:

```md
### YYYY-MM-DD: Milestone Title

**Status:** Planned | In Progress | Blocked | Completed

**Goal:**
Short statement of the intended outcome.

**Decision:**
Summary of the chosen approach.

**Why This Approach:**

- Reason 1
- Reason 2

**Scope:**

- In scope item
- In scope item

**Acceptance Criteria:**

- Measurable success condition
- Measurable success condition

**Next Implementation Step:**

Concrete next action.
```
