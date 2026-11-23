---
name: Precision Editorial Utility
colors:
  surface: '#f8f9fb'
  surface-dim: '#d9dadc'
  surface-bright: '#f8f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#edeef0'
  surface-container-high: '#e7e8ea'
  surface-container-highest: '#e1e2e4'
  on-surface: '#191c1e'
  on-surface-variant: '#464555'
  inverse-surface: '#2e3132'
  inverse-on-surface: '#f0f1f3'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#575e70'
  on-secondary: '#ffffff'
  secondary-container: '#d9dff5'
  on-secondary-container: '#5c6274'
  tertiary: '#414855'
  on-tertiary: '#ffffff'
  tertiary-container: '#59606e'
  on-tertiary-container: '#d4dbeb'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#dce2f7'
  secondary-fixed-dim: '#c0c6db'
  on-secondary-fixed: '#141b2b'
  on-secondary-fixed-variant: '#404758'
  tertiary-fixed: '#dce2f3'
  tertiary-fixed-dim: '#c0c7d6'
  on-tertiary-fixed: '#151c27'
  on-tertiary-fixed-variant: '#404754'
  background: '#f8f9fb'
  on-background: '#191c1e'
  surface-variant: '#e1e2e4'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.005em
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.015em
  label-xs:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
  metric-display:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.25rem
  margin: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

The design system embodies a calm, precise, and analytical workspace tailored for trust-and-safety officers, marketplace integrity teams, and machine learning engineers. The visual identity draws inspiration from the disciplined utilitarianism of Linear and Vercel alongside the typographic restraint of Notion. 

Key attributes:
- **Tone:** Objective, surgical, and authoritative. It presents critical algorithmic classifications without hyperbole.
- **Atmosphere:** Deep focus, low visual noise, structural rhythm, and high signal-to-noise ratio.
- **Style Direction:** Modern Utility Minimalism. The interface relies on hairline structural dividers, low-chroma containment, strict baseline alignment, and a monochrome structural base, punctuated only by a focused primary indigo and crisp semantic alerts.

## Colors

The palette employs a hyper-controlled monochromatic canvas anchored by a single vibrant primary accent and dedicated semantic indicators for ML model outputs.

- **Base Canvas:** Pure white (`#FFFFFF`) forms the foundational surface for the deepest page layers.
- **Card & Container Fill:** Crisp off-white (`#F7F8FA`) defines grouped panels, data cards, and inset tables, preserving soft contrast against pure white without relying on heavy shadows.
- **Text & Content Hierarchy:** Primary text sits at deep charcoal (`#111827`), secondary metadata and captions at balanced slate (`#6B7280`), and tertiary/placeholder states at soft stone (`#9CA3AF`).
- **Primary Interactive:** Indigo (`#4F46E5`) is reserved exclusively for primary actions, active state markers, focused inputs, and targeted highlights.
- **Semantic ML Status:**
  - **Fake (Anomalous):** `#DC2626` (Red 600) with a 10% alpha fill (`#DC26261A`) for tags and score chips.
  - **Suspicious (Drift/Review):** `#D97706` (Amber 600) with a 10% alpha fill (`#D977061A`).
  - **Genuine (Verified):** `#16A34A` (Green 600) with a 10% alpha fill (`#16A34A1A`).
- **Dividers & Strokes:** `#E5E7EB` (Gray 200) strictly rendered at 1px width.

## Typography

The type system is strictly single-family (`Inter`) to eliminate visual friction and mirror the precision of developer workflows and data-dense dashboards.

- **Numbers & Metrics:** Use `font-feature-settings: 'cv02', 'cv03', 'cv04', 'tnum'` across data tables, confidence scores, and probability indices to ensure tabular numeric alignment.
- **Hierarchy:** High visual contrast is achieved through deliberate weight steps (`400`, `500`, `600`) and tight, modern line heights rather than excessive font-size scaling.
- **Case Rules:** All micro-labels (`label-xs`) used for categorization, ML feature weights, and table header fields should use uppercase formatting with +0.04em letter-spacing.

## Layout & Spacing

The layout is built around a desktop-first, fixed maximum width canvas of 1440px with fluid internal content distribution.

- **Grid Model:** 12-column grid with a 1.25rem (20px) gutter and 2rem (32px) outer margins on desktop viewports. Structural content is aligned along vertical keylines demarcated by 1px `#E5E7EB` borders.
- **Spatial Rhythm:** A 4px/8px incremental rhythm is strictly observed. Vertical pacing inside cards prioritizes compactness: 12px (`space-md`) between headers and stats, 20px (`space-lg`) between card sections.
- **Responsive Adaptations:**
  - **Desktop (1440px+):** Pinned sidebar (240px fixed width), 12-column flex body.
  - **Tablet (768px – 1024px):** Collapsed icon-rail sidebar (64px fixed), 8-column layout, gutter drops to 1rem (16px), margin to 1.5rem (24px).
  - **Mobile (<768px):** Off-canvas drawer navigation, single-column vertical stack, gutter and margin set to 1rem (16px).

## Elevation & Depth

This system avoids blurred drop shadows, ambient colored glows, or heavy blurs. Depth is conveyed entirely through tonal layering, crisp borders, and subtle contrast changes.

- **Base Layer:** Deepest canvas layer (`#FFFFFF`).
- **Container Layer:** Card surfaces, data wells, and sub-panels set to `#F7F8FA` with a perimeter stroke of `1px solid #E5E7EB`.
- **Floating Overlays:** Menus, dropdowns, command palettes, and modals utilize `#FFFFFF` with a crisp outline (`1px solid #E5E7EB`) and an ultra-subtle directional shadow: `0 4px 12px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03)`.
- **Interactive State Depth:** Hovering over actionable cards or table rows is indicated by an outline tint change (`#D1D5DB`) or surface shift to pure white (`#FFFFFF`) rather than vertical elevation or scale changes.

## Shapes

The design system enforces a calibrated corner radius strategy anchored at 12px for primary structural containers:

- **Cards, Panels & Data Containers:** Fixed at `12px` (equivalent to `rounded-lg` under Level 2).
- **Interactive Controls (Inputs, Buttons, Badges):** Standardized at `8px` (`0.5rem`) to maintain proportional balance within 12px parent containers.
- **Micro-indicators & Pills:** Semantic status badges and confidence score tags use a `4px` or full pill (`9999px`) treatment depending on their contextual density.
- **Icon containers:** 32x32px or 24x24px squares with a consistent `6px` radius.

## Components

### Buttons
- **Primary:** Background `#4F46E5`, text `#FFFFFF`, radius 8px, padding 8px 16px, font `label-md`. Hover: `#4338CA`. Active: `#3730A3`. Focus: 2px ring of `#4F46E5` with 2px offset.
- **Secondary / Outline:** Background `#FFFFFF`, border `1px solid #E5E7EB`, text `#111827`. Hover: `#F7F8FA`, border `#D1D5DB`.
- **Ghost:** Background transparent, text `#6B7280`. Hover: background `#F7F8FA`, text `#111827`.

### Status Badges & Chips
- Designed for displaying ML classification predictions.
- **Fake:** Border `1px solid #FECACA`, background `#FEF2F2`, text `#DC2626`, font `label-xs`, padding 2px 8px, radius 9999px. Optional leading dot (6px solid `#DC2626`).
- **Suspicious:** Border `1px solid #FDE68A`, background `#FFFBEB`, text `#D97706`, font `label-xs`, padding 2px 8px, radius 9999px.
- **Genuine:** Border `1px solid #BBF7D0`, background `#F0FDF4`, text `#16A34A`, font `label-xs`, padding 2px 8px, radius 9999px.

### Form Inputs & Search Fields
- Height 36px, background `#FFFFFF`, border `1px solid #E5E7EB`, radius 8px, padding 0 12px, font `body-md`.
- Placeholder color: `#9CA3AF`.
- Focus state: border `#4F46E5`, outline none, box-shadow `0 0 0 1px #4F46E5`.
- Prepend/append thin-stroke Lucide icons (stroke width 1.5px, size 16px, color `#6B7280`).

### Checkboxes & Radios
- Size 16x16px, border `1px solid #D1D5DB`, radius 4px (checkbox) or 50% (radio), background `#FFFFFF`.
- Checked state: background `#4F46E5`, border `#4F46E5`, checkmark/radio dot in `#FFFFFF`.

### Data Tables
- Encased in a parent container with background `#FFFFFF`, border `1px solid #E5E7EB`, and `12px` rounded corners.
- **Header:** Height 36px, background `#F7F8FA`, border-bottom `1px solid #E5E7EB`, text `label-xs` in `#6B7280`, uppercase with tabular alignment.
- **Rows:** Height 44px, border-bottom `1px solid #F3F4F6`, alternating row fills disabled. Hover state: background `#F9FAFB`.
- **Text:** Primary columns `body-sm` (`#111827`), numeric ML scores `label-sm` with tabular numerals (`#111827`).

### Minimal ML Metric Cards
- Background `#F7F8FA`, border `1px solid #E5E7EB`, radius 12px, padding 16px.
- Header row: Label in `label-xs` (`#6B7280`), trailing 16px Lucide utility icon (`#9CA3AF`).
- Metric value: `metric-display` (`#111827`) paired with a micro-badge indicating confidence delta or confusion matrix metrics (precision/recall).

### Classical ML Specifics: Feature Importance & Charts
- **Horizontal Bar Feature Weights:** Crisp 6px-height bars inside `#E5E7EB` tracks. Positive feature contributions use `#4F46E5`, negative/risk contributors use `#DC2626`.
- **Charts:** Zero-fill sparklines and ROC/AUC curves with a 1.5px hairline stroke, no background gradients, and muted grid ticks (`#E5E7EB`).