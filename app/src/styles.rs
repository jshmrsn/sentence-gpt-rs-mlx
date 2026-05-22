pub(super) const CSS: &str = r#"
:root {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #f1ead8;
    background: #020403;
}

body {
    margin: 0;
}

button, input {
    font: inherit;
}

.app {
    min-height: 100vh;
    color: #f1ead8;
    background:
        radial-gradient(circle at 16% 12%, rgba(79, 216, 255, 0.16), transparent 24%),
        radial-gradient(circle at 58% 74%, rgba(224, 64, 251, 0.10), transparent 28%),
        linear-gradient(180deg, #020403 0%, #06090d 48%, #030506 100%);
    position: relative;
}

.shell {
    width: min(1460px, calc(100vw - 32px));
    margin: 0 auto;
    padding: 24px 0 48px;
    position: relative;
    z-index: 1;
}

.topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}

.title {
    margin: 0;
    font-size: 26px;
    line-height: 1.2;
    font-weight: 780;
    color: #f7eed8;
}

.subtitle {
    margin: 6px 0 0;
    color: #9d9889;
    font-size: 14px;
}

.actions {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
}

.primary-actions,
.snapshot-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    justify-content: flex-end;
}

.action-label {
    color: #b9b09a;
    font-size: 13px;
    font-weight: 700;
}

.directory-label {
    color: #b9b09a;
    font-size: 12px;
    max-width: 360px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.button {
    border: 1px solid rgba(79, 216, 255, 0.48);
    background: linear-gradient(180deg, rgba(79, 216, 255, 0.30), rgba(79, 216, 255, 0.12));
    color: #f7eed8;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
}

.button.secondary {
    background: rgba(3, 7, 9, 0.68);
    color: #f6d365;
    border-color: rgba(246, 211, 101, 0.42);
}

.button:disabled {
    opacity: 0.55;
    cursor: default;
}

.panel {
    background:
        linear-gradient(180deg, rgba(12, 17, 22, 0.88), rgba(6, 9, 13, 0.82)),
        radial-gradient(circle at 100% 0%, rgba(79, 216, 255, 0.08), transparent 36%);
    border: 1px solid rgba(240, 232, 208, 0.20);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow:
        0 24px 80px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
}

.config-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}

.metric {
    border-left: 3px solid #28533f;
    padding-left: 10px;
}

.metric-label {
    color: #53645c;
    font-size: 12px;
}

.metric-value {
    font-size: 18px;
    font-weight: 720;
}

.progress-track {
    height: 10px;
    border-radius: 999px;
    background: #d5e2d9;
    overflow: hidden;
    margin: 12px 0 8px;
}

.progress-fill {
    height: 100%;
    background: #28533f;
}

.chart {
    width: 100%;
    height: 180px;
    border: 1px solid rgba(240, 232, 208, 0.18);
    border-radius: 6px;
    background: #05080c;
    box-shadow: inset 0 0 36px rgba(0, 0, 0, 0.50);
}

.overview-circuit {
    position: relative;
    min-height: 0;
    overflow: visible;
    border: 1px solid rgba(240, 232, 208, 0.20);
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(0, 0, 0, 0.72), rgba(5, 8, 12, 0.94));
    padding: 18px;
    box-shadow: inset 0 0 64px rgba(0, 0, 0, 0.64);
}

.circuit-mainline {
    position: relative;
    display: grid;
    grid-template-columns: minmax(106px, 0.8fr) 34px minmax(106px, 0.86fr) 34px minmax(180px, 1.35fr) 34px minmax(170px, 1.2fr) 34px minmax(96px, 0.72fr);
    gap: 8px;
    align-items: center;
    z-index: 2;
}

.circuit-node {
    position: relative;
    min-height: 170px;
    border: 2px solid rgba(240, 232, 208, 0.50);
    border-radius: 8px;
    background: rgba(2, 4, 6, 0.58);
    box-shadow:
        0 0 28px rgba(0, 0, 0, 0.58),
        inset 0 0 22px rgba(240, 232, 208, 0.05);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    gap: 8px;
    padding: 12px;
    min-width: 0;
}

.circuit-node:hover,
.circuit-node:focus-within,
.overview-step:hover,
.overview-step:focus-within {
    z-index: 45;
}

.circuit-label {
    color: #f0e8d0;
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
}

.circuit-dim {
    color: #9d9889;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
}

.circuit-arrow {
    color: #b7ad92;
    font-size: 28px;
    font-weight: 900;
    text-align: center;
    text-shadow: 0 0 14px rgba(246, 211, 101, 0.30);
}

.token-matrix {
    display: grid;
    grid-template-columns: 16px;
    gap: 5px;
    justify-content: center;
    align-content: center;
}

.token-dot,
.logit-dot {
    width: 13px;
    height: 13px;
    border-radius: 999px;
    border: 2px solid #c7beaa;
    background: #05080c;
    box-shadow: 0 0 8px rgba(240, 232, 208, 0.12);
}

.token-dot.active {
    background: #f6d365;
    border-color: #f6d365;
    box-shadow: 0 0 16px rgba(246, 211, 101, 0.62);
}

.embedding-probe {
    display: grid;
    grid-template-columns: repeat(2, minmax(16px, 1fr));
    gap: 6px;
    align-content: center;
}

.embedding-probe-cell {
    min-height: 18px;
    border-radius: 3px;
    border: 1px solid rgba(79, 216, 255, 0.28);
    box-shadow: 0 0 12px rgba(79, 216, 255, 0.12);
}

.attention-bank {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    align-content: center;
}

.attention-head {
    display: grid;
    grid-template-columns: 42px 42px;
    gap: 8px;
    align-items: center;
    border: 1px solid rgba(240, 232, 208, 0.22);
    border-radius: 6px;
    padding: 7px;
    background: rgba(255, 255, 255, 0.025);
}

.attention-head-graph {
    position: relative;
    display: grid;
    grid-template-columns: repeat(2, 12px);
    gap: 7px;
}

.attention-head-graph::before,
.attention-head-graph::after {
    content: "";
    position: absolute;
    left: 5px;
    right: 5px;
    top: 20px;
    border-top: 1px solid rgba(240, 232, 208, 0.36);
    transform: rotate(28deg);
}

.attention-head-graph::after {
    transform: rotate(-28deg);
}

.attention-head-graph span {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    border: 1px solid #c7beaa;
    background: #05080c;
}

.attention-projections {
    display: grid;
    grid-template-columns: repeat(2, 18px);
    gap: 4px;
}

.attention-projections span {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border: 1px solid rgba(79, 216, 255, 0.42);
    border-radius: 3px;
    color: #4fd8ff;
    font-size: 10px;
    font-weight: 900;
    background: rgba(79, 216, 255, 0.08);
}

.mlp-mini-network {
    width: 100%;
    height: 110px;
}

.logit-strip {
    display: grid;
    gap: 6px;
    justify-content: center;
    align-content: center;
}

.logit-dot.boundary {
    background: #f0e8d0;
}

.circuit-footer {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 14px;
    color: #b9b09a;
    font-size: 12px;
    font-weight: 760;
    z-index: 2;
}

.overview-sections {
    display: grid;
    gap: 14px;
    min-width: 0;
}

.overview-section {
    border-top: 1px solid #d3ded7;
    padding-top: 12px;
    min-width: 0;
}

.overview-section:first-child {
    border-top: 0;
    padding-top: 0;
}

.overview-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 8px;
}

.overview-step {
    position: relative;
    flex: 1 1 170px;
    min-width: 160px;
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 10px 34px 10px 10px;
}

.overview-step-title {
    color: #53645c;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.overview-step-status {
    margin-top: 4px;
    color: #17202a;
    font-size: 16px;
    font-weight: 760;
}

.overview-step-details {
    display: grid;
    gap: 3px;
    margin-top: 8px;
    color: #53645c;
    font-size: 12px;
    line-height: 1.25;
}

.overview-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #28533f;
    font-size: 18px;
    font-weight: 800;
    min-width: 18px;
}

.layer-overview {
    margin-top: 14px;
    border-top: 1px solid #d3ded7;
    padding-top: 12px;
}

.layer-overview-header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
}

.layer-row {
    display: grid;
    grid-template-columns: 42px minmax(150px, 1fr) 18px minmax(150px, 1.2fr) 18px minmax(150px, 1.2fr) 22px;
    gap: 6px;
    align-items: stretch;
    margin-bottom: 7px;
}

.layer-label {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #e0ebe4;
    border: 1px solid #c8d8ce;
    border-radius: 6px;
    color: #17202a;
    font-size: 12px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

.layer-chunk {
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 8px;
    background: #fbfdfb;
}

.layer-chunk.norm {
    border-left: 4px solid #6b7280;
}

.layer-chunk.attention {
    border-left: 4px solid #1f6feb;
}

.layer-chunk.mlp {
    border-left: 4px solid #b7791f;
}

.layer-chunk-title {
    color: #53645c;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
}

.layer-chunk-main {
    margin-top: 3px;
    color: #17202a;
    font-size: 13px;
    font-weight: 760;
}

.layer-chunk-detail {
    margin-top: 4px;
    color: #53645c;
    font-size: 11px;
    line-height: 1.25;
}

.layer-arrow,
.layer-next-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #28533f;
    font-weight: 800;
}

.layer-stack {
    display: grid;
    gap: 10px;
}

.layer-visual-row {
    display: grid;
    grid-template-columns: 86px minmax(0, 1fr);
    gap: 10px;
    align-items: stretch;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 10px;
    min-width: 0;
}

.layer-visual-label {
    display: grid;
    gap: 8px;
    align-content: start;
}

.layer-mini-stat {
    color: #53645c;
    font-size: 10px;
    line-height: 1.25;
    font-variant-numeric: tabular-nums;
}

.layer-pipeline {
    display: grid;
    grid-template-columns: minmax(130px, 0.75fr) 18px minmax(160px, 1fr) 18px minmax(430px, 2.2fr);
    gap: 6px;
    align-items: stretch;
    min-width: 0;
}

.layer-stage,
.mlp-visual-group {
    position: relative;
    z-index: 1;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    background: #fbfdfb;
}

.layer-stage {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    align-items: center;
    min-width: 0;
    padding: 8px;
}

.layer-stage:hover,
.layer-stage:focus-within {
    z-index: 40;
}

.mlp-visual-group:hover,
.mlp-visual-group:focus-within {
    z-index: 30;
}

.layer-stage.no-params {
    background: #eef5f0;
}

.stage-copy {
    min-width: 0;
    padding-right: 24px;
}

.stage-info-control {
    position: absolute;
    top: 7px;
    right: 7px;
    z-index: 41;
}

.stage-info-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.76);
    background: rgba(255, 255, 255, 0.11);
    color: #ffffff;
    font-size: 11px;
    font-weight: 900;
    line-height: 1;
    cursor: help;
    padding: 0;
}

.stage-info-button:hover,
.stage-info-button:focus-visible {
    border-color: #ffffff;
    background: rgba(255, 255, 255, 0.20);
    outline: none;
}

.stage-info-popover {
    display: none;
    position: absolute;
    top: 24px;
    right: 0;
    width: min(340px, calc(100vw - 64px));
    max-height: 340px;
    overflow: auto;
    border: 1px solid rgba(246, 211, 101, 0.42);
    border-radius: 8px;
    background: rgba(3, 6, 9, 0.98);
    color: #f1ead8;
    padding: 12px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.44);
    z-index: 42;
}

.stage-info-control:hover .stage-info-popover,
.stage-info-control:focus-within .stage-info-popover {
    display: block;
}

.stage-info-title {
    color: #f6d365;
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.stage-info-popover p {
    margin: 0 0 8px;
    color: #d8cfb7;
    font-size: 12px;
    line-height: 1.45;
}

.stage-info-popover p:last-child {
    margin-bottom: 0;
}

.stage-meter {
    height: 5px;
    background: #dfe9e2;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
}

.stage-meter-fill {
    height: 100%;
    background: #1f6feb;
}

.mlp-visual-group {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 7px;
    padding: 8px;
    min-width: 0;
}

.mlp-group-title {
    color: #53645c;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
}

.mlp-subpipeline {
    display: grid;
    grid-template-columns: minmax(112px, 1fr) 14px minmax(112px, 1fr) 14px minmax(112px, 0.85fr) 14px minmax(112px, 1fr);
    gap: 5px;
    align-items: stretch;
    min-width: 0;
}

.layer-arrow.small {
    min-width: 14px;
    font-size: 12px;
}

.embedding-panel {
    min-width: 0;
}

.embedding-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    color: #53645c;
    font-size: 13px;
    margin-bottom: 10px;
}

.embedding-legend {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #53645c;
    font-size: 12px;
    margin-bottom: 10px;
}

.embedding-legend-ramp {
    width: 160px;
    height: 10px;
    border: 1px solid #cfdcd3;
    border-radius: 999px;
    background: linear-gradient(90deg, #c62828, #fbfdfb 50%, #1f6feb);
}

.embedding-scroll {
    width: 100%;
    max-width: 100%;
    max-height: min(520px, 62vh);
    overflow: auto;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    background: #fbfdfb;
    box-sizing: border-box;
}

.embedding-table {
    display: grid;
    gap: 1px;
    width: max-content;
    min-width: 100%;
    padding: 8px;
    align-items: center;
}

.embedding-corner,
.embedding-column-label,
.embedding-norm-label {
    color: #53645c;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    min-height: 14px;
    line-height: 14px;
}

.embedding-corner,
.embedding-token-label {
    position: sticky;
    left: 0;
    z-index: 2;
    background: #fbfdfb;
}

.embedding-corner,
.embedding-column-label,
.embedding-norm-label {
    position: sticky;
    top: 0;
    z-index: 3;
    background: #fbfdfb;
}

.embedding-corner {
    left: 0;
    z-index: 5;
}

.embedding-token-label {
    color: #17202a;
    font-size: 12px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    min-height: 12px;
    line-height: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.embedding-cell {
    width: 12px;
    height: 12px;
    border-radius: 2px;
}

.embedding-norm,
.embedding-norm-label {
    text-align: right;
    position: sticky;
    right: 0;
    background: #fbfdfb;
    z-index: 2;
}

.embedding-norm {
    color: #53645c;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
}

.embedding-norm-label {
    z-index: 5;
}

.controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px auto auto;
    gap: 12px;
    align-items: end;
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.disclosure-button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 0;
    background: transparent;
    color: #17202a;
    padding: 0;
    cursor: pointer;
}

.disclosure-button .section-title {
    margin: 0;
}

.disclosure-arrow {
    width: 14px;
    color: #53645c;
    font-size: 13px;
    font-weight: 800;
    line-height: 1;
}

.field label {
    display: block;
    margin-bottom: 5px;
    font-size: 12px;
    color: #53645c;
}

.checkbox-field {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 38px;
    color: #17202a;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    user-select: none;
}

.checkbox-field input {
    appearance: none;
    width: 19px;
    height: 19px;
    flex: 0 0 19px;
    margin: 0;
    border: 1px solid rgba(90, 214, 255, 0.58);
    border-radius: 5px;
    background: rgba(2, 5, 9, 0.9);
    background-position: center;
    background-repeat: no-repeat;
    background-size: 13px 13px;
    cursor: pointer;
    transition:
        background-color 120ms ease,
        border-color 120ms ease,
        opacity 120ms ease;
}

.checkbox-field input:checked {
    border-color: #f3cf6a;
    background-color: #f3cf6a;
    background-image: url("data:image/svg+xml,%3Csvg width='13' height='13' viewBox='0 0 13 13' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M2.3 6.75L5.1 9.45L10.8 3.25' stroke='%23020609' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
}

.checkbox-field input:hover:not(:disabled) {
    border-color: #7fe0ff;
}

.checkbox-field input:focus-visible {
    outline: 2px solid rgba(90, 214, 255, 0.74);
    outline-offset: 3px;
}

.checkbox-field input:disabled {
    cursor: not-allowed;
    opacity: 0.46;
}

.checkbox-field:has(input:checked) span {
    color: #f3cf6a;
}

.checkbox-field:has(input:disabled) {
    cursor: not-allowed;
}

.text-input {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #b9c8bf;
    border-radius: 6px;
    padding: 8px 10px;
    background: #fff;
}

.search-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    align-items: center;
}

.range {
    width: 100%;
}

.samples {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.sample {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    align-items: start;
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 8px 10px;
    min-height: 22px;
}

.sample-text {
    min-width: 0;
}

.sample-search-button {
    border: 1px solid #b9c8bf;
    background: #fff;
    color: #28533f;
    border-radius: 6px;
    padding: 5px 8px;
    white-space: nowrap;
}

.inspection-panel {
    margin-top: 12px;
    border-top: 1px solid #d3ded7;
    padding-top: 12px;
}

.inspection-token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 12px;
}

.inspection-token {
    position: relative;
    min-width: 28px;
    min-height: 30px;
    border: 1px solid rgba(23, 32, 42, 0.18);
    border-radius: 6px;
    color: #17202a;
    padding: 5px 7px 7px;
    cursor: pointer;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
}

.inspection-token.selected {
    border-color: #17202a;
}

.inspection-token-text {
    display: block;
    line-height: 1;
}

.inspection-prefix-marker {
    position: absolute;
    left: 7px;
    right: 7px;
    bottom: 1px;
    color: #17202a;
    font-size: 11px;
    line-height: 1;
    text-align: center;
}

.inspection-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    color: #53645c;
    font-size: 13px;
    margin-bottom: 10px;
}

.distribution-list {
    display: grid;
    gap: 5px;
}

.distribution-row {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr) 58px;
    gap: 8px;
    align-items: center;
}

.distribution-row.chosen .distribution-token {
    color: #17202a;
    font-weight: 800;
}

.distribution-token,
.distribution-value {
    color: #53645c;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}

.distribution-track {
    height: 12px;
    background: #dfe9e2;
    border-radius: 999px;
    overflow: hidden;
}

.distribution-fill {
    height: 100%;
    background: #1f6feb;
}

.distribution-value {
    text-align: right;
}

.document-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.document-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 12px;
}

.page-button {
    border: 1px solid #b9c8bf;
    background: #fff;
    color: #28533f;
    border-radius: 6px;
    padding: 5px 8px;
    cursor: pointer;
    min-width: 34px;
}

.document-item {
    display: grid;
    grid-template-columns: 46px minmax(0, 1fr);
    gap: 8px;
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 8px 10px;
}

.document-index {
    color: #53645c;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}

.document-text {
    color: #17202a;
    font-size: 13px;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.section-title {
    margin: 0 0 8px;
    font-size: 16px;
    font-weight: 720;
}

.model-summary {
    color: #53645c;
    font-size: 13px;
    margin-bottom: 12px;
}

.layer {
    margin-top: 16px;
}

.metric {
    border-left-color: #4fd8ff;
}

.metric-label,
.model-summary,
.overview-step-title,
.overview-step-details,
.layer-chunk-detail,
.layer-mini-stat,
.embedding-summary,
.embedding-legend,
.embedding-corner,
.embedding-column-label,
.embedding-norm-label,
.embedding-norm,
.field label,
.distribution-token,
.distribution-value,
.document-index {
    color: #aaa491;
}

.metric-value,
.overview-step-status,
.layer-label,
.layer-chunk-main,
.embedding-token-label,
.disclosure-button,
.checkbox-field,
.sample-text,
.document-text,
.section-title,
.distribution-row.chosen .distribution-token {
    color: #f1ead8;
}

.progress-track,
.distribution-track,
.stage-meter {
    background: rgba(240, 232, 208, 0.12);
}

.progress-fill,
.stage-meter-fill,
.distribution-fill {
    background: linear-gradient(90deg, #4fd8ff, #f6d365);
    box-shadow: 0 0 18px rgba(79, 216, 255, 0.28);
}

.overview-section {
    border-top-color: rgba(240, 232, 208, 0.16);
}

.overview-step,
.sample,
.document-item,
.layer-chunk,
.layer-stage,
.mlp-visual-group,
.embedding-scroll {
    background: rgba(5, 8, 12, 0.82);
    border-color: rgba(240, 232, 208, 0.17);
    box-shadow: inset 0 0 24px rgba(255, 255, 255, 0.018);
}

.overview-step {
    border-left: 3px solid rgba(246, 211, 101, 0.38);
}

.overview-arrow,
.layer-arrow,
.layer-next-arrow {
    color: #f6d365;
    text-shadow: 0 0 12px rgba(246, 211, 101, 0.28);
}

.layer-label {
    background: rgba(79, 216, 255, 0.10);
    border-color: rgba(79, 216, 255, 0.34);
}

.layer-chunk.norm {
    border-left-color: #aaa491;
}

.layer-chunk.attention {
    border-left-color: #4fd8ff;
}

.layer-chunk.mlp {
    border-left-color: #f6d365;
}

.layer-stage.no-params {
    background: rgba(8, 16, 14, 0.88);
}

.embedding-legend-ramp {
    border-color: rgba(240, 232, 208, 0.18);
    background: linear-gradient(90deg, #ff4d6d, #070a0e 50%, #4fd8ff);
}

.embedding-corner,
.embedding-token-label,
.embedding-column-label,
.embedding-norm-label,
.embedding-norm {
    background: #070a0e;
}

.text-input,
.range {
    color: #f1ead8;
    background: rgba(2, 4, 6, 0.72);
    border: 1px solid rgba(240, 232, 208, 0.22);
}

.text-input:focus {
    outline: 1px solid rgba(79, 216, 255, 0.72);
    box-shadow: 0 0 0 3px rgba(79, 216, 255, 0.12);
}

.sample-search-button,
.page-button {
    border: 1px solid rgba(246, 211, 101, 0.36);
    background: rgba(5, 8, 12, 0.76);
    color: #f6d365;
}

.inspection-panel {
    border-top-color: rgba(240, 232, 208, 0.16);
}

.inspection-token {
    color: #071018;
    border-color: rgba(240, 232, 208, 0.20);
}

.inspection-token.selected {
    border-color: #f6d365;
}

.inspection-prefix-marker {
    color: #071018;
}

@media (max-width: 820px) {
    .topbar, .controls {
        display: block;
    }

    .actions {
        align-items: flex-start;
        margin-top: 12px;
    }

    .primary-actions,
    .snapshot-actions {
        justify-content: flex-start;
    }

    .status-grid, .config-grid, .samples, .document-list {
        grid-template-columns: 1fr;
    }

    .search-row, .sample {
        grid-template-columns: 1fr;
    }

    .overview-flow {
        display: grid;
        grid-template-columns: 1fr;
    }

    .overview-arrow {
        min-height: 14px;
        transform: rotate(90deg);
    }

    .overview-circuit {
        padding: 14px;
    }

    .circuit-mainline {
        grid-template-columns: 1fr;
    }

    .circuit-arrow {
        transform: rotate(90deg);
    }

    .circuit-footer {
        margin-top: 14px;
    }

    .layer-row,
    .layer-visual-row,
    .layer-pipeline,
    .mlp-subpipeline {
        grid-template-columns: 1fr;
    }

    .layer-arrow {
        min-height: 12px;
        transform: rotate(90deg);
    }

    .layer-next-arrow {
        min-height: 14px;
    }

    .field {
        margin-bottom: 10px;
    }
}
"#;
