pub(super) const CSS: &str = r#"
:root {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #17202a;
    background: #edf4ef;
}

body {
    margin: 0;
}

button, input {
    font: inherit;
}

.app {
    min-height: 100vh;
    background: #edf4ef;
}

.shell {
    width: min(1180px, calc(100vw - 32px));
    margin: 0 auto;
    padding: 20px 0 40px;
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
    font-size: 24px;
    line-height: 1.2;
    font-weight: 760;
}

.subtitle {
    margin: 6px 0 0;
    color: #53645c;
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
    color: #53645c;
    font-size: 13px;
    font-weight: 700;
}

.directory-label {
    color: #53645c;
    font-size: 12px;
    max-width: 360px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.button {
    border: 1px solid #28533f;
    background: #28533f;
    color: #fff;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
}

.button.secondary {
    background: #fff;
    color: #28533f;
}

.button:disabled {
    opacity: 0.55;
    cursor: default;
}

.panel {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #cfdcd3;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
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
    border: 1px solid #d3ded7;
    border-radius: 6px;
    background: #fbfdfb;
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
    flex: 1 1 170px;
    min-width: 160px;
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 10px;
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
    gap: 8px;
    min-height: 38px;
    color: #17202a;
    font-size: 13px;
}

.checkbox-field input {
    width: 16px;
    height: 16px;
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
    box-shadow: 0 0 0 2px rgba(23, 32, 42, 0.12);
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

    .layer-row {
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
