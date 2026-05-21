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
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
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

.controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px auto;
    gap: 12px;
    align-items: end;
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.field label {
    display: block;
    margin-bottom: 5px;
    font-size: 12px;
    color: #53645c;
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

.heatmap-groups {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.heatmap-card {
    min-width: 0;
}

.heatmap-label {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: #314238;
    font-size: 12px;
    margin-bottom: 4px;
}

.heatmap {
    display: grid;
    height: 120px;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    overflow: hidden;
    background: #f7faf8;
}

.cell {
    min-width: 1px;
    min-height: 1px;
}

.layer {
    margin-top: 16px;
}

@media (max-width: 820px) {
    .topbar, .controls {
        display: block;
    }

    .actions {
        justify-content: flex-start;
        margin-top: 12px;
    }

    .status-grid, .heatmap-groups, .samples, .document-list {
        grid-template-columns: 1fr;
    }

    .search-row, .sample {
        grid-template-columns: 1fr;
    }

    .field {
        margin-bottom: 10px;
    }
}
"#;
