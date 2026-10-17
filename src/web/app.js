(() => {
  const d = window.PEGASOLD_DATA;
  if (!d) {
    document.querySelectorAll('[data-needs-data]').forEach((el) => { el.textContent = 'Experiment results are unavailable. Run python -m src.run_all.'; });
    return;
  }
  const byId = (id) => document.getElementById(id);
  const fmt = (x, n = 3) => Number.isFinite(Number(x)) ? Number(x).toFixed(n) : '—';
  const int = (x) => Number(x).toLocaleString('en-US');
  const pct = (x) => `${(Number(x) * 100).toFixed(1)}%`;
  const dollars = (x) => `${int(Math.round(x))} AZN`;
  const m = d.metrics;

  const set = (id, value) => { const el = byId(id); if (el) el.textContent = value; };
  set('test-pill', d.test_summary.replace(/\s+in\s+[\d.]+s$/, ''));
  set('footer-test', d.test_summary);
  set('threshold-inline', int(d.threshold_AZN));
  set('threshold-report', int(d.threshold_AZN));
  set('split-train', `${int(d.split.train)} listings; fits preprocessing and models`);
  set('split-validation', `${int(d.split.validation)} listings; selects model settings`);
  set('split-test', `${int(d.split.test)} listings; final score only`);
  set('split-features', `${d.feature_count} columns (${d.numeric_feature_count} numeric/binary plus encoded categories)`);
  set('test-report', d.test_summary);
  set('snapshot-rows', int(d.final_rows));
  set('snapshot-features', d.feature_count);
  set('snapshot-tests', (d.test_summary.match(/\d+/) || ['—'])[0]);
  set('seed-value', d.seed);
  set('metric-accuracy', pct(m.decision_tree_classification.accuracy));
  set('metric-tree-f1', fmt(m.decision_tree_classification.f1));
  set('metric-tree-auc', fmt(m.decision_tree_classification.roc_auc));
  set('metric-svm-acc', pct(m.pegasos_svm.accuracy));
  set('metric-svm-f1', fmt(m.pegasos_svm.f1));
  set('metric-svm-auc', fmt(m.pegasos_svm.roc_auc));
  set('metric-rmse', fmt(m.decision_tree_regression.rmse));
  set('metric-rmse-azn', dollars(m.decision_tree_regression.rmse_AZN));
  set('metric-mae-azn', dollars(m.decision_tree_regression.mae_AZN));
  set('metric-clean', int(d.final_rows));
  set('metric-features', d.feature_count);

  const menu = document.querySelector('.menu-toggle');
  const nav = document.querySelector('.main-nav');
  menu?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(open));
  });
  nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => nav.classList.remove('open')));

  const toast = (message) => {
    const el = byId('toast');
    if (!el) return;
    el.textContent = message;
    el.classList.add('show');
    window.setTimeout(() => el.classList.remove('show'), 1800);
  };
  byId('copy-command')?.addEventListener('click', async (event) => {
    try {
      await navigator.clipboard.writeText('python -m src.run_all');
      event.currentTarget.textContent = 'Copied';
      toast('Run command copied');
      window.setTimeout(() => { event.currentTarget.textContent = 'Copy'; }, 1500);
    } catch { toast('Command: python -m src.run_all'); }
  });

  const dialog = byId('lightbox');
  document.querySelectorAll('[data-lightbox]').forEach((card) => card.addEventListener('click', () => {
    const image = dialog?.querySelector('img');
    if (!dialog || !image) return;
    image.src = card.dataset.lightbox;
    image.alt = card.querySelector('img')?.alt || 'Expanded project chart';
    dialog.showModal();
  }));
  dialog?.querySelector('.lightbox-close')?.addEventListener('click', () => dialog.close());
  dialog?.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });

  const fmtTable = (x, mode) => mode === 'pct' ? pct(x) : mode === 'money' ? dollars(x) : mode === 'time' ? `${fmt(x, 2)}s` : fmt(x);
  const table = byId('classification-table');
  if (table) {
    const names = [
      ['decision_tree_classification', 'NumPy decision tree', 'scratch'],
      ['sklearn_tree_classification', 'DecisionTreeClassifier', 'reference'],
      ['pegasos_svm', 'NumPy Pegasos SVM', 'scratch'],
      ['linear_svc', 'LinearSVC', 'reference'],
      ['rff_pegasos_svm', 'NumPy RFF Pegasos', 'scratch'],
      ['rbf_svc', 'RBF SVC', 'reference'],
    ];
    table.innerHTML = `<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Fit</th><th>Predict</th></tr></thead><tbody>${names.map(([key, label, cls]) => {
      const x = m[key];
      return `<tr class="${cls}"><td>${label}</td><td>${fmtTable(x.accuracy, 'pct')}</td><td>${fmt(x.precision)}</td><td>${fmt(x.recall)}</td><td>${fmt(x.f1)}</td><td>${fmt(x.roc_auc)}</td><td>${fmtTable(x.fit_seconds, 'time')}</td><td>${fmtTable(x.predict_seconds, 'time')}</td></tr>`;
    }).join('')}</tbody>`;
  }
  const regTable = byId('regression-table');
  if (regTable) {
    const names = [['decision_tree_regression', 'NumPy decision tree', 'scratch'], ['sklearn_tree_regression', 'DecisionTreeRegressor', 'reference']];
    regTable.innerHTML = `<thead><tr><th>Model</th><th>RMSE · log</th><th>MAE · log</th><th>R² · log</th><th>RMSE · AZN</th><th>MAE · AZN</th><th>R² · AZN</th><th>Fit</th></tr></thead><tbody>${names.map(([key, label, cls]) => { const x = m[key]; return `<tr class="${cls}"><td>${label}</td><td>${fmt(x.rmse)}</td><td>${fmt(x.mae)}</td><td>${fmt(x.r2)}</td><td>${dollars(x.rmse_AZN)}</td><td>${dollars(x.mae_AZN)}</td><td>${fmt(x.r2_AZN)}</td><td>${fmtTable(x.fit_seconds, 'time')}</td></tr>`; }).join('')}</tbody>`;
  }

  const cleanRows = [
    ['Raw source', d.raw_rows, 'Rows as supplied; source file stays unchanged.'],
    ['After advert deduplication', d.after_dedup, 'Normalize URL to numeric advert ID; keep latest dated scrape.'],
    ['After scope selection', d.after_scope, 'Baku and three residential listing categories.'],
    ['Final model-ready records', d.final_rows, 'Fixed domain filters applied before split.'],
  ];
  const cleanTable = byId('cleaning-table');
  if (cleanTable) cleanTable.innerHTML = `<thead><tr><th>Pipeline stage</th><th>Rows</th><th>Decision</th></tr></thead><tbody>${cleanRows.map((row) => `<tr><td>${row[0]}</td><td>${int(row[1])}</td><td>${row[2]}</td></tr>`).join('')}</tbody>`;

  const cvTable = byId('cv-table');
  if (cvTable) {
    const cvData = [
      ['Decision tree regression · log RMSE', d.cross_validation.decision_tree_regression],
      ['Decision tree classification · accuracy', d.cross_validation.decision_tree_classification],
      ['Pegasos SVM · accuracy', d.cross_validation.pegasos_svm],
    ];
    cvTable.innerHTML = `<thead><tr><th>Metric</th><th>NumPy model · mean ± SD</th><th>Reference · mean ± SD</th><th>Rows / folds</th></tr></thead><tbody>${cvData.map(([label, x]) => `<tr><td>${label}</td><td>${fmt(x.scratch.mean)} ± ${fmt(x.scratch.std)}</td><td>${fmt(x.sklearn.mean)} ± ${fmt(x.sklearn.std)}</td><td>${int(x.sample_n)} / 5</td></tr>`).join('')}</tbody>`;
  }

  document.querySelectorAll('[data-result-tab]').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('[data-result-tab]').forEach((b) => b.classList.toggle('active', b === button));
    document.querySelectorAll('[data-result-panel]').forEach((panel) => panel.hidden = panel.dataset.resultPanel !== button.dataset.resultTab);
  }));

  const bars = byId('importance-bars');
  if (bars) {
    const values = d.feature_importance.regression.slice(0, 8);
    const max = Math.max(...values.map((x) => x.importance));
    bars.innerHTML = values.map((x) => `<div class="feature-row"><span title="${x.feature}">${x.feature}</span><div class="feature-track"><i style="width:${Math.max(2, x.importance / max * 100)}%"></i></div><b>${(x.importance * 100).toFixed(1)}%</b></div>`).join('');
  }
  const errorTable = byId('error-table');
  if (errorTable) {
    const quartiles = d.regression_error_slices.filter((x) => x.slice.startsWith('training-price-quartile'));
    errorTable.innerHTML = `<thead><tr><th>Test subset (cut points learned from train)</th><th>n</th><th>Median actual price</th><th>MAE · AZN</th><th>MAE · log</th></tr></thead><tbody>${quartiles.map((x, i) => `<tr><td>Training-price quartile ${i + 1}</td><td>${int(x.n)}</td><td>${dollars(x.median_actual_AZN)}</td><td>${dollars(x.mae_AZN)}</td><td>${fmt(x.mae_log)}</td></tr>`).join('')}</tbody>`;
  }
  const backTop = byId('back-top');
  if (backTop) {
    window.addEventListener('scroll', () => backTop.classList.toggle('visible', window.scrollY > 450), { passive: true });
  }
})();
