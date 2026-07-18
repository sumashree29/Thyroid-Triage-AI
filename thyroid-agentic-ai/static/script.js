document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('triage-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const placeholderState = document.getElementById('placeholder-state');
    const resultsContent = document.getElementById('results-content');

    // ── Risk Header elements ──
    const categoryEl = document.getElementById('res-category');
    const scoreEl = document.getElementById('res-score');
    const scoreRingEl = document.getElementById('score-ring');
    const idLabelEl = document.getElementById('res-id-label');

    // ── Stat chips ──
    const chipRisk = document.getElementById('chip-risk');
    const chipConfidence = document.getElementById('chip-confidence');
    const chipPredSet = document.getElementById('chip-pred-set');
    const chipCoverage = document.getElementById('chip-coverage');
    const chipPredSetWrap = document.getElementById('chip-pred-set-wrap');
    const chipCoverageWrap = document.getElementById('chip-coverage-wrap');

    // ── Card bodies ──
    const impressionEl = document.getElementById('res-impression');
    const confounderBody = document.getElementById('res-confounder-body');
    const demographicsEl = document.getElementById('res-demographics');
    const findingsTbody = document.getElementById('findings-tbody');
    const recommendationsEl = document.getElementById('res-recommendations');
    const citationsCountEl = document.getElementById('citations-count');
    const citationsListEl = document.getElementById('res-citations');
    const uncertaintyListEl = document.getElementById('res-uncertainty');
    const patientSummaryEl = document.getElementById('res-patient-summary');

    // ── Doctor-only card IDs ──
    const doctorCards = [
        'card-impression', 'card-confounder', 'card-findings',
        'card-recommendations', 'card-citations', 'card-uncertainty'
    ];

    // Store latest result for view toggling
    window.latestResultData = null;

    // ════════════════════════════════════════════════════
    // FORM SUBMISSION
    // ════════════════════════════════════════════════════
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;

        const fd = new FormData(form);
        const payload = {
            patient_id: 'P-' + Math.floor(Math.random() * 10000).toString().padStart(4, '0'),
            patient_data: {
                age: parseFloat(fd.get('age')),
                sex: fd.get('sex'),
                tsh: parseFloat(fd.get('tsh')),
                t3:  fd.get('t3')  ? parseFloat(fd.get('t3'))  : null,
                tt4: fd.get('tt4') ? parseFloat(fd.get('tt4')) : null,
                t4u: fd.get('t4u') ? parseFloat(fd.get('t4u')) : null,
                fti: fd.get('fti') ? parseFloat(fd.get('fti')) : null,
            },
            audience: fd.get('audience'),
            include_full_report: fd.get('include_full_report') === 'on'
        };

        try {
            const res = await fetch('/triage', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Analysis failed');
            }
            showResults(await res.json());
        } catch (err) {
            alert('Error running analysis: ' + err.message);
            console.error(err);
        } finally {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    });

    // ════════════════════════════════════════════════════
    // RENDER RESULTS
    // ════════════════════════════════════════════════════
    function showResults(data) {
        window.latestResultData = data;

        // Show results panel
        placeholderState.style.display = 'none';
        resultsContent.classList.remove('hidden');
        resultsContent.style.display = 'flex';

        // ── 4a. Risk Header ──
        const cat = data.triage_category || 'ROUTINE';
        categoryEl.textContent = cat.replace('_', ' ');
        categoryEl.className = 'triage-badge';
        scoreRingEl.className = 'circle';

        if (cat === 'URGENT') {
            categoryEl.classList.add('urgent');
            scoreRingEl.classList.add('urgent');
        } else if (cat === 'HIGH_PRIORITY' || cat === 'HIGH PRIORITY') {
            categoryEl.classList.add('high');
            scoreRingEl.classList.add('high');
        } else {
            categoryEl.classList.add('routine');
            scoreRingEl.classList.add('routine');
        }

        const pct = Math.round(data.risk_score * 100);
        animateValue(scoreEl, 0, pct, 1200);
        scoreRingEl.setAttribute('stroke-dasharray', `${pct}, 100`);

        idLabelEl.textContent = data.patient_id || '';

        // Stat chips
        chipRisk.textContent = (data.risk_score * 100).toFixed(1) + '%';
        chipConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';

        if (data.conformal_set) {
            chipPredSet.textContent = '{' + data.conformal_set.prediction_set.join(', ') + '}';
            chipCoverage.textContent = (data.conformal_set.coverage_level * 100).toFixed(0) + '%';
            chipPredSetWrap.style.display = '';
            chipCoverageWrap.style.display = '';
        } else {
            chipPredSetWrap.style.display = 'none';
            chipCoverageWrap.style.display = 'none';
        }

        // ── 4b. Clinical Impression ──
        impressionEl.textContent = data.clinical_impression || 'No impression available.';

        // ── 4c. Confounder Analysis ──
        if (data.confounder_flags && data.confounder_flags.length > 0) {
            confounderBody.innerHTML = data.confounder_flags.map(f => `
                <div class="confounder-flag">
                    <strong>⚠ ${esc(f.interference_type || 'Confounder detected')}</strong>
                    <span class="conf-badge">${esc(f.confidence || 'unknown')}</span>
                    <div class="conf-detail">${esc(f.recommended_follow_up || '')}</div>
                </div>
            `).join('');
        } else {
            confounderBody.innerHTML = '<span class="safe-line">✓ No immunoassay interferences detected</span>';
        }

        // ── 4d. Key Findings ──
        let demoText = '';
        const findings = data.key_findings || [];
        const clinicalFindings = [];

        findings.forEach(f => {
            if (f.status === 'info') {
                demoText += (demoText ? ' · ' : '') + f.marker + ': ' + f.value;
            } else {
                clinicalFindings.push(f);
            }
        });

        demographicsEl.textContent = demoText;

        if (clinicalFindings.length > 0) {
            findingsTbody.innerHTML = clinicalFindings.map(f => {
                const dir = (f.direction || f.status || '').toLowerCase();
                const rangeStr = f.normal_range && f.normal_range !== 'N/A'
                    ? `${f.value} (${f.normal_range})`
                    : f.value;
                return `<tr>
                    <td>${esc(f.marker)}</td>
                    <td>${esc(rangeStr)}</td>
                    <td><span class="direction-badge ${dir}">${esc(f.direction || f.status)}</span></td>
                </tr>`;
            }).join('');
            document.getElementById('card-findings').style.display = '';
        } else {
            document.getElementById('card-findings').style.display = 'none';
        }

        // ── 4e. Recommendations ──
        const recs = data.recommendations || [];
        if (recs.length > 0) {
            recommendationsEl.innerHTML = recs.map(r => `<li>${esc(r)}</li>`).join('');
            document.getElementById('card-recommendations').style.display = '';
        } else {
            document.getElementById('card-recommendations').style.display = 'none';
        }

        // ── 4f. Citations ──
        const cites = data.evidence_citations || [];
        citationsCountEl.textContent = cites.length + ' supporting guideline citation' + (cites.length !== 1 ? 's' : '');

        if (cites.length > 0) {
            citationsListEl.innerHTML = cites.map(c => `
                <li class="citation-item">
                    <span class="citation-source-badge">${esc(c.source)}</span>
                    <span class="citation-category">${esc(c.category)}</span>
                    <div style="margin-top:4px;">${esc(c.text)}</div>
                </li>
            `).join('');
            document.getElementById('card-citations').style.display = '';
        } else {
            document.getElementById('card-citations').style.display = 'none';
        }

        // ── 4g. Uncertainty ──
        const notes = data.uncertainty_notes || [];
        if (notes.length > 0) {
            const isSafe = notes.length === 1 && notes[0].includes('Reasonable confidence');
            uncertaintyListEl.className = 'uncertainty-list' + (isSafe ? ' is-safe' : '');
            uncertaintyListEl.innerHTML = notes.map(n => `<li>${esc(n)}</li>`).join('');
            document.getElementById('card-uncertainty').style.display = '';
        } else {
            document.getElementById('card-uncertainty').style.display = 'none';
        }

        // ── Patient summary (hidden by default) ──
        if (data.summary) {
            patientSummaryEl.textContent = data.summary;
        }

        // Default to patient view
        showPatientView();
    }

    // ════════════════════════════════════════════════════
    // VIEW TOGGLING
    // ════════════════════════════════════════════════════
    window.showPatientView = function () {
        document.getElementById('btn-patient-view').classList.add('active');
        document.getElementById('btn-doctor-view').classList.remove('active');

        // Hide doctor-detail cards, show patient summary
        doctorCards.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        document.getElementById('card-patient-summary').style.display = '';
    };

    window.showDoctorView = function () {
        document.getElementById('btn-doctor-view').classList.add('active');
        document.getElementById('btn-patient-view').classList.remove('active');

        // Show doctor-detail cards, hide patient summary
        doctorCards.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = '';
        });
        document.getElementById('card-patient-summary').style.display = 'none';

        // Re-check data-driven visibility
        const d = window.latestResultData;
        if (d) {
            const cf = d.key_findings || [];
            if (cf.filter(f => f.status !== 'info').length === 0) {
                document.getElementById('card-findings').style.display = 'none';
            }
            if (!(d.recommendations && d.recommendations.length)) {
                document.getElementById('card-recommendations').style.display = 'none';
            }
            if (!(d.evidence_citations && d.evidence_citations.length)) {
                document.getElementById('card-citations').style.display = 'none';
            }
            if (!(d.uncertainty_notes && d.uncertainty_notes.length)) {
                document.getElementById('card-uncertainty').style.display = 'none';
            }
        }
    };

    // ════════════════════════════════════════════════════
    // HELPERS
    // ════════════════════════════════════════════════════
    function animateValue(el, start, end, duration) {
        let t0 = null;
        const step = (ts) => {
            if (!t0) t0 = ts;
            const p = Math.min((ts - t0) / duration, 1);
            el.innerHTML = Math.floor(p * (end - start) + start) + '%';
            if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    function esc(str) {
        if (str == null) return '';
        const d = document.createElement('div');
        d.textContent = String(str);
        return d.innerHTML;
    }

    // ════════════════════════════════════════════════════
    // HEALTH CHECK
    // ════════════════════════════════════════════════════
    fetch('/health')
        .then(r => r.json())
        .then(d => {
            const el = document.getElementById('system-status');
            el.textContent = d.status === 'healthy' ? 'Online' : 'Degraded';
            el.style.color = d.status === 'healthy' ? 'var(--signal-safe-border)' : 'var(--signal-medium-border)';
        })
        .catch(() => {
            const el = document.getElementById('system-status');
            el.textContent = 'Offline';
            el.style.color = 'var(--signal-danger-border)';
        });
});
