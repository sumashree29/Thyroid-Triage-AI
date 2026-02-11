document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('triage-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultPanel = document.getElementById('result-panel');
    const placeholderState = document.getElementById('placeholder-state');
    const resultsContent = document.getElementById('results-content');

    // Result elements
    const categoryEl = document.getElementById('res-category');
    const scoreEl = document.getElementById('res-score');
    const scoreRingEl = document.getElementById('score-ring');
    const summaryEl = document.getElementById('res-summary');
    const reportEl = document.getElementById('res-full-report');
    const confidenceEl = document.getElementById('res-confidence');
    const idEl = document.getElementById('res-id');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI - Set Loading
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;

        // Collect Data
        const formData = new FormData(form);
        const data = {
            patient_id: "P-" + Math.floor(Math.random() * 10000).toString().padStart(4, '0'), // Auto-generate ID
            patient_data: {
                age: parseFloat(formData.get('age')),
                sex: formData.get('sex'),
                tsh: parseFloat(formData.get('tsh')),
                t3: formData.get('t3') ? parseFloat(formData.get('t3')) : null,
                tt4: formData.get('tt4') ? parseFloat(formData.get('tt4')) : null,
                t4u: formData.get('t4u') ? parseFloat(formData.get('t4u')) : null,
                fti: formData.get('fti') ? parseFloat(formData.get('fti')) : null,
            },
            audience: formData.get('audience'),
            include_full_report: formData.get('include_full_report') === 'on'
        };

        try {
            // API Call
            // Note: Since we are serving this from the API itself, relative path works
            const response = await fetch('/triage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const result = await response.json();

            // Display Results
            showResults(result);

        } catch (error) {
            alert("Error running analysis: " + error.message);
            console.error(error);
        } finally {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    });

    function showResults(data) {
        // Store data for view toggling
        window.latestResultData = data;

        // Toggle View
        placeholderState.style.display = 'none';
        resultsContent.classList.remove('hidden');
        resultsContent.style.display = 'flex';

        // Fill Data
        categoryEl.textContent = data.triage_category.replace('_', ' ');

        // Color Coding
        categoryEl.className = ''; // reset
        scoreRingEl.className = 'circle'; // reset base class
        if (data.triage_category === 'URGENT') {
            categoryEl.classList.add('urgent');
            scoreRingEl.classList.add('urgent');
        } else if (data.triage_category === 'HIGH_PRIORITY') {
            categoryEl.classList.add('high');
            scoreRingEl.classList.add('high');
        } else {
            categoryEl.classList.add('routine');
            scoreRingEl.classList.add('routine');
        }

        // Animate Score
        const scorePct = Math.round(data.risk_score * 100);
        animateValue(scoreEl, 0, scorePct, 1500);

        // Animate Ring (stroke-dasharray="current, 100")
        // The circle path length is approx 100 due to normalized viewbox
        // We set stroke-dasharray to allow CSS transition
        scoreRingEl.setAttribute('stroke-dasharray', `${scorePct}, 100`);

        // Text Content - Start with Patient View
        summaryEl.textContent = data.summary || "No summary provided.";
        reportEl.textContent = data.full_report || "No detailed report requested.";
        confidenceEl.textContent = (data.confidence * 100).toFixed(1) + "%";
        idEl.textContent = data.patient_id;

        // Ensure patient view is active
        document.getElementById('btn-patient-view').classList.add('active');
        document.getElementById('btn-doctor-view').classList.remove('active');
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Store latest result data for view toggling
    window.latestResultData = null;

    // Toggle View Functions (make them global)
    window.showPatientView = function () {
        document.getElementById('btn-patient-view').classList.add('active');
        document.getElementById('btn-doctor-view').classList.remove('active');

        if (window.latestResultData) {
            // Show patient-friendly summary
            summaryEl.textContent = window.latestResultData.summary || "No summary available";
        }
    };

    window.showDoctorView = function () {
        document.getElementById('btn-doctor-view').classList.add('active');
        document.getElementById('btn-patient-view').classList.remove('active');

        if (window.latestResultData) {
            // Show detailed clinical report
            summaryEl.textContent = window.latestResultData.full_report || "No detailed report available";
        }
    };

    // Check Health on Load
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            const statusEl = document.getElementById('system-status');
            statusEl.textContent = data.status === 'healthy' ? 'Online' : 'Degraded';
            statusEl.style.color = data.status === 'healthy' ? 'var(--success)' : 'var(--warning)';
        })
        .catch(err => {
            const statusEl = document.getElementById('system-status');
            statusEl.textContent = 'Offline';
            statusEl.style.color = 'var(--danger)';
        });
});
