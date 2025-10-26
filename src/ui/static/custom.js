// Initialize KaTeX for LaTeX rendering
document.addEventListener('DOMContentLoaded', function() {
    // Render LaTeX in all elements with data-latex attribute
    document.querySelectorAll('[data-latex]').forEach(element => {
        try {
            const latex = element.getAttribute('data-latex');
            katex.render(latex, element, {
                throwOnError: false,
                displayMode: false
            });
        } catch (e) {
            console.error('LaTeX rendering error:', e);
        }
    });

    // Add loading state to form submission
    const form = document.querySelector('form');
    const submitButton = form?.querySelector('button[type="submit"]');
    const loadingDiv = document.querySelector('.loading');

    form?.addEventListener('submit', function(e) {
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Solving...';
        }
        if (loadingDiv) {
            loadingDiv.classList.add('active');
        }
    });

    // Copy solution button
    const copyButtons = document.querySelectorAll('.copy-solution');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const solutionText = this.closest('.solution-container').querySelector('.final-answer').textContent;
            navigator.clipboard.writeText(solutionText.trim()).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => button.textContent = 'Copy', 2000);
            });
        });
    });

    // Model status polling
    if (document.querySelector('.model-status')) {
        setInterval(checkModelStatus, 5000);
    }
});

// Check model warmup status
function checkModelStatus() {
    fetch('/warmup_status')
        .then(response => response.json())
        .then(data => {
            const statusEl = document.querySelector('.model-status');
            if (data.models_initialized) {
                statusEl.classList.remove('not-ready');
                statusEl.classList.add('ready');
                statusEl.textContent = 'Models Ready';
            } else {
                statusEl.classList.remove('ready');
                statusEl.classList.add('not-ready');
                statusEl.textContent = data.last_error ? 'Model Error' : 'Models Loading...';
            }
        })
        .catch(console.error);
}