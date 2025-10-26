// Handle LaTeX rendering
document.addEventListener('DOMContentLoaded', function() {
    // Render all elements with data-latex attribute
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
});

// Add loading state to form
document.querySelector('.problem-form')?.addEventListener('submit', function(e) {
    const button = this.querySelector('button[type="submit"]');
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Solving...';
});