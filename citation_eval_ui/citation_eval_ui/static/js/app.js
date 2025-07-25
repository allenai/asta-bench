// JavaScript for Citation Evaluation UI

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Auto-save functionality
    const autoSaveDelay = 30000; // 30 seconds
    let autoSaveTimer;
    
    function scheduleAutoSave() {
        clearTimeout(autoSaveTimer);
        autoSaveTimer = setTimeout(autoSave, autoSaveDelay);
    }
    
    function autoSave() {
        const form = document.getElementById('annotation-form');
        if (form) {
            // Check if any changes have been made
            const formData = new FormData(form);
            const hasChanges = checkForChanges(formData);
            if (hasChanges) {
                saveAnnotation(true); // true for auto-save
            }
        }
    }
    
    function checkForChanges(formData) {
        // Simple check - in a real app, you'd compare with initial state
        for (let [key, value] of formData.entries()) {
            if (key.startsWith('citation_') && value) {
                return true;
            }
            if ((key === 'missing_citations' || key === 'notes') && value.trim()) {
                return true;
            }
        }
        return false;
    }

    // Enhanced text highlighting
    function highlightClaimInText() {
        const textContent = document.getElementById('text-content');
        const claimText = window.currentClaimText;
        
        if (textContent && claimText) {
            let content = textContent.innerHTML;
            
            // More sophisticated text matching
            const escapeRegExp = (string) => {
                return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            };
            
            const escapedClaim = escapeRegExp(claimText);
            const regex = new RegExp(`(${escapedClaim})`, 'gi');
            
            content = content.replace(regex, '<span class="claim-highlight">$1</span>');
            textContent.innerHTML = content;
        }
    }
    
    // Citation highlighting based on current assessments
    function highlightCitations() {
        const textContent = document.getElementById('text-content');
        if (!textContent) {
            console.log('No text-content element found');
            return;
        }
        
        const supportingCitations = window.currentSupportingCitations || [];
        const nonSupportingCitations = window.currentNonSupportingCitations || [];

        let content = textContent.innerHTML;
        
        // Highlight supporting citations
        supportingCitations.forEach(citation => {
            const escapedCitation = citation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${escapedCitation})\\b`, 'g');
            content = content.replace(regex, '<span class="citation-supporting">$1</span>');
        });
        
        // Highlight non-supporting citations
        nonSupportingCitations.forEach(citation => {
            const escapedCitation = citation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${escapedCitation})\\b`, 'g');
            content = content.replace(regex, '<span class="citation-non-supporting">$1</span>');
        });
        
        textContent.innerHTML = content;
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Skip if user is typing in a form field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                const prevClaimBtn = document.querySelector('a[href*="claim=' + (parseInt(window.currentClaim) - 1) + '"]');
                if (prevClaimBtn) prevClaimBtn.click();
                break;
                
            case 'ArrowRight':
                e.preventDefault();
                const nextClaimBtn = document.querySelector('a[href*="claim=' + (parseInt(window.currentClaim) + 1) + '"]');
                if (nextClaimBtn) nextClaimBtn.click();
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                const prevRowBtn = document.querySelector('a[href*="row=' + (parseInt(window.currentRow) - 1) + '"]');
                if (prevRowBtn) prevRowBtn.click();
                break;
                
            case 'ArrowDown':
                e.preventDefault();
                const nextRowBtn = document.querySelector('a[href*="row=' + (parseInt(window.currentRow) + 1) + '"]');
                if (nextRowBtn) nextRowBtn.click();
                break;
                
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    const saveBtn = document.querySelector('button[type="submit"]');
                    if (saveBtn) saveBtn.click();
                }
                break;
        }
    });

    // Form change detection for auto-save
    const form = document.getElementById('annotation-form');
    if (form) {
        form.addEventListener('change', scheduleAutoSave);
        form.addEventListener('input', scheduleAutoSave);
    }

    // Enhanced save annotation function
    window.saveAnnotation = async function(isAutoSave = false) {
        const form = document.getElementById('annotation-form');
        const status = document.getElementById('save-status');
        
        if (!form || !status) return;
        
        const statusMessage = isAutoSave ? 'Auto-saving...' : 'Saving...';
        status.innerHTML = `<div class="alert alert-info">${statusMessage}</div>`;
        
        try {
            // Collect citation labels
            const citationLabels = {};
            const citations = window.currentCitations || [];
            
            citations.forEach((citationId, index) => {
                const radios = document.getElementsByName(`citation_${index}`);
                for (const radio of radios) {
                    if (radio.checked) {
                        // Normalize citation ID by removing surrounding parentheses/brackets
                        let normalizedId = citationId.trim();
                        if ((normalizedId.startsWith('(') && normalizedId.endsWith(')')) ||
                            (normalizedId.startsWith('[') && normalizedId.endsWith(']'))) {
                            normalizedId = normalizedId.slice(1, -1).trim();
                        }
                        citationLabels[normalizedId] = radio.value;
                        break;
                    }
                }
            });
            
            const formData = new FormData();
            formData.append('filename', form.filename.value);
            formData.append('row', form.row.value);
            formData.append('claim', form.claim.value);
            formData.append('citation_labels', JSON.stringify(citationLabels));
            formData.append('missing_citations', form.missing_citations.value);
            formData.append('notes', form.notes.value);
            
            const response = await fetch('/api/save-annotation', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                const successMessage = isAutoSave ? 'Auto-saved' : 'Annotation saved successfully!';
                status.innerHTML = `<div class="alert alert-success">${successMessage}</div>`;
                setTimeout(() => {
                    status.innerHTML = '';
                }, isAutoSave ? 2000 : 3000);
            } else {
                status.innerHTML = '<div class="alert alert-danger">Error: ' + (result.message || 'Unknown error') + '</div>';
            }
            
        } catch (error) {
            const errorMessage = isAutoSave ? 'Auto-save failed: ' : 'Error: ';
            status.innerHTML = `<div class="alert alert-danger">${errorMessage}${error.message}</div>`;
        }
    };

    // Progress indication
    function updateProgress() {
        const progressInfo = document.querySelector('.progress-info');
        if (progressInfo && window.currentClaim !== undefined && window.totalClaims !== undefined) {
            const percentage = Math.round(((parseInt(window.currentClaim) + 1) / parseInt(window.totalClaims)) * 100);
            
            // Add a progress bar if it doesn't exist
            let progressBar = progressInfo.querySelector('.progress');
            if (!progressBar) {
                const progressContainer = document.createElement('div');
                progressContainer.className = 'mt-3';
                progressContainer.innerHTML = `
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar" role="progressbar" style="width: ${percentage}%"></div>
                    </div>
                    <small class="text-light">Claim progress: ${percentage}%</small>
                `;
                progressInfo.appendChild(progressContainer);
            } else {
                const bar = progressBar.querySelector('.progress-bar');
                if (bar) bar.style.width = percentage + '%';
            }
        }
    }

    // Initialize function to be called after variables are set
    window.initializeAnnotations = function() {
        highlightClaimInText();
        highlightCitations();
        // updateProgress();
    }
    
    // Check if we're on the annotate page and variables are already set
    if (window.currentSupportingCitations !== undefined) {
        window.initializeAnnotations();
    }

    // Show keyboard shortcuts help
    function showKeyboardHelp() {
        const helpModal = `
            <div class="modal fade" id="helpModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Keyboard Shortcuts</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <ul class="list-unstyled">
                                <li><kbd>←</kbd> Previous claim</li>
                                <li><kbd>→</kbd> Next claim</li>
                                <li><kbd>↑</kbd> Previous row</li>
                                <li><kbd>↓</kbd> Next row</li>
                                <li><kbd>Ctrl+S</kbd> Save annotation</li>
                                <li><kbd>?</kbd> Show this help</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        if (!document.getElementById('helpModal')) {
            document.body.insertAdjacentHTML('beforeend', helpModal);
        }
    }

    // Help shortcut
    document.addEventListener('keydown', function(e) {
        if (e.key === '?' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            showKeyboardHelp();
            if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
                const modal = new bootstrap.Modal(document.getElementById('helpModal'));
                modal.show();
            }
        }
    });

    // Sticky claim functionality
    function setupStickyClaimBehavior() {
        const claimElement = document.getElementById('current-claim');
        const placeholder = document.getElementById('claim-placeholder');
        
        if (!claimElement || !placeholder) return;
        
        // Store the original position
        const originalTop = claimElement.offsetTop;
        let isSticky = false;
        
        function handleScroll() {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            if (scrollTop >= originalTop && !isSticky) {
                // Make it sticky
                const rect = claimElement.getBoundingClientRect();
                placeholder.style.height = rect.height + 'px';
                placeholder.classList.add('active');
                claimElement.classList.add('is-sticky');
                isSticky = true;
            } else if (scrollTop < originalTop && isSticky) {
                // Return to normal
                placeholder.classList.remove('active');
                claimElement.classList.remove('is-sticky');
                isSticky = false;
            }
        }
        
        // Throttle scroll events for better performance
        let ticking = false;
        function onScroll() {
            if (!ticking) {
                requestAnimationFrame(function() {
                    handleScroll();
                    ticking = false;
                });
                ticking = true;
            }
        }
        
        window.addEventListener('scroll', onScroll);
        
        // Initial check
        handleScroll();
    }
    
    // Initialize sticky behavior
    setupStickyClaimBehavior();

    console.log('Citation Evaluation UI initialized');
    console.log('Press ? for keyboard shortcuts');
});
