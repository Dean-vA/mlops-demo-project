// Error Component (js/components/error.js)
class ErrorComponent {
    constructor() {
        this.container = document.getElementById('error-component');
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="error" id="error-message" style="display: none;">
                <div class="error-content">
                    <span class="error-icon">⚠️</span>
                    <span class="error-text" id="error-text"></span>
                </div>
                <button class="error-close" id="error-close">×</button>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            errorMessage: this.container.querySelector('#error-message'),
            errorText: this.container.querySelector('#error-text'),
            errorClose: this.container.querySelector('#error-close')
        };
    }

    setupEventListeners() {
        this.elements.errorClose.addEventListener('click', () => {
            this.hide();
        });

        document.addEventListener('showError', (event) => {
            this.show(event.detail.message);
        });

        // Auto-hide after 10 seconds
        let autoHideTimer = null;
        this.autoHide = () => {
            if (autoHideTimer) clearTimeout(autoHideTimer);
            autoHideTimer = setTimeout(() => this.hide(), 10000);
        };
    }

    show(message, type = 'error') {
        this.elements.errorText.textContent = message;
        this.elements.errorMessage.className = `error error-${type}`;
        this.elements.errorMessage.style.display = 'block';
        this.autoHide();
    }

    hide() {
        this.elements.errorMessage.style.display = 'none';
    }
}
