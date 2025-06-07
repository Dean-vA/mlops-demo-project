// Status Indicator Component (js/components/status-indicator.js)
class StatusIndicatorComponent {
    constructor() {
        this.container = document.getElementById('status-indicator-component');
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="status-indicator" id="status-indicator">Checking API status...</div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            statusIndicator: this.container.querySelector('#status-indicator')
        };
    }

    setupEventListeners() {
        document.addEventListener('apiHealthChanged', (event) => {
            this.updateStatus(event.detail.status, event.detail.message);
        });
    }

    updateStatus(status, message) {
        this.elements.statusIndicator.textContent = message;
        this.elements.statusIndicator.className = `status-indicator status-${status}`;
    }
}
