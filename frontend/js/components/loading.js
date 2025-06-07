// Loading Component (js/components/loading.js)
class LoadingComponent {
    constructor() {
        this.container = document.getElementById('loading-component');
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="loading" id="loading" style="display: none;">
                <div class="spinner"></div>
                <span id="loading-text">Processing your request...</span>
                <div class="loading-progress" id="loading-progress" style="margin-top: 10px; display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
                    </div>
                    <span id="progress-text">0%</span>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            loading: this.container.querySelector('#loading'),
            loadingText: this.container.querySelector('#loading-text'),
            loadingProgress: this.container.querySelector('#loading-progress'),
            progressFill: this.container.querySelector('#progress-fill'),
            progressText: this.container.querySelector('#progress-text')
        };
    }

    setupEventListeners() {
        document.addEventListener('uploadProgress', (event) => {
            this.updateProgress(event.detail.progress);
        });
    }

    show(message = 'Processing your request...') {
        this.elements.loadingText.textContent = message;
        this.elements.loading.style.display = 'flex';
        this.hideProgress();
    }

    hide() {
        this.elements.loading.style.display = 'none';
        this.hideProgress();
    }

    updateProgress(progress) {
        if (progress > 0) {
            this.elements.loadingProgress.style.display = 'block';
            this.elements.progressFill.style.width = `${progress}%`;
            this.elements.progressText.textContent = `${Math.round(progress)}%`;
        }
    }

    hideProgress() {
        this.elements.loadingProgress.style.display = 'none';
        this.elements.progressFill.style.width = '0%';
        this.elements.progressText.textContent = '0%';
    }
}
